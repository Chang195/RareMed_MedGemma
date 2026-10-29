"""Frozen paired diagnosis + full MedQA/MedMCQA/MMLU clinical evaluation.

Only downloads public benchmarks and performs inference. No optimizer or uploads.
"""
import argparse
from datetime import datetime,timezone
import gc
import json
import importlib.metadata as md
import os
from pathlib import Path
import random
import shutil
import sys
import time
from diagnosis_data import LOCAL,MODEL,REVISION,read_jsonl,sha,target,model_messages,write,audit
from eval_scoring import VERSION,parse_diagnosis,diagnosis_metrics,paired_qa,normalize
import torch
from transformers import AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
CODE=['evaluate.py','scripts/paired_evaluation.py','scripts/eval_scoring.py','scripts/diagnosis_data.py','scripts/pipeline.py']
BENCHMARKS={
 'medqa':dict(repo='GBaker/MedQA-USMLE-4-options',revision='0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18',file='phrases_no_exclude_test.jsonl',split='test',expected=1273),
 'medmcqa':dict(repo='openlifescienceai/medmcqa',revision='91c6572c454088bf71b679ad90aa8dffcd0d5868',file='data/validation-00000-of-00001.parquet',split='validation',expected=4183),
 'mmlu_clinical':dict(repo='cais/mmlu',revision='c30699e8356da336a370243923dbaf21066bb9fe',file='clinical_knowledge/test-00000-of-00001.parquet',split='test',expected=265)
}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic(path,data):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    write(temporary,data); os.replace(temporary,path)

def dataset_download(name,out):
    spec=BENCHMARKS[name]
    file=Path(hf_hub_download(spec['repo'],spec['file'],repo_type='dataset',revision=spec['revision']))
    if file.suffix=='.parquet':
        sys.path.insert(0,str(ROOT/'.eval-deps'))
        try: import pyarrow.parquet as pq
        except ImportError: raise RuntimeError('Run python evaluate.py deps first (isolated pyarrow, existing model environment unchanged)')
        raw=pq.read_table(file).to_pylist()
    else: raw=read_jsonl(file)
    rows=[]
    for i,r in enumerate(raw):
        if name=='medqa':
            options=[r['options'][x] for x in 'ABCD']; answer=r['answer_idx']; question=r['question']
        elif name=='medmcqa':
            options=[r['op'+x] for x in 'abcd']; answer='ABCD'[r['cop']]; question=r['question']
        else:
            options=list(r['choices']); answer='ABCD'[r['answer']]; question=r['question']
        assert len(options)==4 and answer in 'ABCD' and all(isinstance(x,str) and x.strip() for x in options)
        assert isinstance(question,str) and question.strip()
        rows.append(dict(sample_id=f'{name}:{spec["split"]}:{i:05d}',source_id=r.get('id'),question=question,
            choices=options,reference=answer,source_split=spec['split']))
    assert len(rows)==spec['expected'], f'{name} unexpected size {len(rows)}'
    dest=out/'datasets'/f'{name}.jsonl'; dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    return dict(**spec,raw_sha256=sha(file),normalized_file=str(dest),normalized_sha256=sha(dest),count=len(rows))

def prepare(training,out):
    assert not (out/'protocol.json').exists(), 'Protocol already frozen. Run evaluations or choose a new evaluation name before any testing.'
    config=read(training/'config.json'); status=read(training/'status.json')
    assert status['status']=='COMPLETE'
    best=read(training/'best.json'); checkpoint=training/best['checkpoint']
    assert best['step']==min(read(training/'validation-history.json'),key=lambda x:x['mean_sample_nll'])['step']
    assert config['model']==MODEL and config['revision']==REVISION
    base=Path(config['base']); data=Path(config['data'])
    inventory=read(base/'original-inventory.json')
    for name,meta in inventory['files'].items():
        assert sha(base/name)==meta['sha256'], 'Original model integrity failure: '+name
    print('Original pinned model files verified; smoke adapter excluded',flush=True)
    _,data_report=audit(data,out/'data-check')
    for name in ['train.jsonl','validation.jsonl','test.jsonl']:
        assert sha(data/name)==config['dataset_file_sha256'][name]
    processor=AutoProcessor.from_pretrained(training/'processor',use_fast=False,local_files_only=True)
    processor.save_pretrained(out/'processor')
    benchmarks={}
    for name in BENCHMARKS:
        benchmarks[name]=dataset_download(name,out)
        print('Benchmark prepared:',name,benchmarks[name]['count'],flush=True)
    other_labels=sorted({target(r) for split in ['train','validation'] for r in read_jsonl(data/(split+'.jsonl'))
        if r['metadata']['diagnosis_family']!='CADASIL_author_reported'})
    # Output rules are frozen before any test inference, never derived from test answers.
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),model=MODEL,revision=REVISION,base=str(base),
        checkpoint=str(checkpoint),checkpoint_step=best['step'],adapter_sha256=sha(checkpoint/'adapter_model.safetensors'),
        checkpoint_selection='training best.json selected only by validation mean sample loss',
        training_config_sha256=sha(training/'config.json'),data=str(data),data_file_sha256=data_report['package_file_sha256'],
        quantization='NF4 double quant',compute_dtype='float32',attention='sdpa',seed=42,
        diagnosis_prompt=read(data/'manifest.json')['question'],diagnosis_generation=dict(max_new_tokens=64,do_sample=False,num_beams=1,
            use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=read(base/'config.json')['eos_token_id']),
        diagnosis_parser_version=VERSION,other_label_vocabulary=other_labels,
        unresolved_policy='Keep uncertainty/multiple/mention/unparsed separate. Main binary metrics on resolved outputs, with counts/coverage and sensitivity/specificity full-sample bounds. Never silently count unresolved as negatives.',
        qa_prompt_template='Answer the following medical multiple-choice question. Choose the single best option. Reply with only A, B, C, or D.\n\n{question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
        qa_scoring='Zero-shot constrained next-token A/B/C/D log probability; one token per letter and exact chat-prefix alignment required; no CoT, no demonstrations, no generation/explanation parsing',
        benchmarks=benchmarks,noninferiority_margin=None,paired_bootstrap_repeats=10000,
        image_processing=processor.image_processor.to_dict(),chat_template=processor.chat_template,
        processor_file_sha256={f.name:sha(f) for f in (out/'processor').iterdir() if f.is_file()},
        environment=dict(python=sys.version,packages={d.metadata['Name']:d.version for d in md.distributions()},
            torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
        code_sha256={name:sha(ROOT/name) for name in CODE},no_training=True,no_upload=True,
        limits='Source diagnosis labels not clinical gold standards; test CADASIL=5 images/3 split groups. Public QA scores are knowledge-retention checks, not proof that all clinical or imaging abilities are preserved.')
    for name in CODE:
        dest=out/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    atomic(out/'protocol.json',protocol)
    print('PREPARE PASS: protocol frozen. No model predictions generated.',flush=True)

def check_protocol(out):
    p=read(out/'protocol.json')
    for name,digest in p['code_sha256'].items(): assert sha(ROOT/name)==digest, 'Evaluator changed after freeze: '+name
    assert sha(Path(p['checkpoint'])/'adapter_model.safetensors')==p['adapter_sha256']
    for name,digest in p['processor_file_sha256'].items(): assert sha(out/'processor'/name)==digest
    for split in ['validation','test']: assert sha(Path(p['data'])/(split+'.jsonl'))==p['data_file_sha256'][split+'.jsonl']
    for spec in p['benchmarks'].values(): assert sha(Path(spec['normalized_file']))==spec['normalized_sha256']
    return p

def load_model(p,tag):
    model=Gemma3ForConditionalGeneration.from_pretrained(p['base'],local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32),
        dtype=torch.float32,device_map={'':0},attn_implementation='sdpa')
    assert not hasattr(model,'peft_config')
    if tag=='sft': model=PeftModel.from_pretrained(model,p['checkpoint'],is_trainable=False)
    model.eval(); model.config.use_cache=False; model.config.text_config.use_cache=False
    return model

def task_key(task,split): return f'diagnosis-{split}' if task=='diagnosis' else task

def output_rows(out,key,tag):
    return [read(f) for f in sorted((out/'predictions'/key/tag).glob('*.json'))]

def run_task(out,p,model,processor,task,split,tag):
    key=task_key(task,split); done=out/'completed'/f'{key}-{tag}.json'
    if task=='diagnosis':
        if split=='test':
            gate=out/'completed'/f'diagnosis-validation-{tag}.json'
            assert gate.exists() and read(gate)['protocol_sha256']==sha(out/'protocol.json'), 'Complete matching validation run before fixed test inference'
        rows=read_jsonl(Path(p['data'])/(split+'.jsonl'))
    else: rows=read_jsonl(Path(p['benchmarks'][task]['normalized_file']))
    directory=out/'predictions'/key/tag; directory.mkdir(parents=True,exist_ok=True)
    digest=sha(out/'protocol.json')
    if task!='diagnosis':
        letter_ids=[processor.tokenizer(x,add_special_tokens=False)['input_ids'] for x in 'ABCD']
        assert all(len(x)==1 for x in letter_ids), 'Choice scorer requires one token per letter'
        letter_ids=[x[0] for x in letter_ids]
    for i,row in enumerate(rows):
        file=directory/f'{i:06d}.json'
        if file.exists():
            previous=read(file)
            assert previous['sample_id']==row['sample_id'] and previous['protocol_sha256']==digest
            continue
        started=time.monotonic()
        if task=='diagnosis':
            assert sha(Path(p['data'])/row['image'])==row['metadata']['image_sha256']
            messages=model_messages(row,Path(p['data']),include_answer=False)
            assert [m['role'] for m in messages]==['user']
            batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
            ids=batch['input_ids'][0].tolist()
            with torch.inference_mode():
                generated=model.generate(**{k:v.to('cuda') for k,v in batch.items()},**p['diagnosis_generation'])
            new=generated[0,len(ids):].tolist()
            raw=processor.tokenizer.decode(new,skip_special_tokens=False)
            clean=processor.tokenizer.decode(new,skip_special_tokens=True)
            truncated=len(new)==p['diagnosis_generation']['max_new_tokens'] and new[-1] not in p['diagnosis_generation']['eos_token_id']
            result=dict(reference=target(row),reference_cadasil=row['metadata']['diagnosis_family']=='CADASIL_author_reported',
                article_id=row['metadata']['article_id'],split_group_id=row['metadata']['split_group_id'],
                image=row['image'],image_sha256=row['metadata']['image_sha256'],input_user_text=p['diagnosis_prompt'],
                raw_output=raw,clean_output=clean,generated_token_ids=new,truncated=truncated,
                category='unparsed' if truncated else parse_diagnosis(clean,p['other_label_vocabulary']))
        else:
            text=p['qa_prompt_template'].format(question=row['question'],**dict(zip('ABCD',row['choices'])))
            messages=[dict(role='user',content=[dict(type='text',text=text)])]
            rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
            ids=batch['input_ids'][0].tolist()
            for letter,token in zip('ABCD',letter_ids):
                continued=processor.tokenizer(rendered+letter,add_special_tokens=False)['input_ids']
                assert continued==ids+[token], 'Choice-token boundary mismatch; stop rather than change scoring'
            with torch.inference_mode():
                logits=model(**{k:v.to('cuda') for k,v in batch.items()},logits_to_keep=1).logits[0,-1].float()
            assert torch.isfinite(logits).all(), 'Nonfinite MCQ logits'
            logp=torch.log_softmax(logits,dim=-1)[letter_ids].cpu().tolist()
            prediction='ABCD'[max(range(4),key=lambda j:logp[j])]
            result=dict(reference=row['reference'],prediction=prediction,correct=prediction==row['reference'],
                input_user_text=text,choice_log_probabilities=dict(zip('ABCD',logp)),source_split=row['source_split'],
                raw_output=prediction,output_type='constrained_next_token_choice_not_free_generation')
        result.update(sample_id=row['sample_id'],model_variant=tag,model_id=MODEL,model_revision=REVISION,
            adapter_sha256=p['adapter_sha256'] if tag=='sft' else None,task=key,protocol_sha256=digest,
            input_ids=ids,elapsed_seconds=time.monotonic()-started)
        atomic(file,result)
        if (i+1)%10==0 or i==0 or i+1==len(rows): print(tag,key,f'{i+1}/{len(rows)}',flush=True)
    saved=output_rows(out,key,tag)
    assert len(saved)==len(rows)
    dest=out/f'{key}-{tag}.jsonl'
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in saved),encoding='utf-8')
    atomic(done,dict(protocol_sha256=digest,samples=len(rows),output_sha256=sha(dest)))
    print('COMPLETE:',key,tag,len(rows),flush=True)

def report(out,p):
    result=dict(protocol_sha256=sha(out/'protocol.json'),diagnosis={},benchmarks={})
    for split in ['validation','test']:
        key=task_key('diagnosis',split)
        if not all((out/'completed'/f'{key}-{tag}.json').exists() for tag in ['base','sft']): continue
        base=output_rows(out,key,'base'); sft=output_rows(out,key,'sft')
        assert [r['sample_id'] for r in base]==[r['sample_id'] for r in sft]
        changes=[]
        for a,b in zip(base,sft):
            assert a['reference']==b['reference']
            def correct(r):
                return r['category']=='positive' if r['reference_cadasil'] else r['category'] in ['negative','other']
            ca,cb=correct(a),correct(b)
            changes.append(dict(sample_id=a['sample_id'],reference=a['reference'],reference_cadasil=a['reference_cadasil'],
                base_category=a['category'],sft_category=b['category'],base_output=a['clean_output'],sft_output=b['clean_output'],
                change='improved' if cb and not ca else 'regressed' if ca and not cb else 'unchanged',
                unresolved_counted_as_incorrect_for_change_table=True))
        result['diagnosis'][split]=dict(base=diagnosis_metrics(base),sft=diagnosis_metrics(sft),per_sample_changes=changes,
            CADASIL_articles=len({r['article_id'] for r in base if r['reference_cadasil']}),
            CADASIL_split_groups=len({r['split_group_id'] for r in base if r['reference_cadasil']}),
            patient_count='not independently established')
    for name in BENCHMARKS:
        if not all((out/'completed'/f'{name}-{tag}.json').exists() for tag in ['base','sft']): continue
        base=output_rows(out,name,'base'); sft=output_rows(out,name,'sft')
        result['benchmarks'][name]=paired_qa(base,sft)
    result['planned_qa_sets']=list(BENCHMARKS)
    result['pending_qa_sets']=[name for name in BENCHMARKS if name not in result['benchmarks']]
    result['limitations']=p['limits']+' QA bootstrap treats questions as resampling units; question dependencies and pretraining contamination are not eliminated.'
    atomic(out/'paired-report.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','run','report'])
    p.add_argument('--run',required=True); p.add_argument('--eval',required=True)
    p.add_argument('--task',choices=['diagnosis','mmlu_clinical','medqa','medmcqa','qa_all'],default='diagnosis')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--model',choices=['base','sft','both'],default='both')
    a=p.parse_args(); training=LOCAL/'runs'/a.run; out=training/a.eval
    if a.stage=='prepare': prepare(training,out); return
    protocol=check_protocol(out)
    if a.stage=='report': report(out,protocol); return
    processor=AutoProcessor.from_pretrained(out/'processor',use_fast=False,local_files_only=True)
    tasks=list(BENCHMARKS) if a.task=='qa_all' else [a.task]
    for tag in ['base','sft'] if a.model=='both' else [a.model]:
        random.seed(protocol['seed']); torch.manual_seed(protocol['seed'])
        model=load_model(protocol,tag)
        for task in tasks: run_task(out,protocol,model,processor,task,a.split,tag)
        del model; gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__': main()
