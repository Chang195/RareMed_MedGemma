"""Frozen, resumable binary-image diagnosis and paired full MedQA evaluation.

Reuses the validated image loader, processor and original NF4 model loader.
No optimizer, prompt tuning, uploads, dependency installation or model downloads.
"""
import os
from pathlib import Path
import argparse, csv, gc, hashlib, importlib.metadata as md, json, os, random, re, shutil, sys, time, traceback, subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoProcessor
from peft import PeftModel
from PIL import Image, features
from diagnosis_data import sha, read_jsonl
from mixture_data import load_messages
from paired_evaluation import load_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'mixture-evaluation-v1'
RUNS=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'runs/mixture-ratios-v1'))
DATA=ROOT/'RareMed_CADASIL_SFT_Mixture_Experiments'
QAOLD=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'runs/diagnosis-night-01/evaluation-v1'))
ARMS={'A':'dataset_A_20pct','B':'dataset_B_50pct','C':'dataset_C_80pct'}
MODELS=['Base','A','B','C']
CODE=['mixture_evaluate.py','scripts/mixture_evaluation.py','scripts/mixture_data.py','scripts/pipeline.py',
      'scripts/diagnosis_data.py','scripts/paired_evaluation.py','scripts/eval_scoring.py','MedGemma_SFT_Figures/plot_sft_results.py']
QA_PROMPT='Answer the following medical multiple-choice question. Choose the single best option. Reply with only A, B, C, or D.\n\n{question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def jhash(v): return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def norm(s): return ' '.join(re.findall(r'[a-z0-9]+',s.lower()))
def parse_binary(raw):
    s=raw.strip().strip('`* \n\t').strip().rstrip('.!。').strip()
    if re.fullmatch('CADASIL',s,re.I): return 'CADASIL','explicit_positive'
    if re.fullmatch(r'(?:non[\s-]*CADASIL|not\s+CADASIL)',s,re.I): return 'non-CADASIL','explicit_negative'
    if re.search(r'\b(may|might|possible|possibly|uncertain|cannot|could|differential|versus|vs|or)\b',s,re.I): return '', 'uncertain_or_multiple'
    if re.search(r'CADASIL',s,re.I): return '', 'mention_only'
    return '', 'unparsed'
def inputs_digest(protocol): return sha(OUT/'protocol.json')
def overlap_audit(qa):
    train={}
    for arm,folder in ARMS.items():
        for r in read_jsonl(DATA/folder/'train.jsonl'):
            train[r['sample_id']]=norm(json.dumps(r,ensure_ascii=False))
    shingles={}
    for sid,t in train.items():
        words=t.split()
        for i in range(max(0,len(words)-19)): shingles.setdefault(' '.join(words[i:i+20]),set()).add(sid)
    hits=[]
    for q in qa:
        t=norm(q['question']); words=t.split(); matches=set()
        for sid,v in train.items():
            if t and t in v: matches.add(sid)
        for i in range(max(0,len(words)-19)): matches.update(shingles.get(' '.join(words[i:i+20]),()))
        if matches: hits.append(dict(question_id=q['sample_id'],train_ids=sorted(matches)))
    report=dict(unique_train_records=len(train),medqa_questions=len(qa),hits=hits,
       method='Normalized full-question substring and exact contiguous 20-word shingles against all serialized SFT training fields; all A/B/C.',
       limitations='Does not exclude paraphrases, semantic overlap, or base pretraining contamination.',passed=not hits)
    atomic(OUT/'overlap-audit.json',report)
    assert not hits,'Potential MedQA/SFT overlaps require review; no inference started.'
    return report
def prepare():
    assert not (OUT/'protocol.json').exists(),'Protocol already frozen; use probe/run/status/report.'
    OUT.mkdir(exist_ok=True)
    fm=read(ROOT/'MedGemma_SFT_Figures/input_manifest.json')
    models=fm['models']; cfg=read(RUNS/'A/config.json'); base=Path(cfg['base'])
    inventory=read(base/'original-inventory.json')
    for name,meta in inventory['files'].items(): assert sha(base/name)==meta['sha256'],name
    print('Original pinned base snapshot hash verification PASS',flush=True)
    dataset_hashes={}; selected=[]
    for arm,folder in ARMS.items():
        c=read(RUNS/arm/'config.json'); best=read(RUNS/arm/'best.json')
        assert read(RUNS/arm/'status.json')['status']=='COMPLETE'
        assert c['model']==fm['model_id'] and c['revision']==fm['revision']
        assert best['step']==models[arm]['best_step']
        models[arm]['adapter_sha256']=sha(Path(models[arm]['checkpoint'])/'adapter_model.safetensors')
        assert models[arm]['checkpoint']==str(RUNS/arm/best['checkpoint'])
        dataset_hashes[arm]={}
        for split in ['train','val','test']:
            path=DATA/folder/(split+'.jsonl');dataset_hashes[arm][split]=sha(path)
            assert sha(path)==c['source_jsonl_sha256'][split]
            selected.extend((DATA/folder,r) for r in read_jsonl(path))
    assert len({v['val'] for v in dataset_hashes.values()})==1 and len({v['test'] for v in dataset_hashes.values()})==1
    for folder,row in selected:
        image=(folder/row['image']).resolve();assert image.parent==(DATA/'images').resolve()
        assert sha(image)==row['metadata']['image_sha256']
        with Image.open(image) as im: im.convert('RGB').load()
    qa_file=QAOLD/'datasets/medqa.jsonl';qa_spec=read(QAOLD/'protocol.json')['benchmarks']['medqa']
    assert sha(qa_file)==qa_spec['normalized_sha256']
    qa=read_jsonl(qa_file);assert len(qa)==1273 and all(r['reference'] in 'ABCD' and len(r['choices'])==4 for r in qa)
    audit=overlap_audit(qa)
    (OUT/'datasets').mkdir(exist_ok=True);shutil.copy2(qa_file,OUT/'datasets/medqa.jsonl')
    processor=AutoProcessor.from_pretrained(RUNS/'A/processor',use_fast=False,local_files_only=True)
    processor.save_pretrained(OUT/'processor')
    for arm in 'BC':
        other=AutoProcessor.from_pretrained(RUNS/arm/'processor',use_fast=False,local_files_only=True)
        assert other.image_processor.to_dict()==processor.image_processor.to_dict()
        assert other.chat_template==processor.chat_template
        assert other.tokenizer.get_vocab()==processor.tokenizer.get_vocab()
    question=read_jsonl(DATA/ARMS['A']/'val.jsonl')[0]['messages'][0]['content'][1]['text']
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),model_id=fm['model_id'],revision=fm['revision'],
      base=str(base),models=models,source_package=str(DATA),dataset_hashes=dataset_hashes,seed=42,
      processor_sha256={p.name:sha(p) for p in (OUT/'processor').iterdir() if p.is_file()},
      image_processing=processor.image_processor.to_dict(),chat_template=processor.chat_template,
      image_policy='Original PIL RGB via existing mixture_data.load_messages; no manual crop/arrow removal; processor preprocessing unchanged.',
      diagnosis_prompt=question,diagnosis_generation=dict(max_new_tokens=32,do_sample=False,num_beams=1,use_cache=True,
         pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=read(base/'config.json')['eos_token_id']),
      diagnosis_parser='Exact whole-answer CADASIL or non-CADASIL (case/terminal punctuation/outer markup normalized); explicit not CADASIL accepted negative. All other answers unresolved, never substring-positive.',
      unresolved_policy='Retain and count; withhold binary metrics for any unresolved model pending separately declared policy. No label or prompt revision based on test outputs.',
      qa_prompt_template=QA_PROMPT,medqa={**qa_spec,'normalized_file':str(OUT/'datasets/medqa.jsonl')},
      qa_scoring='Zero-shot constrained next-token A/B/C/D log probability; single-token alignment checked; argmax with A-D tie order; no free generation or CoT.',
      overlap_audit=audit,quantization='NF4 double quant',compute_dtype='float32',attention='sdpa',
      validation_gate='Complete 67-image validation per model before its 68-image test; validation does not alter checkpoint, prompt or generation.',
      code_sha256={name:sha(ROOT/name) for name in CODE},
      environment=dict(python=sys.version,packages={k:md.version(k) for k in ['torch','transformers','peft','bitsandbytes','Pillow','huggingface-hub']},
       jpeg_library=features.version('jpg'),gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda),
      no_upload=True,no_training=True,limits=fm['limitations'])
    for name in CODE:
        dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    atomic(OUT/'protocol.json',protocol)
    atomic(OUT/'status.json',dict(status='PREPARED',protocol_sha256=sha(OUT/'protocol.json')))
    print('PREPARE PASS: four checkpoints; shared test 68; full MedQA 1273; overlap audit PASS; protocol frozen.',flush=True)
def check():
    p=read(OUT/'protocol.json')
    for name,h in p['code_sha256'].items(): assert sha(ROOT/name)==h,'Frozen evaluator changed: '+name
    for name,h in p['processor_sha256'].items(): assert sha(OUT/'processor'/name)==h
    for m in 'ABC':
        assert sha(Path(p['models'][m]['checkpoint'])/'adapter_model.safetensors')==p['models'][m]['adapter_sha256']
        for split,h in p['dataset_hashes'][m].items(): assert sha(DATA/ARMS[m]/(split+'.jsonl'))==h
    assert sha(OUT/'datasets/medqa.jsonl')==p['medqa']['normalized_sha256']
    return p
def model_for(p,tag):
    random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.reset_peak_memory_stats()
    model=load_model(p,'base')
    if tag!='Base': model=PeftModel.from_pretrained(model,p['models'][tag]['checkpoint'],is_trainable=False)
    model.eval(); return model
def infer(p,model,processor,row,task):
    start=time.monotonic()
    if task.startswith('diagnosis'):
        folder=DATA/ARMS['A']; assert sha((folder/row['image']).resolve())==row['metadata']['image_sha256']
        messages=load_messages(row,folder,DATA,False)
        assert len(messages)==1 and messages[0]['role']=='user'
        # Assert only image + frozen user task text reached the processor.
        assert [c['type'] for c in messages[0]['content']]==['image','text']
        assert messages[0]['content'][1]['text']==p['diagnosis_prompt']
        batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
        ids=batch['input_ids'][0].tolist()
        with torch.inference_mode():
            generated=model.generate(**{k:v.to('cuda') for k,v in batch.items()},**p['diagnosis_generation'],logits_to_keep=1)
        tokens=generated[0,len(ids):].tolist();clean=processor.tokenizer.decode(tokens,skip_special_tokens=True)
        raw=processor.tokenizer.decode(tokens,skip_special_tokens=False)
        pred,category=parse_binary(clean)
        eos=p['diagnosis_generation']['eos_token_id']; eos=[eos] if isinstance(eos,int) else eos
        truncated=len(tokens)==p['diagnosis_generation']['max_new_tokens'] and tokens[-1] not in eos
        if truncated: pred,category='', 'truncated'
        result=dict(sample_id=row['sample_id'],y_true=row['binary_target'],y_pred=pred,parse_category=category,
          raw_output=raw,clean_output=clean,generated_token_ids=tokens,truncated=truncated,
          cluster_id=row['metadata']['split_group_id'],image_sha256=row['metadata']['image_sha256'],
          input_user_text=p['diagnosis_prompt'],source_article_id=row.get('source_article_id'),source_case_id=row.get('source_case_id'))
    else:
        text=p['qa_prompt_template'].format(question=row['question'],**dict(zip('ABCD',row['choices'])))
        messages=[dict(role='user',content=[dict(type='text',text=text)])]
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
        ids=batch['input_ids'][0].tolist();letter_ids=[processor.tokenizer(c,add_special_tokens=False)['input_ids'] for c in 'ABCD']
        assert all(len(v)==1 for v in letter_ids)
        letter_ids=[v[0] for v in letter_ids]
        for c,t in zip('ABCD',letter_ids): assert processor.tokenizer(rendered+c,add_special_tokens=False)['input_ids']==ids+[t]
        with torch.inference_mode(): logits=model(**{k:v.to('cuda') for k,v in batch.items()},logits_to_keep=1).logits[0,-1].float()
        assert torch.isfinite(logits).all(),'Nonfinite QA logits'
        logp=torch.log_softmax(logits,dim=-1)[letter_ids].cpu().tolist();pred='ABCD'[max(range(4),key=lambda i:logp[i])]
        result=dict(question_id=row['sample_id'],question_sha256=jhash(dict(question=row['question'],choices=row['choices'])),
          gold_answer=row['reference'],predicted_answer=pred,raw_output=pred,correct=pred==row['reference'],
          output_type='constrained_next_token_choice_not_free_generation',choice_log_probabilities=dict(zip('ABCD',logp)),
          input_user_text=text,dataset_revision=p['medqa']['revision'])
    result.update(input_ids=ids,elapsed_seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    return result
def task_rows(task):
    return read_jsonl(OUT/'datasets/medqa.jsonl') if task=='medqa' else read_jsonl(DATA/ARMS['A']/('val.jsonl' if task.endswith('validation') else 'test.jsonl'))
def probe(p):
    processor=AutoProcessor.from_pretrained(OUT/'processor',use_fast=False,local_files_only=True)
    rows=task_rows('diagnosis-validation')
    picks=[next(r for r in rows if r['binary_target']==label) for label in ['CADASIL','non-CADASIL']]
    results=[]
    for tag in ['Base','A']:
        model=model_for(p,tag)
        for row in picks:
            r=infer(p,model,processor,row,'diagnosis-validation')
            results.append(dict(model=tag,task='validation-image',sample_id=r['sample_id'],input_tokens=len(r['input_ids']),
              output_tokens=len(r['generated_token_ids']),elapsed_seconds=r['elapsed_seconds'],peak_cuda_bytes=r['peak_cuda_bytes']))
        r=infer(p,model,processor,task_rows('medqa')[0],'medqa')
        results.append(dict(model=tag,task='qa-finite-logits-and-prefix-alignment',input_tokens=len(r['input_ids']),
           elapsed_seconds=r['elapsed_seconds'],peak_cuda_bytes=r['peak_cuda_bytes']))
        del model;gc.collect();torch.cuda.empty_cache()
    atomic(OUT/'probe.json',dict(status='PASS',protocol_sha256=sha(OUT/'protocol.json'),checks=results,
       note='Validation shape/output and finite QA scoring probe; no prompt/checkpoint tuning, no test scores inspected.'))
    print('PROBE PASS',json.dumps(results),flush=True)
def run(p,models):
    assert read(OUT/'probe.json')['protocol_sha256']==sha(OUT/'protocol.json')
    processor=AutoProcessor.from_pretrained(OUT/'processor',use_fast=False,local_files_only=True)
    digest=sha(OUT/'protocol.json')
    for tag in models:
        tasks=['diagnosis-validation','diagnosis-test','medqa']
        if all((OUT/'completed'/f'{tag}-{t}.json').exists() for t in tasks): continue
        model=model_for(p,tag)
        for task in tasks:
            if task=='diagnosis-test': assert (OUT/'completed'/f'{tag}-diagnosis-validation.json').exists()
            directory=OUT/'predictions'/tag/task;directory.mkdir(parents=True,exist_ok=True)
            rows=task_rows(task);t0=time.monotonic();new_n=0
            for i,row in enumerate(rows):
                dest=directory/f'{i:06d}.json';sid=row.get('sample_id')
                if dest.exists():
                    r=read(dest);assert r['protocol_id']==digest and r.get('sample_id',r.get('question_id'))==sid
                    continue
                atomic(OUT/'status.json',dict(status='RUNNING',model=tag,task=task,item=i+1,total=len(rows),pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat()))
                r=infer(p,model,processor,row,task);new_n+=1
                r.update(model=tag,checkpoint=p['models'][tag]['checkpoint'],model_revision=p['revision'],model_id=p['model_id'],
                   protocol_id=digest,adapter_sha256=p['models'][tag].get('adapter_sha256'),task=task)
                atomic(dest,r)
                if i==0 or (i+1)%10==0 or i+1==len(rows):
                    print(tag,task,f'{i+1}/{len(rows)}',f'{(time.monotonic()-t0)/new_n:.2f}s/new-item',flush=True)
            assert len(list(directory.glob('*.json')))==len(rows)
            atomic(OUT/'completed'/f'{tag}-{task}.json',dict(n=len(rows),protocol_sha256=digest))
            print('COMPLETE',tag,task,len(rows),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    all_done=all((OUT/'completed'/f'{m}-{t}.json').exists() for m in MODELS for t in ['diagnosis-validation','diagnosis-test','medqa'])
    if all_done:
        export_results(p)
        subprocess.run([str(ROOT/'.figure-build/.venv/Scripts/python.exe'),str(ROOT/'MedGemma_SFT_Figures/plot_sft_results.py'),
          '--manifest',str(OUT/'figure_input_manifest.json'),'--outdir',str(ROOT/'MedGemma_SFT_Figures/actual_results')],check=True)
    atomic(OUT/'status.json',dict(status='COMPLETE' if all_done else 'PARTIAL',protocol_sha256=digest,updated_utc=datetime.now(timezone.utc).isoformat()))
def csv_write(file,rows,fields):
    with file.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def export_results(p):
    diag=[];qa=[]
    for tag in MODELS:
        for task,dest in [('diagnosis-test',diag),('medqa',qa)]:
            assert (OUT/'completed'/f'{tag}-{task}.json').exists(),'Cannot export partial four-model comparison.'
            dest.extend(read(f) for f in sorted((OUT/'predictions'/tag/task).glob('*.json')))
    csv_write(OUT/'diagnostic_predictions.csv',diag,['model','sample_id','y_true','y_pred','cluster_id','image_sha256','raw_output','parse_category','checkpoint','protocol_id','model_revision'])
    csv_write(OUT/'medqa_item_results.csv',qa,['model','question_id','question_sha256','gold_answer','predicted_answer','raw_output','checkpoint','protocol_id','dataset_revision','model_revision'])
    reference=[dict(question_id=r['sample_id'],question_sha256=jhash(dict(question=r['question'],choices=r['choices'])),gold_answer=r['reference']) for r in task_rows('medqa')]
    csv_write(OUT/'medqa_reference.csv',reference,['question_id','question_sha256','gold_answer'])
    manifest=read(ROOT/'MedGemma_SFT_Figures/input_manifest.json');digest=sha(OUT/'protocol.json')
    manifest.update(diagnostic_protocol_id=digest,medqa_protocol_id=digest,medqa_dataset_revision=p['medqa']['revision'],
      medqa_expected_n=1273,medqa_sft_overlap_audit_passed=True,medqa_reference_file=str(OUT/'medqa_reference.csv'),
      reference_file=str(ROOT/'MedGemma_SFT_Figures/test_reference.csv'),diagnostic_results=str(OUT/'diagnostic_predictions.csv'),medqa_results=str(OUT/'medqa_item_results.csv'))
    atomic(OUT/'figure_input_manifest.json',manifest)
    print('EXPORT PASS: normalized four-model prediction CSVs and figure_input_manifest.json',flush=True)
def status():
    if (OUT/'status.json').exists(): print(json.dumps(read(OUT/'status.json'),indent=2))
    for tag in MODELS:
        print(tag,{task:len(list((OUT/'predictions'/tag/task).glob('*.json'))) for task in ['diagnosis-validation','diagnosis-test','medqa']})
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','probe','run','status','report'])
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    a=parser.parse_args()
    if a.stage=='prepare': prepare();return
    if a.stage=='status':status();return
    p=check()
    if a.stage=='report':export_results(p);return
    # Prevent two workers from loading models simultaneously on this 6-GB GPU.
    lock=OUT/'worker.lock'
    if lock.exists():
        import ctypes
        previous=read(lock);kernel=ctypes.windll.kernel32
        handle=kernel.OpenProcess(0x1000,False,previous['pid'])
        active=False
        if handle:
            code=ctypes.c_ulong();kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
            active=code.value==259
        assert not active, f"Evaluation worker already active: PID {previous['pid']}"
        lock.unlink()
    with lock.open('x',encoding='utf-8') as f: json.dump(dict(pid=os.getpid(),stage=a.stage),f)
    try:
        if a.stage=='probe':probe(p)
        else:run(p,a.models)
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid(): lock.unlink()
if __name__=='__main__':
    try: main()
    except BaseException as e:
        if not isinstance(e,SystemExit) or e.code:
            if OUT.exists():
                error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat())
                atomic(OUT/'last-error.json',error)
                if not isinstance(e,AssertionError) or 'already active' not in str(e):
                    atomic(OUT/'status.json',dict(status='FAILED',error=str(e),pid=os.getpid(),utc=error['utc']))
        raise
