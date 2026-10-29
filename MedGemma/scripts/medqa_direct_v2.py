"""Separate direct-answer MedQA protocol; validate on fixed TRAIN items first.
Raw greedy generation, no candidate-logit forcing. Old results remain unchanged.
"""
import os
from pathlib import Path
import os
os.environ.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
from pathlib import Path
import argparse,collections,gc,hashlib,json,random,re,sys,time,traceback
import torch
from transformers import AutoProcessor
from peft import PeftModel
from paired_evaluation import load_model
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resume_mixture_evaluation import reliable_atomic as save
OUT=ROOT/'medqa-direct-v2';OLD=ROOT/'mixture-evaluation-v1'
MODELS=['Base','A','B','C']
INSTRUCTION='Select the single best answer to this medical multiple-choice question. Return exactly one uppercase letter from A, B, C, D. Do not include reasoning, explanations, markdown, punctuation, or any other text.'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def setup():
 OUT.mkdir(exist_ok=True)
 p=read(OLD/'protocol.json')
 source=ROOT/'medqa-audit-v1/datasets/source_train.jsonl'
 assert sha(source)=='3a65baf760f17c395058d6699315a7ce8aa767c50cb193fea87ceb94dd790324'
 train=rows(source);test=rows(OLD/'datasets/medqa.jsonl')
 prior=set(read(ROOT/'medqa-audit-v1/audit_protocol.json')['selected_indices']);testq={r['question'] for r in test}
 order=random.Random(20261028).sample(range(len(train)),len(train))
 selected=[i for i in order if i not in prior and train[i]['question'] not in testq][:32]
 spec=dict(original_protocol_sha256=sha(OLD/'protocol.json'),model_id=p['model_id'],revision=p['revision'],
  instruction=INSTRUCTION,selected_train_indices=selected,validation_n=32,validation_seed=20261028,
  validation_gate='Each model must have at least29/32 strictly valid single-letter completed responses (>=90%). Gate uses format only, never answer accuracy.',
  generation=dict(max_new_tokens=32,do_sample=False,num_beams=1,use_cache=True,pad_token_id=p['diagnosis_generation']['pad_token_id'],eos_token_id=p['diagnosis_generation']['eos_token_id']),
  parser='After removing only known terminal EOS token IDs and surrounding whitespace, exact one uppercase A/B/C/D. Everything else or a truncated output is invalid. Invalid counts incorrect in total accuracy and is reported separately.',
  no_hard_token_constraint=True,no_assistant_prefill=True,train_source_sha256=sha(source),test_source_sha256=sha(OLD/'datasets/medqa.jsonl'),
  code_sha256=sha(Path(__file__)),loader_code_sha256=sha(ROOT/'scripts/paired_evaluation.py'),
  disclosure='Post-hoc protocol repair after old test results; prompt checks use TRAIN items, not test selection. Not official CoT benchmark reproduction. One32-item format validation set is small and does not establish score calibration or all abilities retained.')
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==spec,'Frozen protocol/source changed.'
 else:save(OUT/'protocol.json',spec)
 validation=[dict(sample_id=f'medqa:format-train:{i}',source_index=i,question=train[i]['question'],choices=[train[i]['options'][c] for c in 'ABCD'],reference=train[i]['answer_idx']) for i in selected]
 return p,spec,validation,test
def model_for(p,m):
 torch.manual_seed(42);model=load_model(p,'base')
 if m!='Base':model=PeftModel.from_pretrained(model,p['models'][m]['checkpoint'],is_trainable=False)
 model.eval();return model
def infer(model,processor,row,spec):
 prompt=INSTRUCTION+'\n\n'+row['question']+'\n\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',row['choices']))
 msg=[dict(role='user',content=[dict(type='text',text=prompt)])]
 b=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
 start=time.monotonic()
 with torch.inference_mode():
  gen=model.generate(**{k:v.to('cuda') for k,v in b.items()},**spec['generation'],logits_to_keep=1,return_dict_in_generate=True,output_scores=True)
 ids=gen.sequences[0,b['input_ids'].shape[1]:].tolist();trimmed=list(ids)
 while trimmed and trimmed[-1] in spec['generation']['eos_token_id']:trimmed.pop()
 text=processor.tokenizer.decode(trimmed,skip_special_tokens=True).strip()
 truncated=len(ids)==spec['generation']['max_new_tokens'] and ids[-1] not in spec['generation']['eos_token_id']
 valid=bool(re.fullmatch('[ABCD]',text)) and not truncated;prediction=text if valid else ''
 first=gen.scores[0][0].float().softmax(-1);top=torch.argmax(first).item()
 bare=[processor.tokenizer(c,add_special_tokens=False)['input_ids'][0] for c in 'ABCD']
 return dict(sample_id=row['sample_id'],reference=row['reference'],prediction=prediction,correct=prediction==row['reference'],format_valid=valid,truncated=truncated,
  raw_output=processor.tokenizer.decode(ids,skip_special_tokens=False),clean_output=text,generated_token_ids=ids,input_ids=b['input_ids'][0].tolist(),
  input_user_text=prompt,elapsed_seconds=time.monotonic()-start,first_token_probability_mass_bare_letters=float(first[bare].sum()),
  first_generated_token_id=ids[0],full_vocab_first_argmax=top)
def report(stage):
 summary={}
 for m in MODELS:
  rs=[read(f) for f in sorted((OUT/'predictions'/stage/m).glob('*.json'))]
  if not rs:continue
  summary[m]=dict(n=len(rs),format_valid=sum(r['format_valid'] for r in rs),truncated=sum(r['truncated'] for r in rs),correct=sum(r['correct'] for r in rs),
    accuracy=sum(r['correct'] for r in rs)/len(rs),predicted_letters=dict(collections.Counter(r['prediction'] for r in rs)),
    mean_seconds=sum(r['elapsed_seconds'] for r in rs)/len(rs),
    mean_first_token_bare_letter_mass=sum(r['first_token_probability_mass_bare_letters'] for r in rs)/len(rs))
 return summary
def run(stage):
 p,spec,validation,test=setup();digest=sha(OUT/'protocol.json')
 if stage=='test':
  gate=read(OUT/'validation-report.json');assert gate['passed'],'Format validation failed; do not start full test.'
  assert gate['protocol_sha256']==digest
 items=validation if stage=='validation' else test
 processor=AutoProcessor.from_pretrained(OLD/'processor',use_fast=False,local_files_only=True)
 for m in MODELS:
  dest=OUT/'predictions'/stage/m;dest.mkdir(parents=True,exist_ok=True)
  if len(list(dest.glob('*.json')))==len(items):continue
  model=model_for(p,m)
  for i,row in enumerate(items):
   file=dest/f'{i:06d}.json'
   if file.exists():assert read(file)['protocol_sha256']==digest;continue
   r=infer(model,processor,row,spec);r.update(model=m,stage=stage,protocol_sha256=digest,checkpoint=p['models'][m]['checkpoint'])
   save(file,r)
   save(OUT/'status.json',dict(status='RUNNING',stage=stage,model=m,completed=i+1,total=len(items),pid=os.getpid()))
   if (i+1)%8==0 or i==0:print(stage,m,f'{i+1}/{len(items)}',flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 result=report(stage)
 if stage=='validation':
  passed=len(result)==4 and all(r['n']==32 and r['format_valid']>=29 for r in result.values())
  save(OUT/'validation-report.json',dict(protocol_sha256=digest,passed=passed,results=result,not_a_benchmark=True))
 else:save(OUT/'test-report.json',dict(protocol_sha256=digest,results=result))
 save(OUT/'status.json',dict(status='COMPLETE',stage=stage,protocol_sha256=digest))
 print(json.dumps(result,indent=2),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['validation','test','status']);arg=a.parse_args()
 if arg.stage=='status':
  print(json.dumps(read(OUT/'status.json'),indent=2));print(json.dumps(report('validation'),indent=2));return
 run(arg.stage)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'last-error.json',dict(error=str(e),traceback=traceback.format_exc()))
  raise
