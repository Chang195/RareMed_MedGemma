"""Frozen source-label GBM/astrocytoma evaluation, queued after MedQA.
No training, downloads, clinical relabeling, or alteration of existing experiments.
"""
import os
from pathlib import Path
import os
os.environ.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
from pathlib import Path
import argparse,gc,hashlib,json,re,sys,time
import torch
from transformers import AutoProcessor
from peft import PeftModel
from PIL import Image
from paired_evaluation import load_model
from mixture_data import load_messages
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resume_mixture_evaluation import reliable_atomic as save
OUT=ROOT/'tumor-binary-evaluation-v1';OLD=ROOT/'mixture-evaluation-v1';DATA=ROOT/'RareMed_CADASIL_SFT_Mixture_Experiments';FOLDER=DATA/'dataset_A_20pct'
MODELS=['Base','A','B','C']
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
TASKS={
 'gbm':dict(positive='GBM',negative='non-GBM',prompt='Classify this brain imaging figure as GBM (glioblastoma) or non-GBM. Reply with exactly one label: GBM or non-GBM. Do not include explanations or any other text.',
   source_positive='Source diagnosis explicitly says glioblastoma',excluded='Source diagnosis explicitly says astrocytoma; NOS/high-grade labels not assumed definitely non-GBM'),
 'astrocytoma':dict(positive='astrocytoma',negative='non-astrocytoma',prompt='Classify this brain imaging figure as astrocytoma or non-astrocytoma. Reply with exactly one label: astrocytoma or non-astrocytoma. Do not include explanations or any other text.',
   source_positive='Source diagnosis explicitly says astrocytoma',excluded='Source diagnosis explicitly says glioblastoma; not assumed a clinically negative astrocytic tumor')}
def family(row):
 s=row['diagnosis'].casefold()
 if re.search(r'\bglioblastoma\b',s):return 'gbm'
 if re.search(r'\bastrocytoma\b',s):return 'astrocytoma'
 return 'other'
def eligible(task,row):return family(row)!=('astrocytoma' if task=='gbm' else 'gbm')
def ref(task,row):return TASKS[task]['positive'] if family(row)==task else TASKS[task]['negative']
def prepare():
 OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Already frozen.'
 old=read(OLD/'protocol.json');test=rows(FOLDER/'test.jsonl');train=rows(FOLDER/'train.jsonl')
 for arm,folder in [('A','dataset_A_20pct'),('B','dataset_B_50pct'),('C','dataset_C_80pct')]:
  assert sha(DATA/folder/'test.jsonl')==old['dataset_hashes'][arm]['test']
  assert sha(Path(old['models'][arm]['checkpoint'])/'adapter_model.safetensors')==old['models'][arm]['adapter_sha256']
 seen=set()
 for r in test+train:
  file=(FOLDER/r['image']).resolve();assert file.parent==(DATA/'images').resolve()
  if file in seen:continue
  assert sha(file)==r['metadata']['image_sha256']
  with Image.open(file) as im:im.convert('RGB').load()
  seen.add(file)
 summaries={};references={};probes={};exclusions={}
 for task in TASKS:
  chosen=[r for r in test if eligible(task,r)]
  references[task]=[dict(sample_id=r['sample_id'],reference=ref(task,r),source_diagnosis=r['diagnosis'],
    cluster_id=r['metadata']['split_group_id'],source_case_id=r.get('source_case_id'),image_sha256=r['metadata']['image_sha256']) for r in chosen]
  exclusions[task]=[dict(sample_id=r['sample_id'],source_diagnosis=r['diagnosis'],reason=TASKS[task]['excluded']) for r in test if not eligible(task,r)]
  pos=[r for r in chosen if family(r)==task];neg=[r for r in chosen if family(r)!=task]
  summaries[task]=dict(n=len(chosen),positive=len(pos),negative=len(neg),positive_groups=len({r['metadata']['split_group_id'] for r in pos}),
    total_groups=len({r['metadata']['split_group_id'] for r in chosen}),excluded_n=len(exclusions[task]))
  ordered=sorted(train,key=lambda r:r['sample_id'])
  probes[task]=[next(r['sample_id'] for r in ordered if family(r)==task),next(r['sample_id'] for r in ordered if family(r)=='other')]
 assert summaries['gbm']['positive']==8 and summaries['astrocytoma']['positive']==2
 assert summaries['gbm']['n']==66 and summaries['astrocytoma']['n']==60
 protocol=dict(model_id=old['model_id'],revision=old['revision'],base=old['base'],models=old['models'],
  original_protocol_sha256=sha(OLD/'protocol.json'),tasks=TASKS,reference=references,excluded=exclusions,summaries=summaries,probes=probes,
  data_sha256={s:sha(FOLDER/(s+'.jsonl')) for s in ['train','test']},
  processor_sha256=old['processor_sha256'],generation=dict(max_new_tokens=32,do_sample=False,num_beams=1,use_cache=True,
    pad_token_id=old['diagnosis_generation']['pad_token_id'],eos_token_id=old['diagnosis_generation']['eos_token_id']),seed=42,
  parser='Remove declared EOS token IDs; strip surrounding whitespace/outer backticks or asterisks; casefold; exact frozen positive/negative aliases only. Truncated/uncertain/multiple/other outputs remain unresolved, never substring-positive.',
  aliases={'gbm':{'positive':['gbm','glioblastoma','glioblastoma multiforme'],'negative':['non-gbm','non glioblastoma','non-glioblastoma','not gbm']},
    'astrocytoma':{'positive':['astrocytoma'],'negative':['non-astrocytoma','non astrocytoma','not astrocytoma']}},
  image_policy='Same saved processor/PIL RGB original bytes; no manual crop/arrow removal. Task input is original image and new frozen user question only; no source text/metadata/history/reference.',
  unresolved_policy='Retain separately. Withhold main binary performance estimates if any unresolved outputs; report positive/negative unresolved counts.',
  confidence_interval='Class-stratified source-group percentile bootstrap,10000 repetitions,seed42; not verified patient IDs.',
  cohort_definition='Conservative source-label endpoints on fixed old test pool. Exclude cross-category GBM/astrocytoma records from each task rather than assume modern molecular taxonomy. Do not relabel originals.',
  limitations=['Source labels not independent clinical gold standards; astrocytoma NOS label not molecularly reclassified.',
    '8 GBM images/2 source groups;2 astrocytoma images/1 source group. Very limited independent positive support.',
    'Retrospective previously viewed holdout; all models share each task cohort; no prompt/checkpoint selection by these outputs.',
    'No claim of MRI-only sequence-matched cohort or preservation of all other diagnostic abilities.'],
  code_sha256={name:sha(ROOT/name) for name in ['scripts/tumor_binary_evaluation.py','scripts/report_tumor_binary.py','queue_tumor_after_medqa.py','scripts/paired_evaluation.py','scripts/mixture_data.py','scripts/pipeline.py']})
 save(OUT/'protocol.json',protocol);save(OUT/'status.json',dict(status='PREPARED',summaries=summaries));print(json.dumps(summaries,indent=2),flush=True)
def parse(task,text):
 s=text.strip().strip('`* \n\t').strip().rstrip('.!。').strip().casefold()
 for key in ['positive','negative']:
  if s in read(OUT/'protocol.json')['aliases'][task][key]:return TASKS[task][key],'explicit_'+key
 return '','unresolved'
def check():
 p=read(OUT/'protocol.json')
 for n,h in p['code_sha256'].items():assert sha(ROOT/n)==h,'Frozen source changed: '+n
 for s,h in p['data_sha256'].items():assert sha(FOLDER/(s+'.jsonl'))==h
 for name,h in p['processor_sha256'].items():assert sha(OLD/'processor'/name)==h
 for m in 'ABC':assert sha(Path(p['models'][m]['checkpoint'])/'adapter_model.safetensors')==p['models'][m]['adapter_sha256']
 return p
def infer(model,processor,p,task,row):
 assert sha((FOLDER/row['image']).resolve())==row['metadata']['image_sha256']
 messages=load_messages(row,FOLDER,DATA,False);assert len(messages)==1
 messages[0]['content'][1]['text']=p['tasks'][task]['prompt']
 assert [c['type'] for c in messages[0]['content']]==['image','text']
 b=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
 started=time.monotonic()
 with torch.inference_mode():g=model.generate(**{k:v.to('cuda') for k,v in b.items()},**p['generation'],logits_to_keep=1)
 ids=g[0,b['input_ids'].shape[1]:].tolist();trimmed=list(ids)
 while trimmed and trimmed[-1] in p['generation']['eos_token_id']:trimmed.pop()
 text=processor.tokenizer.decode(trimmed,skip_special_tokens=True);pred,category=parse(task,text)
 truncated=len(ids)==p['generation']['max_new_tokens'] and ids[-1] not in p['generation']['eos_token_id']
 if truncated:pred,category='','truncated'
 return dict(sample_id=row['sample_id'],reference=ref(task,row),prediction=pred,category=category,truncated=truncated,
  raw_output=processor.tokenizer.decode(ids,skip_special_tokens=False),clean_output=text,generated_token_ids=ids,input_ids=b['input_ids'][0].tolist(),
  input_user_text=p['tasks'][task]['prompt'],image_sha256=row['metadata']['image_sha256'],cluster_id=row['metadata']['split_group_id'],
  source_diagnosis=row['diagnosis'],elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
def run():
 p=check();digest=sha(OUT/'protocol.json');processor=AutoProcessor.from_pretrained(OLD/'processor',use_fast=False,local_files_only=True)
 train={r['sample_id']:r for r in rows(FOLDER/'train.jsonl')};test={r['sample_id']:r for r in rows(FOLDER/'test.jsonl')}
 for m in MODELS:
  if all((OUT/'completed'/f'{m}-{t}.json').exists() for t in TASKS):continue
  torch.manual_seed(p['seed']);torch.cuda.reset_peak_memory_stats();model=load_model(p,'base')
  if m!='Base':model=PeftModel.from_pretrained(model,p['models'][m]['checkpoint'],is_trainable=False)
  model.eval()
  for task in TASKS:
   probe=[]
   for sid in p['probes'][task]:probe.append(infer(model,processor,p,task,train[sid]))
   save(OUT/f'probe-{m}-{task}.json',dict(protocol_sha256=digest,results=probe))
   assert all(r['prediction'] for r in probe),'Train format probe failed; stop without altering prompt.'
   for i,reference in enumerate(p['reference'][task]):
    f=OUT/'predictions'/task/m/f'{i:06d}.json'
    if f.exists():assert read(f)['protocol_sha256']==digest;continue
    r=infer(model,processor,p,task,test[reference['sample_id']]);r.update(model=m,task=task,checkpoint=p['models'][m]['checkpoint'],model_revision=p['revision'],protocol_sha256=digest)
    save(f,r);save(OUT/'status.json',dict(status='RUNNING',model=m,task=task,completed=i+1,total=p['summaries'][task]['n']))
    if i==0 or (i+1)%10==0:print(m,task,f'{i+1}/{p["summaries"][task]["n"]}',flush=True)
   save(OUT/'completed'/f'{m}-{task}.json',dict(protocol_sha256=digest,n=p['summaries'][task]['n']))
  del model;gc.collect();torch.cuda.empty_cache()
 save(OUT/'status.json',dict(status='INFERENCE_COMPLETE',protocol_sha256=digest))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);arg=a.parse_args()
 if arg.stage=='prepare':prepare()
 else:run()
