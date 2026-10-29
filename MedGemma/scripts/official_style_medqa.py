"""Pinned Google MedQA notebook recipe, adapted to local Transformers/NF4.
Exact upstream prompt, system message, and extract_answer; no test-based tuning.
"""
import os
from pathlib import Path
import os
os.environ.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,random,sys,time,traceback
import torch
from transformers import AutoTokenizer,StoppingCriteria,StoppingCriteriaList
from peft import PeftModel
from paired_evaluation import load_model
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resume_mixture_evaluation import reliable_atomic as save
OUT=ROOT/'medqa-official-style-v1';OLD=ROOT/'mixture-evaluation-v1'
MODELS=['Base','A','B','C'];SYSTEM='SYSTEM INSTRUCTION: think silently if needed.'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def recipe():
 spec=importlib.util.spec_from_file_location('google_official_medqa_recipe',OUT/'sources/official_recipe.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def prepare():
 assert not (OUT/'protocol.json').exists(),'Protocol exists; do not overwrite.'
 sys.path.insert(0,str(ROOT/'.eval-deps'));import pyarrow.parquet as pq
 original=read(OLD/'protocol.json');upstream=read(OUT/'sources/upstream.json');dataset=read(OUT/'sources/dataset_repository.json')
 datasets={};(OUT/'datasets').mkdir(exist_ok=True)
 for split in ['dev','test']:
  raw=pq.read_table(OUT/'sources'/(split+'.parquet')).to_pylist()
  assert len(raw)=={'dev':1272,'test':1273}[split]
  normalized=[]
  for i,r in enumerate(raw):
   data=r['data'];assert data['Correct Option'] in 'ABCD' and len(data['Options'])==4
   assert data['Correct Answer']==data['Options'][data['Correct Option']]
   normalized.append(dict(sample_id=f'medqa:{split}:{i:05d}',source_id=r['id'],data=data))
  file=OUT/'datasets'/(split+'.jsonl');file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in normalized),encoding='utf-8')
  datasets[split]=dict(n=len(raw),sha256=sha(file),raw_sha256=sha(OUT/'sources'/(split+'.parquet')))
 # Verify every test question, option and gold against the previous normalized dataset.
 previous=rows(OLD/'datasets/medqa.jsonl');current=rows(OUT/'datasets/test.jsonl')
 assert all(a['question']==b['data']['Question'] and a['choices']==[b['data']['Options'][c] for c in 'ABCD'] and a['reference']==b['data']['Correct Option'] for a,b in zip(previous,current))
 tokenizer=AutoTokenizer.from_pretrained(original['base'],use_fast=True,local_files_only=True)
 rr=recipe();sample=rr.format_prompt(current[0])
 rendered=tokenizer.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=sample['prompt'])],tokenize=False,add_generation_prompt=True)
 assert rendered.count(SYSTEM)==1
 assert tokenizer(rendered,add_special_tokens=False)['input_ids']==tokenizer.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=sample['prompt'])],tokenize=True,add_generation_prompt=True)
 for m in 'ABC':assert sha(Path(original['models'][m]['checkpoint'])/'adapter_model.safetensors')==original['models'][m]['adapter_sha256']
 protocol=dict(recipe_name='Pinned Google official evaluate_on_medqa notebook; local hardware adaptation, not bit-exact official score reproduction',
  upstream=upstream,dataset_repository=dict(repo=dataset['repo'],revision=dataset['revision']),datasets=datasets,
  model_id=original['model_id'],revision=original['revision'],base=original['base'],models=original['models'],system_message=SYSTEM,
  prompt_function='Exact official_recipe.format_prompt; source extracted unchanged from pinned upstream notebook.',
  answer_parser='Exact official_recipe.extract_answer with upstream ANSWER_PATTERNS, including its ordered first regex match semantics. No fallback parser.',
  primary_scoring='Official parser result == source gold, including a parseable output at length cap, as in notebook. None/invalid counts incorrect; also report completed-output-only conservative correctness.',
  generation=dict(max_new_tokens=10000,max_context_tokens=10000,do_sample=False,num_beams=1,use_cache=True,cache_implementation='offloaded',
    pad_token_id=tokenizer.pad_token_id,eos_token_id=read(Path(original['base'])/'config.json')['eos_token_id']),
  max_length_policy='Effective max new tokens = min(10000,10000-input length), honoring notebook total context cap and requested output budget; input never truncated.',
  seed=42,tokenizer='Original snapshot AutoTokenizer(use_fast=True); official chat template; tokenize rendered text with add_special_tokens=False.',
  official_sampling=dict(temperature=0.0,max_tokens=10000,max_model_len=10000),
  adaptations=['Transformers4.57.1 instead of notebook vLLM0.17.1/Transformers4.57.3; batch1 instead of continuous batching.',
    'Same NF4 double quant/FP32 compute as the existing experiment; notebook does not request4bit. Not an unquantized equivalence claim.',
    'Offloaded KV cache for6GB RTX2060; original model/adapter weights unchanged.',
    'Fixed model revision; upstream notebook model revision not pinned.'],
  probe=dict(split='dev',indices=[0,1],purpose='Two fixed dev items for loading, full generation, EOS and finite outputs; no prompt/checkpoint selection by correctness.'),
  code_sha256={n:sha(ROOT/n) for n in ['scripts/official_style_medqa.py','scripts/report_official_style_medqa.py','run_official_style_medqa.py','scripts/paired_evaluation.py','resume_mixture_evaluation.py']},
  sources_sha256={p.name:sha(p) for p in (OUT/'sources').iterdir() if p.is_file()},
  limits=['Official-style adaptation only: numeric engine/precision differ; cannot promise69.1%.',
    'Prior test results already viewed; this is a post-hoc repaired protocol; prompt/parser/budget now frozen before new test outputs.',
    'No test prompt search, no answer retries, no forced choice decoding, no paid API, no upload.'])
 save(OUT/'protocol.json',protocol);save(OUT/'status.json',dict(status='PREPARED'))
 print('PREPARED: official formatter/system/parser;10000-token budget;dev1272/test1273;all1273 test items match old data.',flush=True)
def check():
 p=read(OUT/'protocol.json')
 for n,h in p['code_sha256'].items():assert sha(ROOT/n)==h,'Frozen source changed: '+n
 for n,h in p['sources_sha256'].items():assert sha(OUT/'sources'/n)==h,'Frozen upstream/data source changed.'
 for split,d in p['datasets'].items():assert sha(OUT/'datasets'/(split+'.jsonl'))==d['sha256']
 for m in 'ABC':assert sha(Path(p['models'][m]['checkpoint'])/'adapter_model.safetensors')==p['models'][m]['adapter_sha256']
 return p
class Progress(StoppingCriteria):
 def __init__(self,n,state):self.n=n;self.state=state;self.started=time.monotonic();self.last=self.started
 def __call__(self,input_ids,scores,**kwargs):
  now=time.monotonic()
  if now-self.last>=20:
   count=input_ids.shape[1]-self.n
   save(OUT/'status.json',dict(status='RUNNING',**self.state,generated_tokens=count,elapsed_seconds=now-self.started,pid=os.getpid()))
   print(self.state['model'],self.state['stage'],self.state['item'],'generated',count,'tokens',f'{now-self.started:.1f}s',flush=True);self.last=now
  return False
def infer(p,rr,model,tokenizer,row,state):
 item=rr.format_prompt(row)
 messages=[dict(role='system',content=p['system_message']),dict(role='user',content=item['prompt'])]
 rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 b=tokenizer(rendered,add_special_tokens=False,return_tensors='pt');n=b['input_ids'].shape[1]
 budget=min(p['generation']['max_new_tokens'],p['generation']['max_context_tokens']-n);assert budget>0,'Prompt exceeds official context cap.'
 save(OUT/'status.json',dict(status='RUNNING',**state,generated_tokens=0,effective_max_new_tokens=budget,pid=os.getpid()))
 started=time.monotonic();progress=Progress(n,state)
 with torch.inference_mode():
  generated=model.generate(**{k:v.to('cuda') for k,v in b.items()},max_new_tokens=budget,do_sample=False,num_beams=1,
   use_cache=True,cache_implementation=p['generation']['cache_implementation'],logits_to_keep=1,
   pad_token_id=p['generation']['pad_token_id'],eos_token_id=p['generation']['eos_token_id'],stopping_criteria=StoppingCriteriaList([progress]))
 ids=generated[0,n:].tolist();text=tokenizer.decode(ids,skip_special_tokens=True);answer=rr.extract_answer(text)
 capped=len(ids)==budget and ids[-1] not in p['generation']['eos_token_id']
 return dict(sample_id=row['sample_id'],source_id=row['source_id'],gold_answer=item['answer'],predicted_answer=answer,
  correct=answer==item['answer'],completed_correct=answer==item['answer'] and not capped,
  raw_output=tokenizer.decode(ids,skip_special_tokens=False),scored_text=text,input_ids=b['input_ids'][0].tolist(),
  generated_token_ids=ids,generated_tokens=len(ids),effective_max_new_tokens=budget,truncated=capped,
  finish_reason='length' if capped else 'eos',input_messages=messages,elapsed_seconds=time.monotonic()-started,
  peak_cuda_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=sha(OUT/'protocol.json'))
def model_for(p,m):
 torch.manual_seed(p['seed']);torch.cuda.reset_peak_memory_stats();model=load_model(p,'base')
 if m!='Base':model=PeftModel.from_pretrained(model,p['models'][m]['checkpoint'],is_trainable=False)
 model.eval();return model
def run(stage):
 p=check();rr=recipe();tokenizer=AutoTokenizer.from_pretrained(p['base'],local_files_only=True,use_fast=True)
 if stage=='test':assert read(OUT/'probe.json')['status']=='PASS' and read(OUT/'probe.json')['protocol_sha256']==sha(OUT/'protocol.json')
 data=rows(OUT/'datasets'/('dev.jsonl' if stage=='probe' else 'test.jsonl'))
 if stage=='probe':data=[data[i] for i in p['probe']['indices']]
 for m in (['Base'] if stage=='probe' else MODELS):
  directory=OUT/'predictions'/stage/m;directory.mkdir(parents=True,exist_ok=True)
  if len(list(directory.glob('*.json')))==len(data):continue
  model=model_for(p,m)
  for i,row in enumerate(data):
   file=directory/f'{i:06d}.json'
   if file.exists():assert read(file)['protocol_sha256']==sha(OUT/'protocol.json');continue
   result=infer(p,rr,model,tokenizer,row,dict(model=m,stage=stage,item=i+1,total=len(data)))
   result.update(model=m,checkpoint=p['models'][m]['checkpoint'],model_revision=p['revision'])
   save(file,result);print('SAVED',m,stage,f'{i+1}/{len(data)}','tokens',result['generated_tokens'],'finish',result['finish_reason'],'parsed',bool(result['predicted_answer']),flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 if stage=='probe':
  completed=[read(f) for f in sorted((OUT/'predictions/probe/Base').glob('*.json'))]
  assert len(completed)==2 and all(r['generated_tokens']>0 for r in completed)
  save(OUT/'probe.json',dict(status='PASS',protocol_sha256=sha(OUT/'protocol.json'),n=2,
   checks=[{k:r[k] for k in ['sample_id','generated_tokens','finish_reason','elapsed_seconds','peak_cuda_bytes']} for r in completed],
   note='Execution/long-generation preflight only; no selection by answer correctness. Unparsed/truncated cases retained.'))
 save(OUT/'status.json',dict(status='COMPLETE',stage=stage,protocol_sha256=sha(OUT/'protocol.json')))
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','probe','test','status']);arg=a.parse_args()
 if arg.stage=='prepare':prepare()
 elif arg.stage=='status':
  print(json.dumps(read(OUT/'status.json'),indent=2));print({m:len(list((OUT/'predictions/test'/m).glob('*.json'))) for m in MODELS})
 else:run(arg.stage)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(OUT/'last-error.json',dict(error=str(e),traceback=traceback.format_exc()));save(OUT/'status.json',dict(status='FAILED',error=str(e)));raise
