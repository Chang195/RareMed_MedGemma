"""Finish first two preselected TRAIN questions under the original prompt.
Fixed extended budget; no prompt selection, no alteration of original experiment.
"""
import os
from pathlib import Path
import os
os.environ.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
from pathlib import Path
import hashlib,json,re,time
import torch
from transformers import AutoProcessor
from paired_evaluation import load_model
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'medqa-audit-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
 for i in range(30):
  try:os.replace(t,p);return
  except PermissionError:
   if i==29:raise
   time.sleep(.1)
basep=read(ROOT/'mixture-evaluation-v1/protocol.json');initial=read(OUT/'audit_protocol.json')
protocol=dict(source='first two already fixed training-split diagnostic questions; no selection by answer/correctness',
 selected_indices=initial['selected_indices'][:2],same_original_user_prompt=True,max_new_tokens=1536,do_sample=False,
 seed=42,quantization='same NF4 base as original',code_sha256=sha(Path(__file__)),
 original_protocol_sha256=sha(ROOT/'mixture-evaluation-v1/protocol.json'),
 parser='Explicit last Final Answer label, otherwise final isolated A-D line or explicit final answer phrase; else unresolved. Truncation never counts as a completed answer.')
if (OUT/'full-answer-protocol.json').exists():assert read(OUT/'full-answer-protocol.json')==protocol
else:save(OUT/'full-answer-protocol.json',protocol)
processor=AutoProcessor.from_pretrained(ROOT/'mixture-evaluation-v1/processor',local_files_only=True,use_fast=False)
model=load_model(basep,'base');model.eval();torch.manual_seed(42)
results=[]
for j in range(2):
 dest=OUT/f'full-answer-{j:02d}.json'
 if dest.exists():results.append(read(dest));continue
 previous=read(OUT/f'sample-{j:02d}.json')
 msg=[dict(role='user',content=[dict(type='text',text=previous['input_text'])])]
 batch=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
 assert batch['input_ids'][0].tolist()==previous['input_ids']
 started=time.monotonic()
 with torch.inference_mode():gen=model.generate(**{k:v.to('cuda') for k,v in batch.items()},max_new_tokens=1536,do_sample=False,num_beams=1,
   use_cache=True,logits_to_keep=1,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=basep['diagnosis_generation']['eos_token_id'])
 tokens=gen[0,batch['input_ids'].shape[1]:].tolist();kept=list(tokens)
 while kept and kept[-1] in basep['diagnosis_generation']['eos_token_id']:kept.pop()
 text=processor.tokenizer.decode(kept,skip_special_tokens=True)
 final=re.findall(r'Final\s*Answer\s*:\s*\**\s*\(?\s*([ABCD])\s*\)?',text,re.I)
 if not final:final=re.findall(r'(?:^|\n)\s*\**\(?([ABCD])\)?\**[.!]?\s*$',text)
 if not final:final=re.findall(r'(?:correct answer is|answer is|answer:)\s*\**\(?([ABCD])\)?',text,re.I)
 truncated=len(tokens)==1536 and tokens[-1] not in basep['diagnosis_generation']['eos_token_id']
 result=dict(source_index=previous['source_index'],gold=previous['gold'],original_bare_prediction=previous['first_step']['bare_prediction'],
  final_prediction=final[-1].upper() if final and not truncated else '',truncated=truncated,raw_output=processor.tokenizer.decode(tokens,skip_special_tokens=False),
  output_token_ids=tokens,output_tokens=len(tokens),input_ids=previous['input_ids'],elapsed_seconds=time.monotonic()-started,
  peak_cuda_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=sha(OUT/'full-answer-protocol.json'))
 save(dest,result);results.append(result)
 print('FULL ANSWER',j+1,'bare',result['original_bare_prediction'],'final',result['final_prediction'],'gold',result['gold'],
   'tokens',len(tokens),'truncated',truncated,'seconds',round(result['elapsed_seconds'],1),flush=True)
save(OUT/'full-answer-summary.json',dict(n=2,not_a_benchmark=True,results=[{k:r[k] for k in ['source_index','gold','original_bare_prediction','final_prediction','truncated','output_tokens','elapsed_seconds']} for r in results]))
