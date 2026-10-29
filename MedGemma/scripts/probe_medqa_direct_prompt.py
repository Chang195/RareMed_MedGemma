"""TRAIN-only format probe: strong instruction vs assistant answer prefill.
No original test prompt, protocol, predictions or model weights are modified.
"""
import os
from pathlib import Path
import os
os.environ.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
from pathlib import Path
import json,re,time,hashlib,collections
import torch
from transformers import AutoProcessor
from paired_evaluation import load_model
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'medqa-audit-v1/direct-prompt-probe';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
 for i in range(30):
  try:os.replace(t,p);return
  except PermissionError:
   if i==29:raise
   time.sleep(.1)
protocol=read(ROOT/'mixture-evaluation-v1/protocol.json')
records=[read(ROOT/f'medqa-audit-v1/sample-{i:02d}.json') for i in range(8)]
instruction='Select the single best answer to this medical multiple-choice question. Return exactly one uppercase letter from A, B, C, D. Do not include reasoning, explanations, markdown, punctuation, or any other text.'
spec=dict(source='same eight fixed TRAIN questions from prior audit; no test-based selection',
 source_indices=[r['source_index'] for r in records],instruction=instruction,prefill='Final Answer: (',
 max_new_tokens=16,do_sample=False,seed=42,conditions=['strong_instruction','strong_instruction_with_answer_prefill'],
 rotation_probe='First fixed training question; rotate choices left by0/1/2/3 using prefill; report selected original option, not clinical adjudication.',
 code_sha256=sha(Path(__file__)),model_revision=protocol['revision'],quantization=protocol['quantization'])
if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==spec
else:save(OUT/'protocol.json',spec)
processor=AutoProcessor.from_pretrained(ROOT/'mixture-evaluation-v1/processor',use_fast=False,local_files_only=True)
model=load_model(protocol,'base');model.eval();torch.manual_seed(42)
results=[]
def run(row,condition,index,rotation=0):
 dest=OUT/f'{condition}-{index:02d}-r{rotation}.json'
 if dest.exists():return read(dest)
 original=[row['options'][c] for c in 'ABCD'];order=list(range(4));order=order[rotation:]+order[:rotation]
 choices=[original[i] for i in order]
 prompt=instruction+'\n\n'+row['question']+'\n\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',choices))
 messages=[dict(role='user',content=[dict(type='text',text=prompt)])]
 prefill=condition=='strong_instruction_with_answer_prefill'
 if prefill:messages.append(dict(role='assistant',content=[dict(type='text',text=spec['prefill'])]))
 kwargs=dict(continue_final_message=True) if prefill else dict(add_generation_prompt=True)
 rendered=processor.apply_chat_template(messages,tokenize=False,**kwargs)
 batch=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',truncation=False,**kwargs)
 if prefill:assert rendered.endswith(spec['prefill']) and not rendered.endswith('<end_of_turn>')
 started=time.monotonic()
 with torch.inference_mode():
  gen=model.generate(**{k:v.to('cuda') for k,v in batch.items()},max_new_tokens=16,do_sample=False,num_beams=1,
   use_cache=True,logits_to_keep=1,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=protocol['diagnosis_generation']['eos_token_id'])
 ids=gen[0,batch['input_ids'].shape[1]:].tolist();truncated=len(ids)==16 and ids[-1] not in protocol['diagnosis_generation']['eos_token_id']
 kept=list(ids)
 while kept and kept[-1] in protocol['diagnosis_generation']['eos_token_id']:kept.pop()
 text=processor.tokenizer.decode(kept,skip_special_tokens=True)
 match=re.fullmatch(r'\s*([ABCD])\s*\)\s*[.!]?\s*',text) if prefill else re.fullmatch(r'\s*([ABCD])\s*',text)
 letter=match.group(1) if match and not truncated else ''
 r=dict(source_index=row['source_index'],condition=condition,rotation=rotation,displayed_option_to_original_index=order,
  original_gold=row['gold'],prediction=letter,selected_original_option='ABCD'[order['ABCD'.index(letter)]] if letter else '',
  raw_output=processor.tokenizer.decode(ids,skip_special_tokens=False),continuation_text=text,
  input_ids=batch['input_ids'][0].tolist(),input_user_text=prompt,assistant_prefill=spec['prefill'] if prefill else '',
  rendered_suffix=rendered[-80:],generated_token_ids=ids,truncated=truncated,strict_format_valid=bool(letter),
  elapsed_seconds=time.monotonic()-started,protocol_sha256=sha(OUT/'protocol.json'))
 save(dest,r);print(condition,index,'rotation',rotation,'letter',repr(letter),'continuation',repr(text),'truncated',truncated,flush=True)
 return r
for condition in spec['conditions']:
 for i,row in enumerate(records):results.append(run(row,condition,i))
rotation_results=[run(records[0],'strong_instruction_with_answer_prefill',0,j) for j in range(4)]
summary={}
for condition in spec['conditions']:
 rs=[r for r in results if r['condition']==condition]
 summary[condition]=dict(n=len(rs),strict_format_valid=sum(r['strict_format_valid'] for r in rs),truncated=sum(r['truncated'] for r in rs),
  train_correct=sum(r['selected_original_option']==r['original_gold'] and r['strict_format_valid'] for r in rs),
  predicted_letters=dict(collections.Counter(r['prediction'] for r in rs)),
  mean_seconds=sum(r['elapsed_seconds'] for r in rs)/len(rs),raw_new_tokens_mean=sum(len(r['generated_token_ids']) for r in rs)/len(rs))
summary['rotation_probe']=[{k:r[k] for k in ['rotation','prediction','selected_original_option','strict_format_valid']} for r in rotation_results]
summary['limitations']='8 TRAIN questions only; already seen in prior mechanism audit, possibly base training. Not benchmark accuracy or an official protocol replication. No hard token restriction was used; format compliance was observed, not guaranteed.'
save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
