"""Small fixed TRAIN-split protocol diagnostic; no test-set prompt selection.

Compares frozen bare-letter scoring with actual greedy answers under the SAME prompt.
Additional paper-style prompts are mechanism checks, not a benchmark replication.
"""
import os
from pathlib import Path
import os
os.environ.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
from pathlib import Path
import hashlib,json,random,re,sys,time,collections
import torch
from transformers import AutoProcessor
from paired_evaluation import load_model
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'medqa-audit-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
 for i in range(30):
  try:os.replace(t,p);return
  except PermissionError:
   if i==29:raise
   time.sleep(.1)
p=read(ROOT/'mixture-evaluation-v1/protocol.json')
source=OUT/'datasets/source_train.jsonl'
assert digest(source)=='3a65baf760f17c395058d6699315a7ce8aa767c50cb193fea87ceb94dd790324'
train=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
test=[json.loads(x) for x in (ROOT/'mixture-evaluation-v1/datasets/medqa.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
test_questions={r['question'] for r in test}
indices=random.Random(42).sample(range(len(train)),len(train))
selected=[i for i in indices if train[i]['question'] not in test_questions][:8]
spec=dict(seed=42,source_split='train',source_sha256=digest(source),source_rows=len(train),selected_indices=selected,
 original_protocol_sha256=digest(ROOT/'mixture-evaluation-v1/protocol.json'),model_revision=p['revision'],
 same_prompt_generation=dict(max_new_tokens=64,do_sample=False,num_beams=1,use_cache=True),
 paper_style_generation=dict(max_new_tokens=256,do_sample=False,num_beams=1,use_cache=True),
 purpose='Failure-mechanism audit only, not held-out accuracy; original training questions may be in model pretraining.',
 code_sha256=digest(Path(__file__)))
protocol=OUT/'audit_protocol.json'
if protocol.exists():assert read(protocol)==spec,'Audit protocol already frozen; choose new directory for changes.'
else:save(protocol,spec)
processor=AutoProcessor.from_pretrained(ROOT/'mixture-evaluation-v1/processor',local_files_only=True,use_fast=False)
t=processor.tokenizer
model=load_model(p,'base');model.eval();torch.manual_seed(42)
families={name:[t(prefix+c,add_special_tokens=False)['input_ids'] for c in 'ABCD'] for name,prefix in [('bare',''),('space',' ')]}
assert all(len(ids)==1 for f in families.values() for ids in f)
families={name:[ids[0] for ids in value] for name,value in families.items()}
eos=p['diagnosis_generation']['eos_token_id']
def encode(text):
 msg=[dict(role='user',content=[dict(type='text',text=text)])]
 return processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
def logits_for(batch):
 with torch.inference_mode():logits=model(**{k:v.to('cuda') for k,v in batch.items()},logits_to_keep=1).logits[0,-1].float()
 assert torch.isfinite(logits).all()
 return logits
def scores(logits):
 prob=logits.softmax(-1);top=torch.topk(prob,10)
 r={'top10':[dict(token_id=int(i),token=t.convert_ids_to_tokens(int(i)),decoded=t.decode([int(i)]),probability=float(v)) for v,i in zip(top.values,top.indices)]}
 for name,ids in families.items():
  vals=prob[ids].cpu().tolist();r[name+'_probabilities']=dict(zip('ABCD',vals));r[name+'_mass']=sum(vals)
  r[name+'_prediction']='ABCD'[max(range(4),key=lambda j:vals[j])]
 return r
def parse(text):
 final=re.findall(r'Final\s*Answer\s*:\s*\(?\s*([ABCD])\s*\)?',text,re.I)
 if final:return final[-1].upper(),'explicit_final_answer'
 s=text.strip().strip('*` \n\t').strip().rstrip('.!').strip()
 if re.fullmatch(r'[ABCD]',s):return s,'exact_letter'
 m=re.fullmatch(r'(?:The\s+(?:correct\s+)?answer\s+is|Answer\s*:)\s*\(?([ABCD])\)?[.!]?',s,re.I)
 return (m.group(1).upper(),'explicit_answer_phrase') if m else ('','unparsed')
def generate(batch,n):
 started=time.monotonic()
 with torch.inference_mode():
  out=model.generate(**{k:v.to('cuda') for k,v in batch.items()},max_new_tokens=n,do_sample=False,num_beams=1,use_cache=True,
   logits_to_keep=1,pad_token_id=t.pad_token_id,eos_token_id=eos,return_dict_in_generate=True,output_scores=True)
 ids=out.sequences[0,batch['input_ids'].shape[1]:].tolist();kept=list(ids)
 while kept and kept[-1] in eos:kept.pop()
 text=t.decode(kept,skip_special_tokens=True).strip();pred,cat=parse(text)
 return dict(raw_output=t.decode(ids,skip_special_tokens=False),text=text,prediction=pred,parse_category=cat,
   generated_token_ids=ids,truncated=len(ids)==n and ids[-1] not in eos,
   first_generation_token_id=ids[0],first_generation_score_top10=scores(out.scores[0][0].float())['top10'],
   elapsed_seconds=time.monotonic()-started)
records=[]
for j,index in enumerate(selected):
 dest=OUT/f'sample-{j:02d}.json'
 if dest.exists():records.append(read(dest));continue
 row=train[index];assert row['answer']==row['options'][row['answer_idx']]
 text=p['qa_prompt_template'].format(question=row['question'],**row['options']);batch=encode(text)
 ls=logits_for(batch);sc=scores(ls)
 result=dict(source_index=index,gold=row['answer_idx'],question=row['question'],options=row['options'],
    input_text=text,input_ids=batch['input_ids'][0].tolist(),first_step=sc,same_prompt_generation=generate(batch,64))
 save(dest,result);records.append(result)
 print('DIRECT',j+1,'/8','bare',sc['bare_prediction'],'space',sc['space_prediction'],'generated',result['same_prompt_generation']['prediction'],
    'top1',repr(sc['top10'][0]['decoded']),'bare_mass',round(sc['bare_mass'],5),'time',round(result['same_prompt_generation']['elapsed_seconds'],2),flush=True)
for j,index in enumerate(selected[:4]):
 dest=OUT/f'paper-style-{j:02d}.json'
 if dest.exists():continue
 row=train[index]
 # Paraphrase of published evaluation style, deliberately NOT called exact reproduction.
 question=row['question']+'\n\n'+'\n'.join(f'{c}. {row["options"][c]}' for c in 'ABCD')
 prompt='Consider this medical multiple-choice question. Explain your reasoning briefly, then end with Final Answer: (X), where X is A, B, C, or D.\n\n'+question
 b=encode(prompt);r=generate(b,256)
 save(dest,dict(source_index=index,gold=row['answer_idx'],input_text=prompt,input_ids=b['input_ids'][0].tolist(),generation=r))
 print('PAPER-STYLE',j+1,'/4','parsed',r['prediction'],'truncated',r['truncated'],'tokens',len(r['generated_token_ids']),flush=True)
paper=[read(OUT/f'paper-style-{j:02d}.json') for j in range(4)]
summary=dict(audit_protocol_sha256=digest(protocol),n=8,source_split='train',not_a_benchmark=True,
  bare=dict(correct=sum(r['first_step']['bare_prediction']==r['gold'] for r in records),predictions=dict(collections.Counter(r['first_step']['bare_prediction'] for r in records))),
  space=dict(correct=sum(r['first_step']['space_prediction']==r['gold'] for r in records),predictions=dict(collections.Counter(r['first_step']['space_prediction'] for r in records))),
  same_prompt_greedy=dict(correct=sum(r['same_prompt_generation']['prediction']==r['gold'] and not r['same_prompt_generation']['truncated'] for r in records),
   unparsed=sum(not r['same_prompt_generation']['prediction'] for r in records),truncated=sum(r['same_prompt_generation']['truncated'] for r in records),
   predictions=dict(collections.Counter(r['same_prompt_generation']['prediction'] for r in records))),
  paper_style=dict(n=4,correct=sum(r['generation']['prediction']==r['gold'] and not r['generation']['truncated'] for r in paper),
   unparsed=sum(not r['generation']['prediction'] for r in paper),truncated=sum(r['generation']['truncated'] for r in paper)),
  mean_bare_probability_mass=sum(r['first_step']['bare_mass'] for r in records)/len(records),
  mean_space_probability_mass=sum(r['first_step']['space_mass'] for r in records)/len(records),
  excluded_hypotheses=['Source test question/option/answer normalization verified on all1273; snapshot and saved processor vocabulary/chat template identical.',
   'No adapter attached; original NF4 base load reused.'],
  limitations='Small fixed training-split mechanism check only. Not an estimate of official MedQA accuracy; no unquantized control, no causal attribution of all gap.')
save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
