"""Local diagnosis training. Train updates; validation loss selects; NO test inference.

Every saved checkpoint is at an accumulation boundary and includes optimizer,
RNG, sample cursor, selection history, fixed configuration and data/code hashes.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from diagnosis_data import LOCAL,MODEL,REVISION,audit,read_jsonl,sha,model_messages,encode,write
import torch
from transformers import AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel

ROOT=Path(__file__).resolve().parents[1]
DATA=LOCAL/'data/diagnosis-sft-v0.1'
BASE=LOCAL/'models/medgemma-1.5-4b-it'/REVISION
SOURCES=['train.py','scripts/diagnosis_train.py','scripts/diagnosis_data.py','scripts/pipeline.py']

def atomic(path,value):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    write(tmp,value); os.replace(tmp,path)

def now(): return datetime.now(timezone.utc).isoformat()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def preserve_base():
    source=LOCAL/'huggingface/hub/models--google--medgemma-1.5-4b-it/snapshots'/REVISION
    assert source.is_dir(), 'Original pinned snapshot missing; no download or model replacement attempted'
    assert not (source/'adapter_config.json').exists()
    BASE.mkdir(parents=True,exist_ok=True)
    inventory={}
    index=read(source/'model.safetensors.index.json')
    expected=set(index['weight_map'].values())
    assert len(expected)==2
    for f in sorted(source.iterdir()):
        if not f.is_file(): continue
        # Snapshot contains model/tokenizer/processor files only, never HF credentials.
        if not (f.suffix in ['.json','.safetensors','.model','.jinja','.txt']): continue
        digest=sha(f); destination=BASE/f.name
        if not destination.exists(): shutil.copy2(f,destination)
        assert sha(destination)==digest, f'Preserved base mismatch: {f.name}'
        inventory[f.name]=dict(sha256=digest,bytes=f.stat().st_size)
        print('Base preserved + SHA256 verified:',f.name,flush=True)
    assert expected.issubset(inventory)
    assert not (BASE/'adapter_config.json').exists()
    record=dict(model=MODEL,revision=REVISION,path=str(BASE),source=str(source),files=inventory,
                description='Independent copy of original official snapshot; no smoke adapter; never merged')
    atomic(BASE/'original-inventory.json',record)
    return record

def prepare(out):
    assert not (out/'latest.json').exists(), 'Training exists: use resume or choose a new --run'
    splits,report=audit(DATA,out/'data-check')
    base=preserve_base()
    processor=AutoProcessor.from_pretrained(BASE,use_fast=False,local_files_only=True)
    processor.save_pretrained(out/'processor')
    lengths={}
    for split in ['train','validation']:
        summaries=[]
        for i,row in enumerate(splits[split]):
            batch,_,prefix=encode(processor,model_messages(row,DATA,True))
            summaries.append(dict(sample_id=row['sample_id'],input_tokens=batch['input_ids'].shape[1],
                supervised_tokens=int((batch['labels']!=-100).sum()),prefix_tokens=prefix))
            if (i+1)%25==0: print('Processor audit:',split,i+1,'/',len(splits[split]),flush=True)
        lengths[split]=summaries
    write(out/'all-token-checks.json',lengths)
    config=dict(created_utc=now(),model=MODEL,revision=REVISION,base=str(BASE),data=str(DATA),
        seed=42,epochs=3,micro_batch=1,gradient_accumulation=4,learning_rate=2e-5,
        optimizer='AdamW',betas=[0.9,0.999],eps=1e-8,weight_decay=0.01,max_grad_norm=1.0,
        scheduler='none',quantization='NF4 double quant',compute_dtype='float32',attention='sdpa',
        lora_rank=4,lora_alpha=8,lora_dropout=0.0,lora_targets='language_model q_proj/v_proj only',
        gradient_checkpointing_use_reentrant=False,checkpoint_every_optimizer_steps=20,
        checkpoint_selection='lowest validation mean per-example assistant-token NLL at epoch end',
        validation_loss_only=True,test_generation=False,baseline_generation=False,
        supervision='assistant正文 only; special/end/user/image/padding masked; no truncation',
        dataset_file_sha256=report['package_file_sha256'],base_inventory_sha256=sha(BASE/'original-inventory.json'),
        source_sha256={name:sha(ROOT/name) for name in SOURCES},
        dependencies={d.metadata['Name']:d.version for d in md.distributions()},
        training_ready=False,limitations='Exploratory source-reported labels, not a clinically validated gold standard')
    if (out/'config.json').exists():
        old=read(out/'config.json')
        comparable=lambda c:{k:v for k,v in c.items() if k!='created_utc'}
        assert comparable(old)==comparable(config), 'Prepared experiment changed: use a new --run'
    else: atomic(out/'config.json',config)
    for name in SOURCES:
        dest=out/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    atomic(out/'status.json',dict(status='PREPARED',updated_utc=now(),train_images=239,validation_images=24,
        test_predictions_generated=False,base_model_preserved=str(BASE)))
    print('PREPARE PASS: original model preserved; all train/validation masks checked; no training started',flush=True)

def load_base():
    model=Gemma3ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32),
        dtype=torch.float32,device_map={'':0},attn_implementation='sdpa')
    assert not hasattr(model,'peft_config')
    model.config.use_cache=False; model.config.text_config.use_cache=False
    return prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})

def get_batch(processor,row):
    batch,_,_=encode(processor,model_messages(row,DATA,True))
    return {k:v.to('cuda') for k,v in batch.items()}

def validation(model,processor,rows):
    model.eval(); total=0.0; weighted=0.0; tokens=0
    with torch.no_grad():
        for row in rows:
            batch=get_batch(processor,row)
            loss=model(**batch).loss
            assert torch.isfinite(loss), 'Nonfinite validation loss'
            n=int((batch['labels']!=-100).sum()); value=float(loss)
            total+=value; weighted+=value*n; tokens+=n
    model.train()
    return dict(mean_sample_nll=total/len(rows),token_weighted_nll=weighted/tokens,
                samples=len(rows),supervised_tokens=tokens)

def save_checkpoint(out,model,opt,state):
    name=f"checkpoint-{state['step']:06d}"
    final=out/name; temporary=out/(name+'.incomplete')
    assert not final.exists(), 'Refusing to overwrite completed checkpoint'
    temporary.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(temporary,safe_serialization=True)
    payload=dict(**state,optimizer=opt.state_dict(),python_rng=random.getstate(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        config_sha256=sha(out/'config.json'))
    torch.save(payload,temporary/'training-state.pt')
    write(temporary/'progress.json',{k:v for k,v in state.items()})
    temporary.rename(final)
    atomic(out/'latest.json',dict(checkpoint=name,step=state['step']))
    if state.get('best_step')==state['step']:
        atomic(out/'best.json',dict(checkpoint=name,step=state['step'],validation_mean_sample_nll=state['best_loss']))
    print('Checkpoint saved:',name,flush=True)
    return final

def train(out,resume):
    config=read(out/'config.json')
    if resume and (out/'status.json').exists() and read(out/'status.json').get('status')=='COMPLETE':
        print('Already COMPLETE. Best checkpoint:',read(out/'best.json'),flush=True)
        return
    for name,digest in config['source_sha256'].items():
        assert sha(ROOT/name)==digest, 'Code changed since prepare: '+name
    for name in ['train.jsonl','validation.jsonl']:
        assert sha(DATA/name)==config['dataset_file_sha256'][name], 'Dataset changed: '+name
    assert sha(BASE/'original-inventory.json')==config['base_inventory_sha256']
    assert torch.cuda.is_available()
    print('CUDA free/total bytes:',torch.cuda.mem_get_info(),flush=True)
    random.seed(config['seed']); torch.manual_seed(config['seed'])
    train_rows=read_jsonl(DATA/'train.jsonl'); val_rows=read_jsonl(DATA/'validation.jsonl')
    processor=AutoProcessor.from_pretrained(out/'processor',use_fast=False,local_files_only=True)
    model=load_base()
    state=dict(step=0,epoch=0,cursor=0,best_loss=None,best_step=None,history=[])
    if resume:
        checkpoint=out/read(out/'latest.json')['checkpoint']
        saved=torch.load(checkpoint/'training-state.pt',map_location='cpu',weights_only=False)
        assert saved['config_sha256']==sha(out/'config.json')
        model=PeftModel.from_pretrained(model,checkpoint,is_trainable=True)
        state={key:saved[key] for key in state}
    else:
        assert not (out/'latest.json').exists(), 'Use resume for existing training'
        targets=[name for name,module in model.named_modules() if name.endswith(('q_proj','v_proj')) and 'language_model' in name]
        assert len(targets)==68
        model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=targets,task_type='CAUSAL_LM'))
    params=[v for v in model.parameters() if v.requires_grad]
    assert sum(v.numel() for v in params)==1114112
    assert all('lora_' in n for n,v in model.named_parameters() if v.requires_grad)
    opt=torch.optim.AdamW(params,lr=config['learning_rate'],betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
    if resume:
        opt.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    total_steps=math.ceil(len(train_rows)/config['gradient_accumulation'])*config['epochs']
    if not resume:
        initial=validation(model,processor,val_rows)
        atomic(out/'initial-validation-loss.json',initial)
        print('Initial validation loss (no generation):',initial,flush=True)
    model.train(); torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
    status=dict(status='RUNNING',updated_utc=now(),step=state['step'],total_steps=total_steps,test_predictions_generated=False)
    atomic(out/'status.json',status)
    with (out/'training.jsonl').open('a',encoding='utf-8') as log:
        try:
            for epoch in range(state['epoch'],config['epochs']):
                order=list(range(len(train_rows))); random.Random(config['seed']+epoch).shuffle(order)
                cursor=state['cursor'] if epoch==state['epoch'] else 0
                while cursor<len(order):
                    group=order[cursor:cursor+config['gradient_accumulation']]
                    opt.zero_grad(set_to_none=True); losses=[]; ids=[]
                    for index in group:
                        row=train_rows[index]
                        loss=model(**get_batch(processor,row)).loss
                        assert torch.isfinite(loss), 'Nonfinite train loss'
                        (loss/len(group)).backward(); losses.append(float(loss.detach())); ids.append(row['sample_id'])
                    norm=torch.nn.utils.clip_grad_norm_(params,config['max_grad_norm'],error_if_nonfinite=True)
                    opt.step(); opt.zero_grad(set_to_none=True)
                    cursor+=len(group); state.update(step=state['step']+1,epoch=epoch,cursor=cursor)
                    entry=dict(step=state['step'],epoch=epoch+1,processed=cursor,train_mean_sample_nll=sum(losses)/len(losses),
                        sample_ids=ids,gradient_norm_before_clip=float(norm),peak_vram=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start)
                    log.write(json.dumps(entry)+'\n'); log.flush()
                    print(f"Step {state['step']}/{total_steps} epoch {epoch+1}/3 samples {cursor}/{len(order)} loss={entry['train_mean_sample_nll']:.5f}",flush=True)
                    status.update(updated_utc=now(),step=state['step'],epoch=epoch+1,processed=cursor,peak_vram=entry['peak_vram'])
                    atomic(out/'status.json',status)
                    if cursor<len(order) and (state['step']==1 or state['step']%config['checkpoint_every_optimizer_steps']==0):
                        save_checkpoint(out,model,opt,state)
                metrics=validation(model,processor,val_rows)
                state['history'].append(dict(epoch=epoch+1,step=state['step'],**metrics))
                if state['best_loss'] is None or metrics['mean_sample_nll']<state['best_loss']:
                    state.update(best_loss=metrics['mean_sample_nll'],best_step=state['step'])
                state.update(epoch=epoch+1,cursor=0)
                save_checkpoint(out,model,opt,state)
                atomic(out/'validation-history.json',state['history'])
                print('Epoch validation:',metrics,flush=True)
            status.update(status='COMPLETE',updated_utc=now(),best=read(out/'best.json'),latest=read(out/'latest.json'))
            atomic(out/'status.json',status)
            print('TRAINING COMPLETE. Base preserved:',BASE,'Best SFT:',out/status['best']['checkpoint'],flush=True)
            print('No test predictions or diagnostic scores generated. Evaluate original and best adapter tomorrow with one fixed protocol.',flush=True)
        except BaseException as exc:
            status.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error_type=type(exc).__name__,updated_utc=now())
            atomic(out/'status.json',status)
            raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','train','resume','status'])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); out=a.out.resolve()
    if a.stage=='status':
        print(json.dumps(read(out/'status.json') if (out/'status.json').exists() else {'status':'NOT_PREPARED'},indent=2))
        return
    out.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare': prepare(out)
    else:
        assert (out/'config.json').exists(), 'Run prepare first'
        train(out,a.stage=='resume')

if __name__=='__main__': main()
