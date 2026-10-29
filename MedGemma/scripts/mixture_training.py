"""Adapt the proven diagnosis trainer without changing its frozen source.

Only dataset paths and per-arm configuration are adapted. The original trainer's
NF4/FP32/SDPA, LoRA, assistant masks, optimization, validation selection and
checkpoint/resume implementation are reused directly.
"""
import argparse
import gc
import importlib.metadata as md
import json
from pathlib import Path
import random
import shutil
from mixture_data import ARMS,audit,load_messages
from diagnosis_data import MODEL,REVISION,sha,read_jsonl,encode,LOCAL
import diagnosis_train as core
import torch
from transformers import AutoProcessor
from peft import get_peft_model,LoraConfig

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['mixture_sft.py','scripts/mixture_training.py','scripts/mixture_data.py','scripts/diagnosis_train.py','scripts/diagnosis_data.py','scripts/pipeline.py']

def prepare(package,out):
    assert all(not (out/arm/'latest.json').exists() for arm in ARMS), 'Training already exists: use resume or a new --run'
    rows,report=audit(package,out/'data-check')
    inventory=core.read(core.BASE/'original-inventory.json')
    for name,info in inventory['files'].items():
        assert sha(core.BASE/name)==info['sha256'], 'Original weight integrity failure: '+name
    processor=AutoProcessor.from_pretrained(LOCAL/'runs/diagnosis-night-01/processor',use_fast=False,local_files_only=True)
    all_masks={}
    for arm,(dirname,pct) in ARMS.items():
        folder=package/dirname; arm_out=out/arm; view=arm_out/'loader-data'; view.mkdir(parents=True,exist_ok=True)
        selected_images={}; tokens=[]
        for split in ['train','val']:
            normalized=[]
            for i,row in enumerate(rows[arm][split]):
                image=(folder/row['image']).resolve()
                assert image.parent==(package/'images').resolve()
                selected_images[str(image)]=row['metadata']['image_sha256']
                # Keep source JSONLs unchanged; only resolve paths in a separate recorded loader view.
                copied=dict(row,image=str(image),image_path=str(image))
                normalized.append(copied)
                key=row['sample_id']
                if key not in all_masks:
                    batch,_,n=encode(processor,load_messages(row,folder,package,True))
                    all_masks[key]=dict(sample_id=key,prefix_tokens=n,input_tokens=batch['input_ids'].shape[1],
                        target=row['binary_target'],supervised_tokens=int((batch['labels']!=-100).sum()))
                tokens.append(dict(split=split,**all_masks[key]))
            destination=view/('validation.jsonl' if split=='val' else 'train.jsonl')
            destination.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in normalized),encoding='utf-8')
            print('All masks checked:',arm,split,len(normalized),flush=True)
        processor.save_pretrained(arm_out/'processor')
        core.write(arm_out/'token-checks.json',tokens)
        config=dict(created_utc=core.now(),model=MODEL,revision=REVISION,base=str(core.BASE),data=str(view),
            source_package=str(package),arm=arm,CADASIL_ratio=pct/100,train_records=100,validation_records=67,
            seed=42,epochs=3,micro_batch=1,gradient_accumulation=4,learning_rate=2e-5,optimizer='AdamW',
            betas=[0.9,0.999],eps=1e-8,weight_decay=0.01,max_grad_norm=1.0,scheduler='none',
            quantization='NF4 double quant',compute_dtype='float32',attention='sdpa',lora_rank=4,lora_alpha=8,
            lora_dropout=0.0,lora_targets='language_model q_proj/v_proj only',gradient_checkpointing_use_reentrant=False,
            checkpoint_every_optimizer_steps=20,optimizer_steps_per_epoch=25,total_optimizer_steps=75,
            checkpoint_selection='lowest shared-validation mean per-example assistant-token NLL at epoch end; same criterion in every arm',
            validation_class_counts=dict(CADASIL=5,non_CADASIL=62),validation_loss_only=True,test_generation=False,baseline_generation=False,
            supervision='assistant label text only, no special/end/user/image/padding loss; existing encode() unchanged',
            dataset_file_sha256={name:sha(view/name) for name in ['train.jsonl','validation.jsonl']},
            source_jsonl_sha256={split:sha(folder/(split+'.jsonl')) for split in ['train','val','test']},
            image_file_sha256=selected_images,base_inventory_sha256=sha(core.BASE/'original-inventory.json'),
            source_sha256={name:sha(ROOT/name) for name in SOURCES},
            dependencies={d.metadata['Name']:d.version for d in md.distributions()},
            pixel_hash_warning=report['pixel_hash_differences'],
            pixel_policy='Keep original image bytes; freeze this machine Pillow 12 runtime and record upstream/local differences. No repair/re-encoding or environment downgrade.',
            image_processing=processor.image_processor.to_dict(),chat_template=processor.chat_template,
            training_ready=False,clinical_adjudication=False,original_or_previous_adapters_loaded=False,
            limitations='Exploratory source-label experiment; source-class confounding; reused known test cases; unbalanced validation criterion does not prove CADASIL performance')
        if (arm_out/'config.json').exists():
            old=core.read(arm_out/'config.json')
            assert {k:v for k,v in old.items() if k!='created_utc'}=={k:v for k,v in config.items() if k!='created_utc'}, 'Prepared configuration changed: use a new --run'
        else: core.atomic(arm_out/'config.json',config)
        for name in SOURCES:
            destination=arm_out/'source'/name; destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,destination)
        core.atomic(arm_out/'status.json',dict(status='PREPARED',arm=arm,train_images=100,validation_images=67,
            total_steps=75,test_predictions_generated=False,pixel_hash_warning_retained=True))
    core.atomic(out/'study.json',dict(status='PREPARED',arms=list(ARMS),source_package=str(package),model=MODEL,revision=REVISION,
        train_sizes=[100,100,100],CADASIL_percent=[20,50,80],steps_per_arm=75,
        same_val_test_jsonls=True,no_training_yet=True,pixel_hash_warning=report['pixel_hash_differences']))
    print('PREPARE PASS: A/B/C prepared, no optimizer steps. Run check before train.',flush=True)

def configure(out):
    config=core.read(out/'config.json')
    for name,digest in config['source_sha256'].items(): assert sha(ROOT/name)==digest, 'Prepared code changed: '+name
    for name in ['torch','transformers','peft','bitsandbytes','pillow']:
        # Package metadata spelling may differ, compare canonical names.
        versions={k.lower().replace('_','-'):v for k,v in config['dependencies'].items()}
        assert md.version(name)==versions[name], 'Prepared environment changed: '+name
    package=Path(config['source_package']); dirname=ARMS[config['arm']][0]
    for split,digest in config['source_jsonl_sha256'].items(): assert sha(package/dirname/(split+'.jsonl'))==digest
    for image,digest in config['image_file_sha256'].items(): assert sha(Path(image))==digest
    core.DATA=Path(config['data'])
    return config

def probe(out):
    config=configure(out)
    marker=out/'probe.json'
    if marker.exists() and core.read(marker).get('status')=='PASS' and core.read(marker)['config_sha256']==sha(out/'config.json'):
        print('Existing train-only probe PASS:',config['arm'],flush=True); return
    random.seed(42); torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(out/'processor',use_fast=False,local_files_only=True)
    model=core.load_base()
    targets=[name for name,module in model.named_modules() if name.endswith(('q_proj','v_proj')) and 'language_model' in name]
    assert len(targets)==68
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=targets,task_type='CAUSAL_LM'))
    rows=read_jsonl(core.DATA/'train.jsonl'); results=[]; model.train(); torch.cuda.reset_peak_memory_stats()
    for label in ['CADASIL','non-CADASIL']:
        row=next(r for r in rows if r['binary_target']==label)
        model.zero_grad(set_to_none=True)
        loss=model(**core.get_batch(processor,row)).loss
        assert torch.isfinite(loss)
        loss.backward()
        grads=[v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
        assert len(grads)==136 and all(torch.isfinite(g).all() for g in grads)
        b=[v.grad for name,v in model.named_parameters() if 'lora_B' in name]
        assert all(g is not None and torch.count_nonzero(g)>0 for g in b)
        entry=dict(sample_id=row['sample_id'],label=label,loss=float(loss.detach()),gradient_tensors=len(grads),nonzero_B_gradients=len(b),peak_vram=torch.cuda.max_memory_allocated())
        results.append(entry); print('Train-only probe:',config['arm'],entry,flush=True)
    core.atomic(marker,dict(status='PASS',samples=results,optimizer_steps=0,adapter_discarded=True,config_sha256=sha(out/'config.json')))
    del model; gc.collect(); torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','check','train','resume','status'])
    p.add_argument('--arm',choices=['A','B','C','all'],required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--data',type=Path,required=True)
    a=p.parse_args(); package=a.data.resolve(); out=a.out.resolve()
    arms=list(ARMS) if a.arm=='all' else [a.arm]
    if a.stage=='prepare': prepare(package,out); return
    for arm in arms:
        current=out/arm
        if a.stage=='status':
            print(arm,json.dumps(core.read(current/'status.json') if (current/'status.json').exists() else dict(status='NOT_PREPARED'),ensure_ascii=False)); continue
        assert (current/'config.json').exists(), 'Run prepare first'
        if a.stage=='resume' and (current/'status.json').exists() and core.read(current/'status.json').get('status')=='COMPLETE':
            print('Already complete:',arm,flush=True); continue
        configure(current); probe(current)
        if a.stage in ['train','resume']:
            print('Starting independent SFT arm',arm,'from ORIGINAL base; 75 optimizer steps',flush=True)
            core.train(current,resume=a.stage=='resume' and (current/'latest.json').exists())
            gc.collect(); torch.cuda.empty_cache()
    if a.stage in ['train','resume']:
        states={arm:core.read(out/arm/'status.json')['status'] for arm in ARMS}
        print('STUDY STATUS:',json.dumps(states),flush=True)

if __name__=='__main__': main()
