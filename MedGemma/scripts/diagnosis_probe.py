"""Disposable train-only forward/backward check; no optimizer updates/checkpoint."""
import random
from diagnosis_data import LOCAL,MODEL,REVISION,read_jsonl,target,model_messages,encode,write,sha
import torch
from transformers import AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training

def main():
    root=LOCAL/'data/diagnosis-sft-v0.1'
    out=LOCAL/'runs/diagnosis-v0.1/preflight'
    rows=read_jsonl(root/'train.jsonl')
    selected=[next(r for r in rows if r['metadata']['diagnosis_family']=='CADASIL_author_reported'),
              max((r for r in rows if r['metadata']['diagnosis_family']!='CADASIL_author_reported'),key=lambda r:len(target(r)))]
    random.seed(42); torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,use_fast=False,local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    model=Gemma3ForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32),
        dtype=torch.float32,device_map={'':0},attn_implementation='sdpa')
    assert not hasattr(model,'peft_config'), 'Base must not contain an adapter'
    model.config.use_cache=False
    model.config.text_config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    targets=[name for name,module in model.named_modules() if name.endswith(('q_proj','v_proj')) and 'language_model' in name]
    assert len(targets)==68
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=targets,task_type='CAUSAL_LM'))
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==1114112
    model.train(); report=[]
    for row in selected:
        model.zero_grad(set_to_none=True)
        batch,_,_=encode(processor,model_messages(row,root,True))
        loss=model(**{k:v.to('cuda') for k,v in batch.items()}).loss
        assert torch.isfinite(loss)
        loss.backward()
        gradients=[v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        b_grads=[(name,v.grad) for name,v in model.named_parameters() if 'lora_B' in name]
        assert all(g is not None and torch.count_nonzero(g)>0 for _,g in b_grads)
        report.append(dict(sample_id=row['sample_id'],loss=float(loss.detach()),
            supervised_tokens=int((batch['labels']!=-100).sum()),gradient_tensors=len(gradients),
            nonzero_lora_B_gradients=len(b_grads),peak_vram=torch.cuda.max_memory_allocated()))
        write(out/'train-forward-backward.json',dict(status='IN_PROGRESS',samples=report))
        print(report[-1],flush=True)
    snapshot=LOCAL/'huggingface/hub/models--google--medgemma-1.5-4b-it/snapshots'/REVISION
    assert not (snapshot/'adapter_config.json').exists()
    write(out/'train-forward-backward.json',dict(status='PASS',samples=report,seed=42,
        base_model=MODEL,revision=REVISION,base_snapshot=str(snapshot),smoke_adapter_loaded=False,
        optimizer_steps=0,weights_saved=False,train_only=True,quantization='NF4 double quant FP32 compute',
        attention='sdpa',rank=4,alpha=8,dropout=0.0,targets=targets,gradient_checkpointing_use_reentrant=False,
        code_sha256={name:sha(__import__('pathlib').Path(__file__).parent/name) for name in ['pipeline.py','diagnosis_data.py','diagnosis_probe.py']}))
    print('Train-only forward/backward PASS; disposable adapter discarded, zero optimizer steps',flush=True)

if __name__=='__main__': main()
