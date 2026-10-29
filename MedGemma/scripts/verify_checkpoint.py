"""Reload adapter and check finite training-sample loss; no held-out metrics."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from pipeline import LOCAL, validate, conversation, encode, write

out=LOCAL/'runs/pilot'
revision=json.loads((out/'model-revision.json').read_text())['sha']
state=load_file(str(out/'adapter/adapter_model.safetensors'))
assert all(torch.isfinite(t).all() for t in state.values())
b_tensors={k:t for k,t in state.items() if 'lora_B' in k}
assert b_tensors and all(torch.count_nonzero(t)>0 for t in b_tensors.values()), 'Some adapter blocks did not update'
rows=validate(LOCAL/'data/caption-pilot-v0.1',out)
processor=AutoProcessor.from_pretrained(out/'processor',use_fast=False)
base=Gemma3ForConditionalGeneration.from_pretrained('google/medgemma-1.5-4b-it',revision=revision,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32),
    dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
model=PeftModel.from_pretrained(base,out/'adapter',is_trainable=False)
model.eval(); model.config.use_cache=False
losses=[]
with torch.no_grad():
    for row in rows:
        batch,_,_=encode(processor,conversation(row,LOCAL/'data/caption-pilot-v0.1'))
        loss=model(**{k:v.to('cuda') for k,v in batch.items()}).loss
        assert torch.isfinite(loss)
        losses.append(dict(sample_id=row['sample_id'],loss=float(loss)))
path=out/'adapter/adapter_model.safetensors'
result=dict(status='PASS',adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            finite_adapter_tensors=len(state),updated_lora_B_blocks=len(b_tensors),reloaded_losses=losses,
            interpretation='training samples only; checkpoint integrity/overfit smoke, not generalization evaluation')
write(out/'checkpoint-check.json',result)
print(json.dumps(result,indent=2))
