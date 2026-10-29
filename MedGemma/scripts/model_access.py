"""Check gated access and save actual processor without downloading model weights."""
import os
from pathlib import Path
import json, os
from pathlib import Path
root=Path(os.environ.get('RAREMED_LOCAL_ROOT',str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))))))
os.environ.setdefault('HF_HOME',str(root/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoProcessor
model='google/medgemma-1.5-4b-it'
out=root/'runs/model-access'; out.mkdir(parents=True,exist_ok=True)
try:
    revision=HfApi().model_info(model).sha
    hf_hub_download(model,'config.json',revision=revision)
    processor=AutoProcessor.from_pretrained(model,revision=revision,use_fast=False)
    processor.save_pretrained(out/'processor')
    result=dict(status='PASS',model=model,sha=revision,template=processor.chat_template,
        special_tokens=processor.tokenizer.special_tokens_map,image_processing=processor.image_processor.to_dict())
except Exception as e:
    # Do not record HTTP headers, local credentials, or full exception URLs.
    result=dict(status='BLOCKED',error_type=type(e).__name__,model=model,
                next_action='Accept model terms in browser, then run local hf auth login with read token')
    (out/'access.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); raise SystemExit(2)
(out/'access.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('template','image_processing','special_tokens')},indent=2))
