import os
from pathlib import Path
import json, os, platform, shutil, subprocess, sys
from pathlib import Path
import importlib.metadata as md
import torch
import psutil

out = Path(os.environ.get('RAREMED_LOCAL_ROOT',str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))))))/'runs/environment'; out.mkdir(parents=True, exist_ok=True)
report = dict(python=sys.version, executable=sys.executable, os=platform.platform(),
              torch=torch.__version__, cuda_runtime=torch.version.cuda,
              cuda_available=torch.cuda.is_available(), packages={d.metadata['Name']: d.version for d in md.distributions()})
report.update(ram_total=psutil.virtual_memory().total,ram_available=psutil.virtual_memory().available,
              disk_free={d:shutil.disk_usage(d).free for d in ['C:/','G:/']})
report['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
if torch.cuda.is_available():
    report.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  vram=torch.cuda.get_device_properties(0).total_memory,
                  bf16_native_supported=torch.cuda.is_bf16_supported(including_emulation=False))
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(64, 16, quant_type='nf4', compute_dtype=torch.float16).cuda()
    x = torch.randn(2, 64, device='cuda', dtype=torch.float16, requires_grad=True)
    y = layer(x); y.float().square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    report['nf4_forward_backward'] = 'PASS'
(out/'probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(out/'requirements.freeze.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(report['packages'].items()))+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'packages'}, indent=2))
