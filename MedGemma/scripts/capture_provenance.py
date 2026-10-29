"""Capture local scripts/dependencies/config hashes; never include credentials."""
import os
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
local=Path(os.environ.get('RAREMED_LOCAL_ROOT',str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))))))
out=local/'runs/pilot/provenance'
out.mkdir(parents=True,exist_ok=True)
sources=list((root/'scripts').glob('*.py'))+[root/n for n in ('requirements.txt','run.py','setup.ps1','local-env.ps1','run.ps1')]
hashes={}
for source in sources:
    target=out/'source'/source.relative_to(root)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    hashes[str(source.relative_to(root))]=hashlib.sha256(source.read_bytes()).hexdigest()
packages={d.metadata['Name']:d.version for d in md.distributions()}
(out/'dependencies.freeze.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items()))+'\n',encoding='utf-8')
(out/'sources.json').write_text(json.dumps(dict(python=sys.version,sha256=hashes),indent=2),encoding='utf-8')
print('Local source and dependency provenance captured')
