"""Separate-process Pillow 10.3 audit; never imported by model runtime."""
import os
from pathlib import Path
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.image-audit-deps'))
from PIL import Image,features

def main():
    assert Image.__version__=='10.3.0'
    root=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'data/RareMed_CADASIL_SFT_Mixture_Experiments'))
    out=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'runs/mixture-experiments/preflight')); out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((root/'manifests/sample_manifest.csv').open(encoding='utf-8-sig')))
    mismatches=[]
    for r in rows:
        path=root/'images'/Path(r['image']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256']
        with Image.open(path) as im:
            rgb=im.convert('RGB'); pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        if pixel!=r['pixel_sha256']: mismatches.append(r['sample_id'])
    report=dict(Pillow=Image.__version__,jpeg_library=features.version('jpg'),libjpeg_turbo=features.version('libjpeg_turbo'),
        verified_files=len(rows),pixel_hash_mismatches=mismatches,
        status='PASS_REPRODUCES_UPSTREAM_PIXELS' if not mismatches else 'DIFFERENCES_REMAIN',
        training_environment_modified=False,source_images_modified=False)
    (out/'upstream-decoder-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
