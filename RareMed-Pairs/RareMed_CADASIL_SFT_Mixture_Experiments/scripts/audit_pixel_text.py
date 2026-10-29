"""Local OCR screening of source images; no model API or image alteration.
Install rapidocr_onnxruntime separately; --runtime can point to a pip --target directory.
Cached OCR results are keyed by image SHA-256 and can be reused offline.
"""
import argparse, concurrent.futures, hashlib, importlib.metadata, json, re, sys, threading
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path);args=ap.parse_args()
    if args.runtime:sys.path.insert(0,str(args.runtime.resolve()))
    from rapidocr_onnxruntime import RapidOCR
    import rapidocr_onnxruntime
    base=Path(__file__).resolve().parents[1]
    target=base/'inputs/pixel_ocr.json'
    result=json.loads(target.read_text(encoding='utf8')) if target.exists() else {}
    local=threading.local()
    paths=sorted((base/'inputs/raremed_images').glob('*'))+sorted((base/'inputs/medpix_images').glob('*'))
    def one(p):
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in result:return digest,result[digest]
        try:
            if not hasattr(local,'engine'):local.engine=RapidOCR(intra_op_num_threads=1,inter_op_num_threads=1)
            found,_=local.engine(str(p));texts=[{'text':r[1],'confidence':float(r[2])} for r in (found or [])]
            return digest,{'status':'completed','text':texts}
        except Exception as e:return digest,{'status':'error','error':str(e)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,(digest,value) in enumerate(pool.map(one,paths),1):
            result[digest]=value
            if i%25==0 or i==len(paths):
                target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
                print(f'OCR {i}/{len(paths)}',flush=True)
    terms=re.compile(r'CADASIL|NOTCH3|glioblastoma|astrocytoma|multiple\s+sclerosis|leukodystrophy|\bGBM\b',re.I)
    flags=[{'sha256':k,'text':v.get('text')} for k,v in result.items() if terms.search(' '.join(x['text'] for x in v.get('text',[])))]
    model_dir=Path(rapidocr_onnxruntime.__file__).parent/'models'
    metadata={'engine':'RapidOCR CPU','versions':{m:importlib.metadata.version(m) for m in ['rapidocr_onnxruntime','onnxruntime','numpy','opencv-python']},'models':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.glob('*.onnx')},'workers':4,'OCR_text_confidence_threshold':.5,'limitations':'OCR screening only; not proof of no text, PHI or diagnostic overlays'}
    (base/'inputs/pixel_ocr_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps({'images':len(result),'diagnosis_term_flags':flags},ensure_ascii=True))

if __name__=='__main__':main()
