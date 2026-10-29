"""Check or fetch checksum-locked public benchmark metadata; never calls an LLM."""
import argparse,hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');ap.add_argument('--seed-model-lock',action='store_true');a=ap.parse_args()
 lock=json.loads((ROOT/'metadata/source_lock.json').read_text(encoding='utf8'));results=[]
 for item in lock['files']:
  p=ROOT/item['destination'];status='missing'
  if p.exists():status='verified' if sha(p)==item['sha256'] else 'hash_mismatch'
  if status=='hash_mismatch':raise ValueError('Existing source differs; refusing overwrite: '+item['destination'])
  if status=='missing' and a.download:
   p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.download-part')
   if item.get('bundled_metadata'):shutil.copy2(ROOT/item['bundled_metadata'],temp)
   else:
    with urllib.request.urlopen(item['url'],timeout=180) as r,temp.open('wb') as f:shutil.copyfileobj(r,f)
   if sha(temp)!=item['sha256']:temp.unlink();raise ValueError('Downloaded source checksum mismatch: '+item['destination'])
   temp.replace(p);status='verified'
  results.append({'path':item['destination'],'status':status})
 if a.seed_model_lock:
  target=ROOT/'models/medcpt/manifest.json';source=ROOT/'metadata/medcpt_model_lock.json'
  if target.exists() and json.loads(target.read_text(encoding='utf8'))!=json.loads(source.read_text(encoding='utf8')):raise ValueError('Model manifest differs; refusing overwrite')
  target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():shutil.copy2(source,target)
 print(json.dumps({'sources':results,'download_requested':a.download,'paid_calls':0},indent=2))
 raise SystemExit(0 if all(r['status']=='verified' for r in results) else 1)
if __name__=='__main__':main()
