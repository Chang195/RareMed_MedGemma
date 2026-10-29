"""Allowlisted code/results release. Private working directories are never traversed.
--check validates sources; --build creates a GitHub upload ZIP next to this project.
"""
import argparse,hashlib,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def files():
 fixed=['README.md','.gitignore','.gitattributes','requirements.txt','requirements-medcpt.txt','pipeline.py']
 paths=[ROOT/n for n in fixed]
 for folder,extensions in [('scripts',{'.py','.md'}),('tools',{'.py'}),('configs',{'.json','.txt'}),('docs',{'.md'}),('.github',{'.yml','.yaml'}),('results',{'.json','.csv','.md','.png','.svg'})]:
  paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts]
 paths += [ROOT/'metadata'/n for n in ['script_registry.json','source_lock.json','multicare_record.json','medcpt_model_lock.json','collection_lock.json']]
 paths += [ROOT/'assets/README.md']
 exp=ROOT/'RareMed_CADASIL_SFT_Mixture_Experiments'
 paths += list((exp/'scripts').glob('*.py'))+[exp/'README.md',exp/'requirements.txt']
 return sorted(set(p for p in paths if p.name!='organize_project.py'))
def check(paths):
 problems=[]
 for p in paths:
  if not p.exists():problems.append({'path':str(p.relative_to(ROOT)),'problem':'missing'});continue
  if not p.resolve().is_relative_to(ROOT):problems.append({'path':str(p.relative_to(ROOT)),'problem':'outside_project'});continue
  if p.stat().st_size>10*1024*1024:problems.append({'path':str(p.relative_to(ROOT)),'problem':'unexpected_large_file'})
  if p.suffix not in {'.png'}:
   text=p.read_text(encoding='utf-8-sig')
   if re.search(r'\bsk-[A-Za-z0-9_-]{20,}\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',text):problems.append({'path':str(p.relative_to(ROOT)),'problem':'possible_secret_redacted'})
   if re.search(r'[A-Za-z]:[\\/]Users[\\/][^\s"\'<>]+',text):problems.append({'path':str(p.relative_to(ROOT)),'problem':'personal_absolute_path'})
 registry=json.loads((ROOT/'metadata/script_registry.json').read_text(encoding='utf8'))
 for name,item in registry.items():
  p=ROOT/'scripts'/name
  if not p.exists():problems.append({'path':'scripts/'+name,'problem':'missing_retained_module'})
  elif hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:problems.append({'path':'scripts/'+name,'problem':'frozen_implementation_changed'})
  for dep in item['dependencies']:
   if not (ROOT/'scripts'/dep).exists():problems.append({'path':'scripts/'+name,'problem':'missing_dependency','dependency':dep})
 if problems:raise ValueError(json.dumps(problems,ensure_ascii=True,indent=2))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--build',action='store_true');a=ap.parse_args();paths=files();check(paths)
 manifest={'files':{p.relative_to(ROOT).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths},'raw_data_or_API_payloads_included':False,'code_license_selected':False}
 if a.build:
  out=ROOT.parent/'RareMed-Pairs_GitHub_ready.zip'
  with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
   for p in paths:
    info=zipfile.ZipInfo('RareMed-Pairs/'+p.relative_to(ROOT).as_posix(),date_time=(2026,10,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
   z.writestr('RareMed-Pairs/RELEASE_MANIFEST.json',json.dumps(manifest,indent=2))
  with zipfile.ZipFile(out) as z:assert z.testzip() is None
  print(json.dumps({'status':'PASS','zip':str(out),'files':len(paths)+1,'bytes':out.stat().st_size,'private_working_data_included':False},indent=2))
 else:print(json.dumps({'status':'PASS','allowlisted_files':len(paths),'private_working_data_included':False,'frozen_script_hashes_verified':len(json.loads((ROOT/'metadata/script_registry.json').read_text(encoding='utf8')))},indent=2))
if __name__=='__main__':main()
