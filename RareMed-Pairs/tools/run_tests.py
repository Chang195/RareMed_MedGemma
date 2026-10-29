"""Run portable unit tests, or local tests that also require frozen source data."""
import argparse,json,os,re,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--local',action='store_true');a=ap.parse_args()
 sys.path.insert(0,str(ROOT/'scripts'))
 tmp=ROOT/'.runtime_tmp/tests';tmp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(tmp)
 names=[];results=[]
 for p in sorted((ROOT/'scripts').glob('test_*.py')):
  if p.stem=='test_medpix_glioma_eval' and not a.local:continue
  names.append(p.stem)
  env={**os.environ,'TMP':str(tmp),'TEMP':str(tmp),'TMPDIR':str(tmp)}
  run=subprocess.run([sys.executable,'-m','unittest','discover','-s','scripts','-p',p.name],cwd=ROOT,env=env,capture_output=True,text=True)
  text=run.stdout+run.stderr;m=re.search(r'Ran (\d+) tests?',text)
  results.append({'module':p.stem,'tests':int(m[1]) if m else 0,'passed':run.returncode==0})
  print(p.stem+': '+('PASS' if run.returncode==0 else 'FAIL'),flush=True)
  if run.returncode:print(text)
 print(json.dumps({'profile':'local_frozen_data' if a.local else 'portable_unit','modules':len(names),'tests':sum(r['tests'] for r in results),'failed_modules':sum(not r['passed'] for r in results)}))
 raise SystemExit(0 if all(r['passed'] for r in results) else 1)
if __name__=='__main__':main()
