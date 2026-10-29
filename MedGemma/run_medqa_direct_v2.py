"""User entry point for validated v2 evaluation; preserves original experiment."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'medqa-direct-v2'
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','status'],default='status',nargs='?');a=p.parse_args()
 env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
 if a.stage=='status':return subprocess.call([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/medqa_direct_v2.py'),'status'],cwd=ROOT,env=env)
 validation=json.loads((OUT/'validation-report.json').read_text());assert validation['passed'],'Format gate failed.'
 lock=OUT/'test-worker.lock'
 if lock.exists():
  import ctypes
  prior=json.loads(lock.read_text());h=ctypes.windll.kernel32.OpenProcess(0x1000,False,prior['pid']);active=False
  if h:
   code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h);active=code.value==259
  assert not active,'Evaluation already running; do not launch duplicate.'
  lock.unlink()
 with lock.open('x') as f:json.dump(dict(pid=os.getpid(),started=time.time()),f)
 try:
  spec=dict(protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
    code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['run_medqa_direct_v2.py','scripts/medqa_direct_v2.py','scripts/report_medqa_direct_v2.py','MedGemma_SFT_Figures/plot_sft_results.py']})
  frozen=OUT/'test-execution.json'
  if frozen.exists():assert json.loads(frozen.read_text())==spec,'Frozen execution code changed.'
  else:frozen.write_text(json.dumps(spec,indent=2),encoding='utf-8')
  subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/medqa_direct_v2.py'),'test'],cwd=ROOT,env=env,check=True)
  subprocess.run([str(ROOT/'.figure-build/.venv/Scripts/python.exe'),str(ROOT/'scripts/report_medqa_direct_v2.py')],cwd=ROOT,env=env,check=True)
  return 0
 finally:
  if lock.exists() and json.loads(lock.read_text())['pid']==os.getpid():lock.unlink()
if __name__=='__main__':raise SystemExit(main())
