"""Wait for current MedQA's successful report and worker exit, then serial tumor jobs."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,sys,time,traceback
from resume_mixture_evaluation import reliable_atomic as save
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'tumor-binary-evaluation-v1';PARENT=ROOT/'medqa-direct-v2'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def alive(pid):
 h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
 if not h:return False
 code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h);return code.value==259
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['run','status'],default='status',nargs='?');arg=a.parse_args()
 if arg.stage=='status':print(json.dumps(read(OUT/'queue-status.json'),indent=2));return
 protocol=read(OUT/'protocol.json')
 for n,h in protocol['code_sha256'].items():assert sha(ROOT/n)==h,'Queued source changed: '+n
 lock=OUT/'queue.lock'
 if lock.exists():
  assert not alive(read(lock)['pid']),'Queue already running.'
  lock.unlink()
 with lock.open('x') as f:json.dump(dict(pid=os.getpid()),f)
 try:
  parent_digest=sha(PARENT/'protocol.json')
  save(OUT/'queue-config.json',dict(parent_protocol_sha256=parent_digest,tumor_protocol_sha256=sha(OUT/'protocol.json'),pid=os.getpid()))
  print('WAITING: current MedQA must finish successfully, paired report must exist, worker must exit.',flush=True)
  while True:
   assert sha(PARENT/'protocol.json')==parent_digest,'Parent protocol changed.'
   success=False
   if (PARENT/'paired_report.json').exists():
    report=read(PARENT/'paired_report.json');success=report['protocol_sha256']==parent_digest and all(report['results'][m]['medqa_n']==1273 for m in ['Base','A','B','C'])
   worker=PARENT/'test-worker.lock'
   active=worker.exists() and alive(read(worker)['pid'])
   if success and not active:break
   if not active and not success:
    raise RuntimeError('MedQA worker stopped without complete paired report. Queue blocked; do not start tumor inference.')
   save(OUT/'queue-status.json',dict(status='WAITING_FOR_MEDQA',parent_protocol_sha256=parent_digest,updated_unix=time.time()))
   time.sleep(15)
  save(OUT/'queue-status.json',dict(status='RUNNING_TUMOR_TASKS',started_unix=time.time()))
  env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
  subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/tumor_binary_evaluation.py'),'run'],cwd=ROOT,env=env,check=True)
  subprocess.run([str(ROOT/'.figure-build/.venv/Scripts/python.exe'),str(ROOT/'scripts/report_tumor_binary.py')],cwd=ROOT,env=env,check=True)
  save(OUT/'queue-status.json',dict(status='COMPLETE',finished_unix=time.time()));print('QUEUE COMPLETE',flush=True)
 finally:
  if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'queue-status.json',dict(status='BLOCKED_OR_FAILED',error=str(e),traceback=traceback.format_exc()))
  raise
