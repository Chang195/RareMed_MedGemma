"""Resume the frozen evaluator with Windows file-replace retries only.

Operational I/O patch: no change to prompts, inference, parser, scoring or protocol.
The original frozen evaluator and its source hashes remain unchanged.
"""
import os
from pathlib import Path
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv/Scripts/python.exe'
def reliable_atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        for attempt in range(30):
            try:
                os.replace(tmp,path);return
            except PermissionError:
                if attempt==29:raise
                time.sleep(min(.05*2**min(attempt,5),1))
    finally:
        try:tmp.unlink(missing_ok=True)
        except PermissionError:pass

def main():
    env=os.environ.copy()
    env.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
       HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',
       PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',WANDB_DISABLED='true')
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        return subprocess.call([str(PYTHON),'-u',str(Path(__file__).resolve()),*sys.argv[1:]],cwd=ROOT,env=env)
    os.environ.update(env)
    sys.path.insert(0,str(ROOT/'scripts'))
    import mixture_evaluation as evaluator
    evaluator.atomic=reliable_atomic
    if len(sys.argv)==1:sys.argv.append('run')
    if sys.argv[1] not in ['run','status','report']:
        raise ValueError('I/O recovery wrapper only supports run, status or report.')
    if sys.argv[1]=='run':
        p=ROOT/'mixture-evaluation-v1'
        record=dict(patch='unique temporary file, flush/fsync and retry PermissionError on os.replace; same JSON contents and atomic replacement semantics',
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          protocol_sha256=hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest(),
          prior_error_sha256=hashlib.sha256((p/'last-error.json').read_bytes()).hexdigest() if (p/'last-error.json').exists() else None,
          created_unix=time.time(),inference_and_scoring_changed=False)
        reliable_atomic(p/f'io-recovery-{time.time_ns()}.json',record)
    evaluator.main()
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        import traceback
        reliable_atomic(ROOT/'mixture-evaluation-v1/last-error.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),utc_unix=time.time()))
        reliable_atomic(ROOT/'mixture-evaluation-v1/status.json',dict(status='FAILED',error=str(e),pid=os.getpid(),utc_unix=time.time()))
        raise
