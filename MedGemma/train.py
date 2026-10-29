"""VS Code Python entry for prepare/train/resume/status. No evaluation generation."""
import os
from pathlib import Path
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

def main():
    root=Path(__file__).resolve().parent
    python=root/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    p=argparse.ArgumentParser(description='Diagnosis QLoRA: preserve base, train only, evaluate later')
    p.add_argument('stage',choices=['prepare','train','resume','status'])
    p.add_argument('--run',default='diagnosis-night-01')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run): p.error('Use a simple experiment name')
    local=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local')))))
    out=local/'runs'/a.run
    command=[str(python),'-u',str(root/'scripts/diagnosis_train.py'),a.stage,'--out',str(out)]
    env=os.environ.copy()
    env.update(RAREMED_LOCAL_ROOT=str(local),HF_HOME=str(local/'huggingface'),HF_HUB_DISABLE_TELEMETRY='1',
        DO_NOT_TRACK='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    if a.stage=='status': return subprocess.call(command,cwd=root,env=env)
    out.mkdir(parents=True,exist_ok=True)
    log=out/(a.stage+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.log')
    print('Output:',out,'\nLog:',log,flush=True)
    with log.open('w',encoding='utf-8') as f:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace')
        try:
            for line in child.stdout:
                print(line,end='',flush=True); f.write(line); f.flush()
            code=child.wait()
        except KeyboardInterrupt:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.terminate(); child.wait()
            print('Stopped; resume from last completed checkpoint with: python train.py resume --run',a.run)
            return 130
    print('Stage PASS' if code==0 else 'Stage FAILED; see log:',log)
    return code

if __name__=='__main__': raise SystemExit(main())
