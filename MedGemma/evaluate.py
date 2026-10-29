"""User-started paired evaluation. Never trains or changes existing checkpoints."""
import os
from pathlib import Path
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import re

def main():
    root=Path(__file__).resolve().parent; python=root/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    p=argparse.ArgumentParser(description='Prepare fixed protocol, run base/SFT evaluations, report paired changes')
    p.add_argument('stage',choices=['deps','prepare','run','report'])
    p.add_argument('--run',default='diagnosis-night-01')
    p.add_argument('--eval',default='evaluation-v1')
    p.add_argument('--task',choices=['diagnosis','mmlu_clinical','medqa','medmcqa','qa_all'],default='diagnosis')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--model',choices=['base','sft','both'],default='both')
    a=p.parse_args()
    if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',n) for n in [a.run,a.eval]): p.error('Use simple experiment names')
    if a.stage=='deps':
        return subprocess.call([str(root/'.tools/bin/uv.exe'),'pip','install','--target',str(root/'.eval-deps'),'--no-deps','pyarrow==19.0.1'])
    out=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'runs'))/a.run/a.eval
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',
        TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',WANDB_DISABLED='true')
    command=[str(python),'-u',str(root/'scripts/paired_evaluation.py'),a.stage,'--run',a.run,'--eval',a.eval,
        '--task',a.task,'--split',a.split,'--model',a.model]
    path=out/(a.stage+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.log')
    print('Log:',path,flush=True)
    with path.open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace')
        try:
            for line in child.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            return child.wait()
        except KeyboardInterrupt:
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.terminate(); child.wait()
            print('Interrupted. Rerun identical command to continue saved predictions.'); return 130

if __name__=='__main__': raise SystemExit(main())
