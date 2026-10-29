"""VS Code launcher for the received 20/50/80% mixture package."""
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
    root=Path(__file__).resolve().parent; python=root/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    p=argparse.ArgumentParser(description='Prepare/check/train three mixture arms independently from original MedGemma')
    p.add_argument('stage',choices=['prepare','check','train','resume','status'])
    p.add_argument('--arm',choices=['A','B','C','all'],default='all')
    p.add_argument('--run',default='mixture-ratios-v1')
    p.add_argument('--data',type=Path,default=root/'RareMed_CADASIL_SFT_Mixture_Experiments')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run): p.error('Use a simple experiment name')
    out=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'runs'))/a.run
    command=[str(python),'-u',str(root/'scripts/mixture_training.py'),a.stage,'--arm',a.arm,'--out',str(out),'--data',str(a.data.resolve())]
    env=os.environ.copy(); env.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),
        HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    if a.stage=='status': return subprocess.call(command,cwd=root,env=env)
    out.mkdir(parents=True,exist_ok=True)
    file=out/(a.stage+'-'+a.arm+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.log')
    print('Data:',a.data.resolve(),'\nResults:',out,'\nLog:',file,flush=True)
    with file.open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        try:
            for line in child.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            code=child.wait()
        except KeyboardInterrupt:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.terminate(); child.wait()
            print('Stopped. Use the same --run with resume --arm all to continue.'); return 130
    print('Stage PASS' if code==0 else 'Stage FAILED; see log:',file)
    return code

if __name__=='__main__': raise SystemExit(main())
