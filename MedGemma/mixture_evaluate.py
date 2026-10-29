"""Evaluate the frozen Base/A/B/C mixture experiment; no training or downloads."""
import os
from pathlib import Path
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    env=os.environ.copy()
    env.update(HF_HOME=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
      HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',
      PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',WANDB_DISABLED='true')
    raise SystemExit(subprocess.call([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/mixture_evaluation.py'),*sys.argv[1:]],cwd=ROOT,env=env))
