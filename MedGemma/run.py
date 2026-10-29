"""Python-only entry: project venv, local cache, per-stage console logs."""
import os
from pathlib import Path
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    python = ROOT / '.venv/Scripts/python.exe'
    if not python.exists():
        raise SystemExit('Project .venv is missing.')
    if Path(sys.executable).resolve() != python.resolve():
        return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser(description='RareMed Python-only local workflow')
    parser.add_argument('stage', choices=['validate', 'processor', 'forward', 'smoke'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run', help='Experiment name shared by all stages (default: python-pilot-01)')
    group.add_argument('--out', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--steps', type=int, choices=range(1, 6), default=2)
    args = parser.parse_args()
    local = Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))))))
    name = args.run or 'python-pilot-01'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        parser.error('--run must be a simple folder name, without path separators')
    out = (args.out or local / 'runs' / name).resolve()
    data = (args.data or local / 'data/caption-pilot-v0.1').resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(RAREMED_LOCAL_ROOT=str(local), HF_HOME=str(local / 'huggingface'),
               HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', WANDB_DISABLED='true',
               TOKENIZERS_PARALLELISM='false', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    command = [str(python), '-u', str(ROOT / 'scripts/pipeline.py'), args.stage,
               '--data', str(data), '--out', str(out), '--steps', str(args.steps)]
    path = out / (args.stage + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.log')
    print('Output:', out, '\nLog:', path, flush=True)
    with path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = child.wait()
        except KeyboardInterrupt:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate(); child.wait()
            print('\nStopped. Partial log retained:', path)
            return 130
    print('Stage PASS' if code == 0 else 'Stage FAILED; see log:', path)
    return code

if __name__ == '__main__':
    raise SystemExit(main())
