"""Allowlisted, code-only release. Never traverses raw datasets or credential caches."""
import os
from pathlib import Path
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'RareMed-Pairs'
OUTPUT=ROOT/'GitHub_release_v2'
ENTRIES=[
    'run.py','train.py','evaluate.py','reparse.py','mixture_sft.py','mixture_evaluate.py',
    'run_medqa_direct_v2.py','queue_tumor_after_medqa.py','resume_mixture_evaluation.py',
    'run_original_test_pipeline_v2.py','collect_independent_test.py',
    'pipeline_compatibility_patch.py','pipeline_runtime_repair.py',
    'repair_collection_checkpoint.py','migrate_collection_manifest.py',
    'prepare_eligible_test_extraction.py','audit_new_test_candidates.py',
    'run_other_disease_search_1000.py','run_tumor_fulltext.py','run_tumor_images.py','pmc_cloud_images.py',
    'requirements.txt','build_github_release.py',
    'test_pmc_cloud_images.py','test_pipeline_runtime_repair.py',
]
DOCS=['VSCODE.zh-CN.md','MEDPIX_RATIO_STUDY.zh-CN.md','MIXTURE_SFT_VSCODE.zh-CN.md',
      'MIXTURE_EVALUATION.zh-CN.md','MEDQA_DIRECT_V2.zh-CN.md','TUMOR_BINARY_QUEUE.zh-CN.md',
      'TUMOR_CLOUD_ACQUISITION.zh-CN.md','COLLECTION_RECOVERY.zh-CN.md','ORIGINAL_PIPELINE_V2.zh-CN.md']


def sha(data):return hashlib.sha256(data).hexdigest()


def portable_python(data):
    """Only release copies change. Replace machine drive literals, not frozen originals."""
    text=data.decode('utf-8-sig')
    tree=ast.parse(text)
    offsets=[0]
    for line in text.splitlines(True):offsets.append(offsets[-1]+len(line))
    replacements=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value.replace('\\','/')
            base=str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))))
            if value==base or value.startswith(base+'/'):
                # AST columns are UTF-8 byte offsets, convert to character offsets safely.
                lines=text.splitlines(True)
                start=offsets[node.lineno-1]+len(lines[node.lineno-1].encode()[:node.col_offset].decode())
                end=offsets[node.end_lineno-1]+len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
                tail=value[len(base):].lstrip('/')
                expr="str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local')))"
                expr+=(' / '+repr(tail) if tail else '')+')'
                replacements.append((start,end,expr))
    for a,b,replacement in sorted(replacements,reverse=True):text=text[:a]+replacement+text[b:]
    if replacements:
        tree=ast.parse(text)
        pos=0
        for node in tree.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                pos=node.end_lineno
            elif isinstance(node,ast.ImportFrom) and node.module=='__future__':pos=node.end_lineno
            else:break
        lines=text.splitlines(True)
        lines.insert(pos,'import os\nfrom pathlib import Path\n')
        text=''.join(lines)
    ast.parse(text)
    return text.encode('utf8'),len(replacements)


def main():
    if OUTPUT.exists():
        raise ValueError('Release directory already exists. Preserve it and choose a new output version; no overwrite.')
    spec=importlib.util.spec_from_file_location('source_release',SOURCE/'tools/release_project.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    source_files=mod.files();mod.check(source_files)
    OUTPUT.mkdir()
    records={}
    def add(src,relative,portable=False):
        raw=src.read_bytes();data=raw;edits=0
        if portable and src.suffix=='.py':data,edits=portable_python(data)
        if src.suffix=='.md' and portable:
            text=data.decode('utf-8-sig')
            text=text.replace(str(ROOT),'$PROJECT_ROOT').replace(str(ROOT).replace('\\','/'),'$PROJECT_ROOT')
            data=text.encode('utf8')
        target=OUTPUT/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records[relative]={'source_sha256':sha(raw),'release_sha256':sha(data),'bytes':len(data),
                           'portable_path_literal_edits':edits,'original_changed':False}
    for src in source_files:add(src,'RareMed-Pairs/'+src.relative_to(SOURCE).as_posix())
    for name in ENTRIES:add(ROOT/name,'MedGemma/'+name,True)
    for src in sorted((ROOT/'scripts').glob('*.py')):add(src,'MedGemma/scripts/'+src.name,True)
    add(ROOT/'MedGemma_SFT_Figures/plot_sft_results.py','MedGemma/MedGemma_SFT_Figures/plot_sft_results.py',True)
    for name in DOCS:add(ROOT/name,'MedGemma/docs/'+name,True)
    # These auxiliary modules supply imports/configuration for current entry points, but their CLIs are deprecated.
    for name in ['run_other_disease_search.py']:
        add(ROOT/name,'MedGemma/'+name,True)
    (OUTPUT/'README.md').write_text(PUBLIC_README,encoding='utf8')
    (OUTPUT/'MedGemma/README.md').write_text(MODEL_README,encoding='utf8')
    (OUTPUT/'MedGemma/local-settings.example.ps1').write_text(
        "$env:RAREMED_LOCAL_ROOT = Join-Path $env:USERPROFILE 'RareMed_MedGemma_Local'\n"
        "# Configure HF login / DEEPSEEK_API_KEY interactively in your local terminal.\n",encoding='utf8')
    (OUTPUT/'RELEASE_SCOPE.md').write_text(SCOPE,encoding='utf8')
    (OUTPUT/'.gitignore').write_text(IGNORE,encoding='utf8')
    (OUTPUT/'.github/workflows').mkdir(parents=True)
    (OUTPUT/'.github/workflows/check.yml').write_text(CI,encoding='utf8')
    problems=[];compiled=0
    secret=re.compile(r'\b(?:ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{20,})\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    forbidden={'.jsonl','.safetensors','.pt','.pth','.ckpt','.bin','.zip','.pptx'}
    for p in OUTPUT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(OUTPUT).as_posix()
        if p.suffix in forbidden:problems.append({'path':rel,'issue':'forbidden_payload_extension'})
        if p.stat().st_size>10*1024*1024:problems.append({'path':rel,'issue':'unexpected_large_file'})
        if p.suffix in {'.py','.md','.json','.txt','.yml','.yaml','.ps1'}:
            text=p.read_text(encoding='utf-8-sig')
            if secret.search(text):problems.append({'path':rel,'issue':'possible_secret_value_not_printed'})
            if re.search(r'[A-Za-z]:[\\/]Users[\\/]49851',text):problems.append({'path':rel,'issue':'personal_absolute_path'})
        if p.suffix=='.py':
            compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');compiled+=1
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if not problems else 'FAIL',
        'compiled_python_files':compiled,'problems':problems,'raw_data_included':False,
        'weights_included':False,'api_requests_sent':0,'github_push_performed':False,
        'code_license_selected':False,'files':records}
    (ROOT/'github-release-audit').mkdir(exist_ok=True)
    (ROOT/'github-release-audit/build_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    if problems:raise ValueError(json.dumps(problems,indent=2))
    manifest={p.relative_to(OUTPUT).as_posix():{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
              for p in OUTPUT.rglob('*') if p.is_file()}
    (OUTPUT/'RELEASE_MANIFEST.json').write_text(json.dumps({'files':manifest,'code_only_release':True},indent=2),encoding='utf8')
    print(json.dumps({'status':'PASS','output':str(OUTPUT),'files':len(manifest)+1,
                      'compiled_python_files':compiled,'private_data_included':False},indent=2))


PUBLIC_README='''# RareMed-Pairs / MedGemma experiments

Public-literature image–case–source diagnosis curation and local MedGemma QLoRA experiments.
The two directories must remain siblings. No patient-level clinical validation is claimed.

## Layout

- `RareMed-Pairs/`: original curation pipeline, frozen source locks, prompts, schemas and aggregate extraction results.
- `MedGemma/`: local model training/evaluation and versioned collection adapters.

No original medical images, article/case text, JSONL datasets, API payloads, predictions, tokens or model weights are included.
The original curation environment and model environment have different dependency pins. Use separate Python 3.11 environments.
Do not install both requirements files into one environment.

## Local paths and credentials

Set `RAREMED_LOCAL_ROOT` locally for model/data/cache storage before model commands.
Release copies replace legacy G-drive literals with that variable or a home-directory default.
Some historical experiment commands still require local manifests/checkpoints not distributed here.
Do not interpret the presence of a script as a turnkey reproduction of unpublished runs.
Hugging Face acceptance/login is local. DeepSeek reads `DEEPSEEK_API_KEY` from the environment and requires explicit API opt-in.
Never commit `.env`, credentials or the generated working folders.

## Recommended collection order

In `MedGemma/`, use `run_other_disease_search_1000.py` for GBM/astrocytoma first-page searches,
`run_tumor_fulltext.py` for original-rule BioC screening, and `run_tumor_images.py` for official PMC Cloud images.
Each supports a check/plan phase. See `MedGemma/docs/TUMOR_CLOUD_ACQUISITION.zh-CN.md`.
`collect_independent_test.py` supplies a history helper but its expanded-query CLI is disabled.
The old `run_other_disease_search.py` is retained for dependency compatibility only. Do not use its multi-page CLI.
Historical CADASIL v2 pagination is separate from the single-page tumor protocol.

## Scientific boundaries

Source-reported diagnosis is not independently adjudicated clinical ground truth.
Article, figure, case group and patient are different units. Keep source/case groups isolated and review cross-article duplicates.
Use train only for fitting, validation for selection, and a frozen independent test for final evaluation.
The current collection adapters do not automatically create a clinically reviewed test set.
Direct-answer MedQA differs from the official reasoning protocol.

## Checks

In `MedGemma/`: install Pillow, then `python -m unittest test_pmc_cloud_images.py test_pipeline_runtime_repair.py -v`.
One optional local-run integration test skips when its unpublished fixture is absent.
In a separate environment for `RareMed-Pairs/`: install its requirements and run `python tools/run_tests.py`.

See `RELEASE_SCOPE.md` for limitations and rights. No open-source license has been selected.
'''
MODEL_README='''# MedGemma model experiments

Model: google/medgemma-1.5-4b-it, project-pinned revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b.
The proven local recipe uses NF4 double quantization, FP32 compute, SDPA and language q_proj/v_proj LoRA rank 4.
This is an experimental 6 GB GPU setup, not the official full-precision evaluation configuration.

Install a compatible PyTorch build separately, then requirements.txt in a dedicated model environment.
The data-pipeline requirements in ../RareMed-Pairs are separate.
For non-model collection commands, avoid loading GPU weights or changing the proven model environment.

Use the repository-level README for current entry points. docs/ contains historical method notes,
some of which reference local artifacts that are intentionally excluded. Their existence is not evidence
that those datasets, results or model checkpoints are available in this code-only release.
No medical reference answers have been generated or relabeled by this release builder.
'''
SCOPE='''# Release scope and rights

This package includes explicitly selected source code, configurations, documentation, and the original
RareMed-Pairs pipeline's already allowlisted aggregate extraction metrics/plots. It excludes raw medical
images, patient/case text, source-response caches, benchmark question files, per-item predictions,
models/adapters/checkpoints, local virtual environments, and unpublished slide decks.

No code license has been selected. Publishing to GitHub does not automatically grant an open-source license.
Third-party source licenses and model terms remain applicable. This package does not relicense medical materials.

The release builder preserves frozen local experiment source files. Only release copies replace machine-specific
G-drive path literals with RAREMED_LOCAL_ROOT / a home-directory default. Full before/after hashes stay in
the local build audit. Original RareMed-Pairs source files remain byte-identical.

Historical replay and end-to-end evaluation require separately obtained data, accepted model terms,
and local run manifests/checkpoints. CPU-only checks do not validate GPU model performance.
Release path adaptations change source hashes. Use the original local source to resume a frozen historical
experiment. Do not replace its source with this portable release or bypass its protocol checks.
Secret scanning is a bounded pattern scan, not a guarantee that every possible sensitive value is detected.
'''
IGNORE='''__pycache__/
*.py[cod]
.venv*/
.runtime/
.cache/
.eval-deps/
.figure-build/
.chart-data-*/
.vscode/
.runtime_tmp/
data/
reports/
models/
runs/
output/
archives/
original-pipeline-*/
independent-test-collection-*/
pmc-cloud-cache-*/
caption-pilot-*/
diagnosis-sft-*/
RareMed_CADASIL_SFT_Mixture_Experiments/inputs/
RareMed_CADASIL_SFT_Mixture_Experiments/images/
RareMed_CADASIL_SFT_Mixture_Experiments/dataset_*/
RareMed_CADASIL_SFT_Mixture_Experiments/manifests/
mixture-evaluation-*/
tumor-binary-evaluation-*/
medqa-*/
MedGemma_SFT_Figures/
!MedGemma/MedGemma_SFT_Figures/
MedGemma/MedGemma_SFT_Figures/*
!MedGemma/MedGemma_SFT_Figures/plot_sft_results.py
GitHub_release*/
github-release-audit/
*.safetensors
*.pt
*.pth
*.ckpt
*.bin
*.jsonl
*.pptx
*.zip
*.log
*.tmp
*.part
.env
.env.*
!.env.example
*.pem
*.key
token
stored_tokens
id_rsa*
id_ed25519*
'''
CI='''name: Code checks
on: [push, pull_request]
jobs:
  adapters:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install Pillow==12.0.0
      - run: python -m compileall -q MedGemma RareMed-Pairs
      - run: python -m unittest test_pmc_cloud_images.py test_pipeline_runtime_repair.py -v
        working-directory: MedGemma
  original-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install -r requirements.txt
        working-directory: RareMed-Pairs
      - run: python tools/run_tests.py
        working-directory: RareMed-Pairs
'''


if __name__=='__main__':main()
