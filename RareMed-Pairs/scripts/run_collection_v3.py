"""Reprocess the fixed 995-article discovery pool under the user's v3 rules.

Reviews retained. Only CC BY-NC-ND is excluded by license. Disease/MRI signals
are assessed in full text. No image/patient deduplication, no paid API calls.
Existing downloads are reused; outputs and checkpoints use new names.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from collect import ROOT, dump, digest
from audit_licenses import read_metadata
from license_rules import collection_decision


def build_candidates(candidates, metadata):
    rows=[]
    for item in candidates:
        aid=item['article_id']
        hint=metadata[aid].get('license') or ''
        excluded=not collection_decision('',hint)['collect']
        rows.append({**item,'queue':'excluded_license_by_nc_nd' if excluded else 'fulltext_required',
                     'article_license_hint':hint,'review_exclusion':False,
                     'policy_version':'collection-v3','training_ready':False})
    return rows


def run_stage(script, arguments, log_path=None):
    command=[sys.executable,str(ROOT/'scripts'/script)]+arguments
    if log_path:
        with log_path.open('w',encoding='utf-8') as log:
            return subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    return subprocess.run(command,cwd=ROOT).returncode


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run',action='store_true',help='Show selection only; do not run downloads or rewrite data')
    args=ap.parse_args()
    source=ROOT/'data/manifests/brain_mri_discovery_1000_sources.json'
    candidates=json.loads(source.read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/'configs/discovery_1000.json').read_text(encoding='utf-8'))
    metadata,hashes=read_metadata(cfg)
    rows=build_candidates(candidates,metadata)
    print(json.dumps({'input_articles':len(rows),'review_excluded':0,
        'metadata_license_queues':dict(Counter(r['queue'] for r in rows)),
        'license_rule':'exclude CC BY-NC-ND; retain other/unknown declarations as collection candidates',
        'image_patient_deduplication':False,'paid_api_calls':0,'dry_run':args.dry_run},indent=2),flush=True)
    if args.dry_run:
        return
    path=ROOT/'data/manifests/brain_mri_all_candidates_v3.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    dump(ROOT/'reports/collection_v3_policy.json',{'input_articles':len(rows),'review_excluded':0,
        'excluded_license_family':'CC BY-NC-ND','deduplication_stage':'deferred',
        'unknown_license':'retain with pending rights status','source_sha256':digest(source),
        'metadata_sha256':hashes,'collection_is_not_redistribution_approval':True})
    print('STEP 1/3: full-text screening; reviews retained',flush=True)
    if run_stage('screen_fulltext.py',['--config','configs/fulltext_screening_v3.json']):
        raise SystemExit('Full-text stage failed. Inspect its output before continuing.')
    print('STEP 2/3: image/caption collection; exclude BY-NC-ND only',flush=True)
    code=run_stage('acquire_screened.py',['--config','configs/acquire_v3.json'])
    receipt_path=ROOT/'reports/collection_v3_receipt.json'
    if not receipt_path.exists():
        raise SystemExit('No acquisition receipt produced.')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    manifest=ROOT/'data/manifests/brain_mri_fulltext_v3.jsonl'
    figures=ROOT/'data/processed/collection_v3_figures.jsonl'
    if receipt.get('manifest_sha256')!=digest(manifest) or receipt.get('output_sha256')!=digest(figures):
        raise SystemExit('Acquisition outputs do not match the current inputs.')
    if code not in (0,1):
        raise SystemExit('Acquisition stopped unexpectedly; inspect receipt.')
    print('STEP 3/3: prepare model inputs; no API calls',flush=True)
    if receipt['figure_candidates']:
        log=ROOT/'reports/collection_v3_model_preparation.log'
        if run_stage('link_cases.py',['prepare','--figures',str(figures)],log):
            raise SystemExit('Model-input preparation failed. See '+str(log))
    report={'input_articles':len(rows),'review_excluded':0,
        'metadata_license_excluded':sum(r['queue']=='excluded_license_by_nc_nd' for r in rows),
        'fulltext_queues':json.loads((ROOT/'reports/brain_mri_fulltext_v3_receipt.json').read_text(encoding='utf-8'))['queues'],
        'acquisition_outcomes':dict(Counter(r['status'] for r in receipt['results'])),
        'figure_records':receipt['figure_candidates'],'decoded_images':receipt['decoded_images'],
        'new_api_calls':0,'training_ready':0,'image_patient_deduplication':'not_run',
        'acquisition_return_code':code,'rights_status':'source licenses retained; release approval not inferred'}
    dump(ROOT/'reports/collection_v3_summary.json',report)
    print('\nCOLLECTION V3 COMPLETE\n'+json.dumps(report,indent=2))


if __name__=='__main__':
    main()
