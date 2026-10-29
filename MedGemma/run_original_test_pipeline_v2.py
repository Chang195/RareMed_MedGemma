"""Original pipeline with approved cursor pagination and BioC identity compatibility, plus history exclusion."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from collect_independent_test import exclusions

HOME = Path(__file__).resolve().parent
SOURCE = HOME.parent / 'RareMed-Pairs'
RUN = HOME / 'original-pipeline-new-test-v2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


def source_files():
    spec = importlib.util.spec_from_file_location('original_release', SOURCE / 'tools/release_project.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    paths = mod.files()
    mod.check(paths)
    return paths


def prepare():
    from pipeline_compatibility_patch import prepare_patched
    ex = exclusions()
    files = prepare_patched(SOURCE, RUN, source_files(), ex)
    return files, ex


def filter_queue(rows, excluded):
    return [r for r in rows if r['article_id'] not in excluded]


def verify_filtered(ex):
    cfg = read(RUN / 'configs/discovery_1000.json')
    target = RUN / 'data/manifests' / (cfg['run_id'] + '_sources.json')
    receipt = RUN / 'local_run/history_filter_receipt.json'
    if not receipt.exists() or not target.exists() or read(receipt)['filtered_sha256'] != sha(target):
        raise ValueError('Run search first through this wrapper. No verified filtered search manifest.')
    rows = read(target)
    if any(r['article_id'] in ex['article_ids'] for r in rows):
        raise ValueError('Historical article overlap')
    if not rows:
        raise ValueError('No new articles under unchanged search. Do not widen queries/pages without a separately approved protocol.')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', help='check or an unchanged stage from original pipeline.py')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-api', action='store_true')
    a = ap.parse_args()
    files, ex = prepare()
    if a.stage == 'check':
        print(json.dumps({'status': 'PASS', 'identical_original_files': sum(not r['changed'] for r in files.values()), 'patched_files': [n for n,r in files.items() if r['changed']],
            'excluded_historical_articles': len(ex['article_ids']), 'original_configs_unchanged': True,
            'original_prompts_and_schema_unchanged': True, 'output_root': str(RUN),
            'network_calls': 0, 'paid_api_calls': 0}, indent=2))
        return
    # Ask the original parser for every stage; no replacement stage implementations.
    command = [sys.executable, str(RUN / 'pipeline.py'), a.stage]
    if a.dry_run:
        command.append('--dry-run')
    if a.allow_api:
        command.append('--allow-api')
    eligibility = RUN / 'local_run/extraction_eligibility.json'
    if a.stage in {'extract-plan', 'extract'} and eligibility.exists():
        subset = read(eligibility)
        manifest = RUN / subset['eligible_manifest']
        if sha(manifest) != subset['eligible_sha256']:
            raise ValueError('Eligible extraction subset changed')
        if a.stage == 'extract' and not a.dry_run and not a.allow_api:
            raise SystemExit('Paid extraction requires explicit --allow-api; no request sent.')
        command = [sys.executable, str(RUN/'scripts/run_deepseek_batch.py'), '--figures', str(manifest),
                   '--run-id', 'deepseek_collection_v3', '--limit', '1000']
        if a.stage == 'extract-plan' or a.dry_run:
            command.append('--dry-run')
    if a.stage not in {'list', 'search'} and not a.dry_run:
        verify_filtered(ex)
    if a.stage == 'collect' and not a.dry_run:
        cfg = read(RUN / 'configs/acquire_v3.json')
        manifest = RUN / cfg['screened_manifest']
        if manifest.exists():
            snapshot = RUN / 'local_run/manifest_snapshots' / (sha(manifest)+'.jsonl')
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                shutil.copy2(manifest, snapshot)
    if a.stage == 'search' and not a.dry_run:
        cfg = read(RUN / 'configs/discovery_1000.json')
        target = RUN / 'data/manifests' / (cfg['run_id'] + '_sources.json')
        original = RUN / 'local_run/unfiltered_search_sources.json'
        if original.exists():
            verify_filtered(ex)
            print('Frozen search reused. Original queries/pageSize/order unchanged; cursor pagination enabled.'); return
        subprocess.run(command, cwd=RUN, check=True)
        rows = read(target)
        save(original, rows)
        filtered = filter_queue(rows, set(ex['article_ids']))
        save(target, filtered)
        save(RUN / 'local_run/history_filter_receipt.json', {
            'unfiltered_sha256': sha(original), 'filtered_sha256': sha(target),
            'searched_articles': len(rows), 'historical_removed': len(rows) - len(filtered),
            'new_articles': len(filtered), 'historical_overlap_after_filter': 0,
            'only_method_change': 'Approved cursor pagination; remove historical article IDs before unchanged fulltext screening; BioC identity alias compatibility.',
            'search_receipt_note': 'Original search receipt describes unfiltered discovery; this receipt records the sole subsequent queue change.'})
        print(json.dumps({'searched_articles': len(rows), 'historical_removed': len(rows)-len(filtered),
                          'new_articles': len(filtered), 'historical_overlap_after_filter': 0}, indent=2))
        return
    raise SystemExit(subprocess.run(command, cwd=RUN).returncode)


if __name__ == '__main__':
    main()
