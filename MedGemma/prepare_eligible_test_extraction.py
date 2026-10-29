"""Exclude user-declined long inputs and missing-image figures, preserving raw records."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import sys
import run_original_test_pipeline_v2 as runner
from pipeline_runtime_repair import atomic_dump


def main():
    root = runner.RUN
    runner.prepare()
    source = root / 'data/processed/collection_v3_figures.jsonl'
    plan = runner.read(root / 'reports/deepseek_collection_v3_plan.json')
    long_ids = {aid for aid, reason in plan['preflight_issues'].items()
                if reason.startswith('Prompt exceeds pilot character limit')}
    if len(long_ids) != len(plan['preflight_issues']):
        raise ValueError('Unexpected preflight issues; do not silently exclude')
    rows = [json.loads(s) for s in source.read_text(encoding='utf8').split('\n') if s.strip()]
    kept, excluded = [], []
    for row in rows:
        reasons = []
        if row['article_id'] in long_ids:
            reasons.append('user_excluded_overlong_article_input')
        if not row.get('assets'):
            reasons.append('user_excluded_missing_image')
        if reasons:
            excluded.append({'sample_id': row['sample_id'], 'article_id': row['article_id'], 'reasons': reasons})
        else:
            for asset in row['assets']:
                if runner.sha(root / asset['path']) != asset['sha256']:
                    raise ValueError('Image hash mismatch')
            kept.append(row)
    out = root / 'data/processed/collection_v3_extract_eligible_figures.jsonl'
    record = root / 'local_run/extraction_eligibility.json'
    if record.exists():
        raise ValueError('Subset already prepared; use extract-plan, do not overwrite')
    backup = root / 'local_run' / ('extraction-input-backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir()
    shutil.copy2(root / 'reports/deepseek_collection_v3_plan.json', backup / 'original_extraction_plan.json')
    if (root / 'reports/linkage_preparation_receipt.json').exists():
        shutil.copy2(root / 'reports/linkage_preparation_receipt.json', backup / 'linkage_preparation_receipt.json')
    # No external inference has run. Preserve prepared inputs before regenerating the same prompt template.
    for aid in {r['article_id'] for r in kept}:
        folder = root / 'data/model_inputs' / aid
        shutil.copytree(folder, backup / aid)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in kept), encoding='utf8')
    log = root / 'reports/eligible_linkage_preparation.log'
    with log.open('w', encoding='utf8') as stream:
        subprocess.run([sys.executable, str(root / 'scripts/link_cases.py'), 'prepare', '--figures', str(out)],
                       cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    receipt = {'source_sha256': runner.sha(source), 'eligible_manifest': out.relative_to(root).as_posix(),
        'eligible_sha256': runner.sha(out), 'source_figures': len(rows), 'eligible_figures': len(kept),
        'eligible_assets': sum(len(r['assets']) for r in kept),
        'eligible_articles': len({r['article_id'] for r in kept}), 'excluded_figures': len(excluded),
        'overlong_articles': sorted(long_ids), 'exclusions': excluded, 'backup': str(backup),
        'prompt_schema_and_model_parameters_changed': False, 'network_calls': 0, 'paid_api_calls': 0}
    atomic_dump(record, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='exclusions'}, indent=2))


if __name__ == '__main__':
    main()
