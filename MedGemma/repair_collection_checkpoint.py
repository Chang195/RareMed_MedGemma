"""Explicit one-time, verified upgrade of the interrupted v2 run. No network."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from pipeline_runtime_repair import atomic_dump, is_challenge
import pipeline_compatibility_patch as patch
import run_original_test_pipeline_v2 as runner


def main():
    root = runner.RUN
    done = root / 'local_run/runtime_repair_receipt.json'
    if done.exists():
        print(done.read_text(encoding='utf8'))
        return
    verification = runner.read(root / 'local_run/file_verification.json')
    candidates = []
    checkpoint = root / 'data/manifests/collection_v3_checkpoint.json'
    for p in [checkpoint, checkpoint.with_name(checkpoint.name+'.tmp')]:
        if p.exists():
            value = runner.read(p)
            source = root / value['config']['screened_manifest']
            if runner.sha(source) != value['manifest_sha256']:
                raise ValueError('Checkpoint source hash mismatch')
            if value['config'] != runner.read(root / 'configs/acquire_v3.json'):
                raise ValueError('Checkpoint config mismatch')
            allowed = {json.loads(s)['article_id'] for s in source.read_text(encoding='utf8').split('\n')
                       if s.strip() and json.loads(s)['queue'] in value['config']['queues']}
            if not set(value['completed']).issubset(allowed):
                raise ValueError('Unexpected article in recovery checkpoint')
            for result in value['completed'].values():
                for figure in result['figures']:
                    for asset in figure['assets']:
                        if runner.sha(root / asset['path']) != asset['sha256']:
                            raise ValueError('Recovery image hash mismatch')
            candidates.append((p, value))
    if not candidates:
        raise ValueError('No checkpoint to recover')
    best_path, best = max(candidates, key=lambda pv: len(pv[1]['completed']))
    for _, smaller in candidates:
        for aid, result in smaller['completed'].items():
            if best['completed'].get(aid) != result:
                raise ValueError('Recovery checkpoints conflict; no automatic merge')
    for name, entry in verification.items():
        if runner.sha(root / name) != entry['run_sha256']:
            raise ValueError('Run code changed before repair: '+name)
    backup = root / 'local_run' / ('runtime-repair-backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir()
    for p, _ in candidates:
        shutil.copy2(p, backup / p.name)
    shutil.copy2(root / 'scripts/collect.py', backup / 'collect.py')
    shutil.copy2(root / 'local_run/file_verification.json', backup / 'file_verification.json')
    original = runner.SOURCE / 'scripts/collect.py'
    upgraded = patch.patched_bytes('scripts/collect.py', original.read_bytes())
    atomic_dump(backup / 'upgrade_plan.json', {'new_collect_sha256': hashlib.sha256(upgraded).hexdigest(),
        'recovered_from': str(best_path), 'recovered_articles': len(best['completed'])})
    target = root / 'scripts/collect.py'
    target.write_bytes(upgraded)
    helper = Path(__file__).with_name('pipeline_runtime_repair.py')
    shutil.copy2(helper, root / 'scripts/pipeline_runtime_repair.py')
    held = []
    for aid, result in best['completed'].items():
        html = root / 'data/raw' / aid / 'article.html'
        meta = html.with_name(html.name+'.download.json')
        if result['status']['status'] == 'failed' and html.exists() and is_challenge(html):
            if not meta.exists() or runner.read(meta)['sha256'] != runner.sha(html):
                raise ValueError('Challenge cache hash mismatch')
            result['status'] = {**result['status'], 'prior_status': 'failed',
                'status': 'held_access_challenge', 'reason': 'Verified cached browser challenge; no automatic bypass or retry',
                'cached_response_sha256': runner.sha(html), 'training_ready': False}
            held.append(aid)
    atomic_dump(checkpoint, best)
    records, ex = runner.prepare()
    figures = [f for result in best['completed'].values() for f in result['figures']]
    receipt = {'status': 'RECOVERED', 'network_calls': 0, 'paid_api_calls': 0,
        'recovered_articles': len(best['completed']), 'recovered_figures': len(figures),
        'image_hashes_verified': sum(len(f['assets']) for f in figures),
        'access_challenges_held': len(held), 'held_article_ids': held,
        'backup': str(backup), 'checkpoint_sha256': runner.sha(checkpoint),
        'medical_filters_changed': False, 'old_source_pipeline_modified': False}
    atomic_dump(done, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='held_article_ids'}, indent=2))


if __name__ == '__main__':
    main()
