"""Revalidate saved acquisition entries against a successfully refreshed fulltext manifest."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import run_original_test_pipeline_v2 as runner
from pipeline_runtime_repair import atomic_dump


def validate(root, checkpoint):
    config = runner.read(root / 'configs/acquire_v3.json')
    if config != checkpoint['config'] or checkpoint.get('exclusion_hashes') != {}:
        raise ValueError('Configuration/exclusion mismatch; no automatic migration')
    manifest = root / config['screened_manifest']
    receipt = runner.read(root / 'reports/brain_mri_fulltext_v3_receipt.json')
    if runner.sha(manifest) != receipt['output_sha256']:
        raise ValueError('Fulltext receipt mismatch')
    rows = [json.loads(s) for s in manifest.read_text(encoding='utf8').split('\n') if s.strip()]
    by_id = {r['article_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate article IDs')
    sys.path.insert(0, str(root / 'scripts'))
    from collect import parse_bioc
    figures, assets = 0, 0
    for aid, saved in checkpoint['completed'].items():
        row = by_id.get(aid)
        if not row or row['queue'] not in config['queues']:
            raise ValueError('Previously processed article no longer eligible: '+aid)
        xml = root / row['fulltext_source_path']
        if runner.sha(xml) != row['source_sha256']:
            raise ValueError('Current fulltext source hash mismatch: '+aid)
        metadata = runner.read(xml.with_name(xml.name+'.download.json'))
        if metadata['sha256'] != runner.sha(xml):
            raise ValueError('Download provenance mismatch: '+aid)
        passages = parse_bioc(xml, aid)
        for fig in saved['figures']:
            if fig['article_id'] != aid:
                raise ValueError('Wrong figure/article association')
            evidence = fig['caption_evidence']
            if root / evidence['path'] != xml or not any(
                    p['offset'] == evidence['offset'] and p['text'] == fig['caption'] for p in passages):
                raise ValueError('Saved caption no longer matches source evidence: '+fig['sample_id'])
            figures += 1
            for asset in fig['assets']:
                if runner.sha(root / asset['path']) != asset['sha256']:
                    raise ValueError('Saved image hash mismatch')
                assets += 1
    return manifest, {'validated_articles': len(checkpoint['completed']),
                      'validated_figures_and_captions': figures, 'validated_image_hashes': assets}


def main():
    root = runner.RUN
    runner.prepare()
    checkpoint_path = root / 'data/manifests/collection_v3_checkpoint.json'
    checkpoint = runner.read(checkpoint_path)
    manifest, evidence = validate(root, checkpoint)
    new_hash = runner.sha(manifest)
    if checkpoint['manifest_sha256'] == new_hash:
        print('Checkpoint already matches current manifest. '+json.dumps(evidence)); return
    folder = root / 'local_run' / ('manifest-migration-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir()
    shutil.copy2(checkpoint_path, folder / checkpoint_path.name)
    shutil.copy2(manifest, folder / manifest.name)
    shutil.copy2(root / 'reports/brain_mri_fulltext_v3_receipt.json', folder / 'fulltext_receipt.json')
    old_hash = checkpoint['manifest_sha256']
    checkpoint['manifest_sha256'] = new_hash
    atomic_dump(checkpoint_path, checkpoint)
    audit = {'status': 'MIGRATED_AFTER_PER_ARTICLE_REVALIDATION', 'previous_manifest_sha256': old_hash,
        'current_manifest_sha256': new_hash, **evidence, 'medical_rules_changed': False,
        'saved_outputs_changed': False, 'network_calls': 0, 'paid_api_calls': 0,
        'old_manifest_snapshot_available': False,
        'limitation': 'Old fulltext manifest was overwritten by original retry stage. Exact whole-manifest diff is unavailable; migration relies on frozen code/config plus explicit source/caption/image verification for every saved article.',
        'backup': str(folder)}
    atomic_dump(folder / 'migration_receipt.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
