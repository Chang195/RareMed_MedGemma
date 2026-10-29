"""Original acquisition search with only disease terms retargeted; no paid API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import run_original_test_pipeline_v2 as original
from collect_independent_test import exclusions
from pipeline_compatibility_patch import patched_bytes
from pipeline_runtime_repair import atomic_dump

TOPICS = {
    'gbm': '(glioblastoma OR "glioblastoma multiforme" OR GBM)',
    'astrocytoma': '(astrocytoma OR astrocytomas)',
}


def history():
    ex = exclusions()
    ids = set(ex['article_ids'])
    evidence = list(ex['source_files'])
    # Include all returned records from previous searches, not only retained candidates.
    for folder in ['original-pipeline-new-test-v1', 'original-pipeline-new-test-v2', 'independent-test-collection-v1']:
        for p in (original.HOME / folder / 'data/raw/search').glob('*.json'):
            if p.name.endswith('.download.json'):
                continue
            response = original.read(p)
            for r in response.get('resultList', {}).get('result', []):
                if r.get('pmcid'):
                    ids.add(r['pmcid'])
            evidence.append({'path': str(p), 'sha256': original.sha(p)})
    return {'article_ids': sorted(ids), 'source_files': evidence,
            'note': 'Frozen before both tumor searches. Articles shared between these new searches remain explicitly cross-listed, not treated as independent cases.'}


def prepare(disease):
    root = original.HOME / ('original-pipeline-'+disease+'-test-v1')
    cfg = original.read(original.SOURCE / 'configs/discovery_1000.json')
    cfg['queries'] = [TOPICS[disease]+' AND (MRI OR "magnetic resonance") AND OPEN_ACCESS:Y']
    hashes = {}
    for p in original.source_files():
        name = p.relative_to(original.SOURCE).as_posix()
        raw = p.read_bytes()
        data = patched_bytes(name, raw)
        if name == 'configs/discovery_1000.json':
            data = json.dumps(cfg, ensure_ascii=False, indent=2).encode('utf8')
        target = root / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Frozen search implementation changed: '+name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = {'original_sha256': hashlib.sha256(raw).hexdigest(),
                        'run_sha256': hashlib.sha256(data).hexdigest(), 'changed': raw != data}
    helper = Path(__file__).with_name('pipeline_runtime_repair.py')
    (root / 'scripts/pipeline_runtime_repair.py').write_bytes(helper.read_bytes())
    local = root / 'local_run'
    local.mkdir(exist_ok=True)
    atomic_dump(local/'file_verification.json', hashes)
    hist = local / 'historical_exclusions.json'
    if not hist.exists():
        atomic_dump(hist, history())
    return root, cfg, original.read(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('disease', choices=TOPICS)
    ap.add_argument('stage', choices=['check', 'search'])
    a = ap.parse_args()
    root, cfg, ex = prepare(a.disease)
    if a.stage == 'check':
        print(json.dumps({'status':'PASS','query':cfg['queries'], 'page_size':cfg['page_size'],
            'excluded_prior_articles':len(ex['article_ids']), 'output':str(root),
            'paid_api_calls':0,'only_disease_terms_changed':True,
            'caution':'Search only. Original fulltext disease rules still target CADASIL and must be explicitly retargeted before any collect stage.'},indent=2))
        return
    target = root / 'data/manifests' / (cfg['run_id']+'_sources.json')
    frozen = root / 'local_run/unfiltered_search_sources.json'
    if frozen.exists():
        print((root / 'local_run/history_filter_receipt.json').read_text(encoding='utf8'))
        return
    print('SEARCH START: '+cfg['queries'][0], flush=True)
    subprocess.run([sys.executable,str(root/'pipeline.py'),'search'],cwd=root,check=True)
    rows = original.read(target)
    atomic_dump(frozen, rows)
    kept = [r for r in rows if r['article_id'] not in ex['article_ids']]
    atomic_dump(target, kept)
    receipt = {'disease':a.disease,'searched_articles':len(rows),
        'historical_removed':len(rows)-len(kept),'new_articles':len(kept),
        'historical_overlap_after_filter':0,'query':cfg['queries'][0],
        'unfiltered_sha256':original.sha(frozen),'filtered_sha256':original.sha(target),
        'paid_api_calls':0,'diagnoses_assigned':0,'output_root':str(root)}
    atomic_dump(root/'local_run/history_filter_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
