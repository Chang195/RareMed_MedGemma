"""Isolated, unpaid candidate discovery using the existing RareMed-Pairs collector."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'RareMed-Pairs'
OUT = HERE / 'independent-test-collection-v1'
QUERIES = [
    '(CADASIL OR "NOTCH3") AND (MRI OR "magnetic resonance") AND OPEN_ACCESS:Y',
    '("sporadic cerebral small vessel disease" OR "Binswanger" OR CARASIL OR "HTRA1" OR "multiple sclerosis") AND ("case report" OR "case series") AND (MRI OR "magnetic resonance") AND OPEN_ACCESS:Y',
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    if p.suffix == '.jsonl':
        return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(p)


def exclusions():
    lock = SOURCE / 'metadata/collection_lock.json'
    if not lock.exists():
        raise ValueError('Missing historical collection lock: ' + str(lock))
    paths = [lock]
    # Frozen full figure pool and all prior train/validation/test labels, not only selected arms.
    for name in ['diagnosis-sft-v0.1', 'caption-pilot-v0.1', 'RareMed_CADASIL_SFT_Mixture_Experiments/inputs']:
        paths.extend((HERE / name).glob('*.jsonl'))
    # Also include any original search snapshots still present in the transferred pipeline.
    paths.extend((SOURCE / 'data/raw/search').glob('*.json'))
    paths.extend((SOURCE / 'data/manifests').glob('*sources.json'))
    ids, images, files = set(), set(), []

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {'sha256', 'image_sha256', 'file_sha256'} and isinstance(v, str) and re.fullmatch('[a-fA-F0-9]{64}', v):
                    images.add(v.lower())
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            ids.update(re.findall(r'\bPMC\d+\b', value, re.I))

    for p in sorted(set(paths)):
        walk(load(p))
        files.append({'path': str(p), 'sha256': sha(p)})
    ids = {s.upper() for s in ids}
    if not set(load(lock)['article_ids']).issubset(ids):
        raise ValueError('Historical article exclusion check failed')
    return {'article_ids': sorted(ids), 'prior_asset_hashes': sorted(images), 'source_files': files,
            'coverage': 'Known local search lock plus available source snapshots and prior dataset provenance. Does not establish absence of cross-article patient duplicates or model pretraining exposure.'}


def collector():
    sys.path.insert(0, str(SOURCE / 'scripts'))
    m = importlib.import_module('collect')
    # Reuse the implementation while keeping every new output outside historical directories.
    m.ROOT = OUT
    return m


def main():
    raise SystemExit('DEPRECATED: User requires unchanged original pipeline. Use run_original_test_pipeline.py instead. This prior expanded-query collector is disabled.')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['check', 'search', 'collect'])
    ap.add_argument('--limit', type=int, default=10, help='First N frozen candidates, not N additional articles')
    ap.add_argument('--pages', type=int, default=5, help='Bounded pages per query, 100 records/page')
    a = ap.parse_args()
    if not 1 <= a.limit <= 200 or not 1 <= a.pages <= 50:
        ap.error('limit must be 1..200; pages must be 1..50')
    if not SOURCE.is_dir():
        raise ValueError('Pipeline folder not found: ' + str(SOURCE))
    m = collector()
    ex = exclusions()
    protocol = {'queries': QUERIES, 'exclusions_sha256': hashlib.sha256(json.dumps(ex, sort_keys=True).encode()).hexdigest(),
                'collector_sha256': sha(SOURCE / 'scripts/collect.py'),
                'license_rules_sha256': sha(SOURCE / 'scripts/license_rules.py'),
                'wrapper_sha256': sha(Path(__file__)), 'license_policy': 'exclude_by_nc_nd',
                'training_ready': False, 'paid_api_calls': 0}
    if a.stage == 'check':
        OUT.mkdir(exist_ok=True)
        write(OUT / 'preflight.json', {'protocol': protocol, 'exclusions': ex})
        print(json.dumps({'status': 'PREFLIGHT PASS', 'excluded_articles': len(ex['article_ids']),
                          'historical_hashes': len(ex['prior_asset_hashes']), 'output': str(OUT),
                          'note': 'No network, no downloads, no model calls.'}, indent=2))
        return
    frozen = OUT / 'protocol.json'
    if frozen.exists() and load(frozen) != protocol:
        raise ValueError('Protocol/source/exclusions changed. Preserve this run and choose a new OUT directory.')
    if a.stage == 'search':
        candidate_file = OUT / 'candidates.json'
        if candidate_file.exists():
            print('Frozen search already exists. Use collect, or a new OUT for a new search.'); return
        write(frozen, protocol)
        write(OUT / 'exclusions.json', ex)
        dl = m.Downloader()
        hits, skipped, paging = {}, set(), []
        for qi, query in enumerate(QUERIES):
            cursor = '*'
            for page in range(a.pages):
                url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode({
                    'query': query, 'format': 'json', 'pageSize': 100, 'resultType': 'core',
                    'cursorMark': cursor, 'sort': 'FIRST_PDATE_D desc'})
                response = load(dl.get(url, OUT / f'data/raw/search/query{qi}-page{page}.json'))
                rows = response.get('resultList', {}).get('result', [])
                paging.append({'query': query, 'page': page, 'hitCount': response.get('hitCount'), 'returned': len(rows)})
                for r in rows:
                    aid = r.get('pmcid', '').upper()
                    if not re.fullmatch('PMC[0-9]+', aid):
                        continue
                    if aid in ex['article_ids']:
                        skipped.add(aid); continue
                    item = hits.setdefault(aid, {'article_id': aid, 'title': r.get('title'), 'doi': r.get('doi'),
                        'year': r.get('pubYear'), 'article_license_hint': r.get('license', ''),
                        'query_indices': [], 'status': 'unreviewed_candidate', 'training_ready': False})
                    if qi not in item['query_indices']:
                        item['query_indices'].append(qi)
                print(f'Query {qi+1}, page {page+1}: {len(rows)} hits, {len(hits)} new article candidates', flush=True)
                nxt = response.get('nextCursorMark')
                if not rows or not nxt or nxt == cursor:
                    break
                cursor = nxt
        candidates = sorted(hits.values(), key=lambda r: (min(r['query_indices']), -(int(r.get('year') or 0)), r['article_id']))
        assert not set(hits) & set(ex['article_ids'])
        write(candidate_file, candidates)
        write(OUT / 'search_report.json', {'new_candidates': len(candidates), 'old_articles_rejected': len(skipped),
            'previous_article_overlap': 0, 'paging': paging, 'pages_per_query_cap': a.pages,
            'search_is_bounded_not_systematic': True, 'diagnoses_not_inferred_from_queries': True})
        print(f'SEARCH COMPLETE: {len(candidates)} new article candidates, {len(skipped)} old articles rejected. No final test labels assigned.')
        return
    if not frozen.exists():
        raise ValueError('Run check and search first')
    candidates = load(OUT / 'candidates.json')
    dl = m.Downloader()
    results, all_figures = [], []
    for item in candidates[:a.limit]:
        aid = item['article_id']
        if aid in ex['article_ids']:
            raise ValueError('Old article reached acquisition: ' + aid)
        checkpoint = OUT / f'checkpoints/{aid}.json'
        if checkpoint.exists() and load(checkpoint)['status'].get('status') != 'failed':
            saved = load(checkpoint)
            for fig in saved['figures']:
                for asset in fig.get('assets', []):
                    if sha(OUT / asset['path']) != asset['sha256']:
                        raise ValueError('Resumed image hash mismatch')
        else:
            print('Collecting ' + aid, flush=True)
            try:
                figs, status = m.collect_article(aid, dl, mri_only=True,
                    license_policy=protocol['license_policy'], license_hint=item.get('article_license_hint', ''))
                saved = {'figures': figs, 'status': status}
            except Exception as exc:
                saved = {'figures': [], 'status': {'article_id': aid, 'status': 'failed', 'error': str(exc)}}
            write(checkpoint, saved)
        results.append(saved['status'])
        for fig in saved['figures']:
            duplicate = any(asset.get('sha256') in ex['prior_asset_hashes'] for asset in fig.get('assets', []))
            all_figures.append({**fig, 'prior_exact_asset_match': duplicate,
                'independent_test_status': 'quarantine_old_asset' if duplicate else 'pending_case_label_rights_and_near_duplicate_review',
                'training_ready': False, 'test_ready': False})
        print(aid + ': ' + saved['status']['status'], flush=True)
    target = OUT / 'candidate_figures.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in all_figures), encoding='utf8')
    report = {'attempted_articles': len(results), 'figures': len(all_figures),
        'old_asset_quarantined': sum(r['prior_exact_asset_match'] for r in all_figures),
        'failures': sum(r.get('status') == 'failed' for r in results), 'results': results,
        'test_ready': 0, 'paid_api_calls': 0, 'output_sha256': sha(target),
        'next': 'Review real brain MRI, original case linkage, source diagnosis, license, pixel text and near-duplicate/cross-article case overlap. Do not evaluate models to select candidates.'}
    write(OUT / 'collection_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
