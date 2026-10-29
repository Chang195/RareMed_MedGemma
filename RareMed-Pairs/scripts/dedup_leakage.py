"""Local image/text duplicate candidates, conservative split groups and leakage audit."""
import argparse
import base64
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from html import escape
import io
import json
from pathlib import Path
import re
from medcpt_local import ROOT, sha
from inventory_dataset import dhash
from dataset_leakage import UnionFind, normalize, audit_records


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def safe_path(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Path outside intended root')
    return path


@lru_cache(maxsize=1)
def dct_matrix():
    import numpy as np
    n = np.arange(32)
    matrix = np.cos(np.pi * (2*n[None, :]+1) * np.arange(8)[:, None]/64)
    matrix[0] *= 1/(2**0.5)
    return matrix


def phash(image):
    import numpy as np
    from PIL import Image
    array = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (dct_matrix() @ array @ dct_matrix().T).ravel()[1:]  # DC excluded: 63 bits.
    median = np.median(low)
    value = 0
    for bit in low > median: value = (value << 1) | int(bit)
    return f'{value:016x}'


def fingerprints(path):
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    size = image.size
    pixel_hash = hashlib.sha256(str(size).encode() + b'RGB' + image.tobytes()).hexdigest()
    variants = {}
    # Conservative border-crop hypotheses, not arbitrary panel-crop detection.
    for fraction, name in [(1.0, 'full'), (0.9, 'center90'), (0.75, 'center75')]:
        w, h = size
        dx, dy = int(w*(1-fraction)/2), int(h*(1-fraction)/2)
        crop = image.crop((dx, dy, w-dx, h-dy))
        variants[name] = phash(crop)
        variants[name+'_mirror'] = phash(ImageOps.mirror(crop))
    gray = image.convert('L').resize((32, 32))
    return {'pixel_sha256': pixel_hash, 'size': list(size), 'dhash64': dhash(image),
            'phash63_variants': variants, 'low_information': float(np.asarray(gray).std()) < 5}


def decoded_pixel_hash(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    return hashlib.sha256(str(image.size).encode()+b'RGB'+image.tobytes()).hexdigest()


def near_score(a, b):
    best = min(((int(ha, 16)^int(hb, 16)).bit_count(), na, nb)
               for na, ha in a['phash63_variants'].items() for nb, hb in b['phash63_variants'].items())
    return best


def text_similarity(a, b):
    ta, tb = re.findall(r'\w+', normalize(a)), re.findall(r'\w+', normalize(b))
    if min(len(ta), len(tb)) < 12: return 0.0
    if ta == tb: return 1.0
    if min(len(ta), len(tb))/max(len(ta), len(tb)) < .65: return 0.0
    sa = set(zip(ta, ta[1:], ta[2:]))
    sb = set(zip(tb, tb[1:], tb[2:]))
    return len(sa & sb)/len(sa | sb) if sa | sb else 0.0


def draft_split(group_id, seed):
    value = int(hashlib.sha256((str(seed)+':'+group_id).encode()).hexdigest()[:16], 16)/2**64
    return 'train' if value < .8 else 'validation' if value < .9 else 'test'


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def write_rows(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values), encoding='utf-8')


def review_html(pairs, mapping, output, maximum=60):
    from PIL import Image
    cache = {}
    def picture(row):
        asset = row['assets'][0]
        if asset['path'] not in cache:
            with Image.open(safe_path(ROOT, asset['path'])) as image:
                image = image.convert('RGB'); image.thumbnail((560, 440))
                data = io.BytesIO(); image.save(data, format='JPEG', quality=80)
            cache[asset['path']] = 'data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()
        return '<img src="'+cache[asset['path']]+'">'
    cards = []
    for pair in pairs[:maximum]:
        sides = []
        for name in ('sample_a', 'sample_b'):
            row = mapping[pair[name]]
            sides.append('<div><h3>'+escape(row['sample_id'])+'</h3>'+picture(row)+'<pre>'+escape(row['caption'])+'</pre></div>')
        cards.append('<article><h2>'+escape(', '.join(pair['reasons']))+'</h2><p>'+escape(pair['review_status'])+'</p><div class="grid">'+''.join(sides)+'</div></article>')
    output.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>重复候选</title><style>body{font:16px/1.7 'Microsoft YaHei',Arial;margin:25px;background:#f3f6fa}article{background:white;padding:20px;margin:20px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{max-width:100%;max-height:440px}pre{white-space:pre-wrap;font:14px/1.6 Consolas;background:#f4f7fa;padding:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>重复与相似候选</h1><p>仅用于查看；相似不等于同图或同患者，原文件未删除。默认把未解决的相似关系放在同一建议分组，可能过度合并。</p>'''+f'<p>展示前{min(maximum,len(pairs))}对，共{len(pairs)}对；完整结果见duplicate_pairs.jsonl。</p>'+''.join(cards)+'</html>', encoding='utf-8')


def build(args):
    source, output = args.source.resolve(), args.output.resolve()
    upstream = json.loads((source.parent/'summary.json').read_text(encoding='utf-8'))
    if sha(source) != upstream['inventory_sha256']: raise ValueError('Inventory changed')
    records = rows(source)
    mapping = {r['sample_id']: r for r in records}
    if len(mapping) != len(records): raise ValueError('Duplicate sample IDs')
    output.mkdir(parents=True, exist_ok=True)
    exposure = json.loads(args.exposure.read_text(encoding='utf-8'))
    unknown = set(exposure['sample_ids']) - mapping.keys()
    if unknown: raise ValueError('Known exposure IDs not present in inventory')
    identities = {'source_sha256': sha(source), 'script_sha256': sha(Path(__file__)),
        'audit_script_sha256': sha(ROOT/'scripts/dataset_leakage.py'), 'exposure_sha256': sha(args.exposure),
        'phash_distance': args.phash_distance, 'caption_jaccard': args.caption_jaccard, 'seed': args.seed}
    if args.decisions: identities['decisions_sha256'] = sha(args.decisions)
    decisions = {}
    if args.decisions:
        data = json.loads(args.decisions.read_text(encoding='utf-8'))
        if data['source_manifest_sha256'] != sha(source): raise ValueError('Decisions refer to another source')
        for d in data['decisions']:
            pair = tuple(sorted((d['sample_a'], d['sample_b'])))
            if pair in decisions or pair[0] == pair[1] or any(s not in mapping for s in pair):
                raise ValueError('Invalid or duplicate decision pair')
            if d['decision'] not in {'duplicate', 'same_case', 'unrelated', 'uncertain'} or not d.get('reason', '').strip():
                raise ValueError('Decision and reason required')
            decisions[pair] = d
    profiles = {}
    for i, row in enumerate(records, 1):
        found = []
        for asset in row['assets']:
            path = safe_path(ROOT, asset['path'])
            if sha(path) != asset['sha256']: raise ValueError('Image changed: '+row['sample_id'])
            if asset['sha256'] not in profiles:
                profiles[asset['sha256']] = {'sha256': asset['sha256'], **fingerprints(path)}
            found.append(profiles[asset['sha256']])
        row['dedup_asset_fingerprints'] = found
        if i % 100 == 0 or i == len(records): print(f'Fingerprinting {i}/{len(records)}', flush=True)
    pairs = []
    for i, a in enumerate(records):
        for b in records[i+1:]:
            reasons, metrics = [], {}
            exact = False
            for fa in a['dedup_asset_fingerprints']:
                for fb in b['dedup_asset_fingerprints']:
                    if fa['sha256'] == fb['sha256']:
                        reasons.append('exact_file'); exact = True
                    elif fa['pixel_sha256'] == fb['pixel_sha256']:
                        reasons.append('exact_decoded_pixels'); exact = True
                    elif not fa['low_information'] and not fb['low_information']:
                        ratio_a = fa['size'][0]/fa['size'][1]
                        ratio_b = fb['size'][0]/fb['size'][1]
                        if max(ratio_a,ratio_b)/min(ratio_a,ratio_b) <= 1.8:
                            distance, va, vb = near_score(fa, fb)
                            d = (int(fa['dhash64'],16)^int(fb['dhash64'],16)).bit_count()
                            if distance <= args.phash_distance or d <= 4:
                                reasons.append('perceptual_image_candidate')
                                metrics.update(phash63_distance=distance, variant_a=va, variant_b=vb, dhash64_distance=d)
            similarity = text_similarity(a['caption'], b['caption'])
            if similarity >= args.caption_jaccard:
                reasons.append('caption_similarity_candidate'); metrics['caption_trigram_jaccard'] = similarity
            key = tuple(sorted((a['sample_id'], b['sample_id'])))
            decision = decisions.get(key)
            if decision and decision['decision'] in {'same_case', 'duplicate'}:
                reasons.append('reviewed_'+decision['decision'])
            if reasons:
                suppressed = bool(decision and decision['decision'] == 'unrelated' and not exact)
                pairs.append({'sample_a': a['sample_id'], 'sample_b': b['sample_id'], 'reasons': sorted(set(reasons)), 'metrics': metrics,
                    'review_status': 'exact_identity' if exact else decision['decision'] if decision else 'unreviewed_candidate',
                    'conservative_group_edge': not suppressed, 'decision_reason': decision.get('reason') if decision else None})
    image_groups = UnionFind(mapping)
    groups = UnionFind(mapping)
    by_article = defaultdict(list)
    by_patient = defaultdict(list)
    for row in records:
        by_article[row['article_id']].append(row['sample_id'])
        if row.get('case_identity_verified') and row.get('verified_patient_id'):
            by_patient[row['verified_patient_id']].append(row['sample_id'])
        ancestors = row.get('parent_sample_ids', [])
        if isinstance(ancestors, str): ancestors = [ancestors]
        for ancestor in ancestors:
            if ancestor in mapping: groups.union(row['sample_id'], ancestor)
    for bucket in [*by_article.values(), *by_patient.values()]:
        for sid in bucket[1:]: groups.union(bucket[0], sid)
    for pair in pairs:
        if pair['conservative_group_edge']: groups.union(pair['sample_a'], pair['sample_b'])
        if any(r in pair['reasons'] for r in ['exact_file', 'exact_decoded_pixels', 'reviewed_duplicate']):
            image_groups.union(pair['sample_a'], pair['sample_b'])
    image_members = defaultdict(list)
    group_members = defaultdict(list)
    for row in records:
        image_members[image_groups.find(row['sample_id'])].append(row)
        group_members[groups.find(row['sample_id'])].append(row['sample_id'])
    exposed_groups = {groups.find(row['sample_id']) for row in records if row['sample_id'] in exposure['sample_ids'] or row['article_id'] in exposure.get('article_ids', [])}
    for row in records:
        sid = row['sample_id']
        image_group = image_groups.find(sid)
        group = groups.find(sid)
        members = image_members[image_group]
        # Canonical image index only; different captions/labels remain archived.
        keeper = min(members, key=lambda r: (not r['inventory_queue'].startswith('candidate_'),
                     -max(a['size'][0]*a['size'][1] for a in r['dedup_asset_fingerprints']), r['sample_id']))['sample_id']
        row.update(image_duplicate_group=image_group, canonical_image_sample_id=keeper,
                   dedup_role='canonical_image_record' if sid == keeper else 'redundant_image_record_retained',
                   split_group_id=group, split='unassigned',
                   proposed_split=('train' if group in exposed_groups else draft_split(group,args.seed))
                                  if row['inventory_queue'].startswith('candidate_') else 'hold',
                   split_reason='known_smoke_training_exposure' if group in exposed_groups else 'conservative_group_hash_draft',
                   training_ready=False)
    write_rows(output/'records.jsonl', records)
    write_rows(output/'duplicate_pairs.jsonl', pairs)
    save(output/'groups.json', dict(group_members))
    save(output/'review_decisions.template.json', {'source_manifest_sha256': sha(source), 'decisions': []})
    audit = audit_records(records, [p for p in pairs if p['conservative_group_edge']], 'proposed_split', exposure)
    save(output/'draft_split_audit.json', audit)
    review_html(sorted(pairs, key=lambda p: (p['review_status'] != 'exact_identity', p['sample_a'], p['sample_b'])), mapping, output/'review.html')
    candidates = [r for r in records if r['inventory_queue'].startswith('candidate_')]
    summary = {**identities, 'records': len(records), 'caption_task_candidates': len(candidates),
        'exact_or_reviewed_duplicate_groups': sum(len(v)>1 for v in image_members.values()),
        'redundant_image_records_retained': sum(r['dedup_role']=='redundant_image_record_retained' for r in records),
        'candidate_pairs': len(pairs), 'pair_reason_counts': dict(Counter(reason for p in pairs for reason in p['reasons'])),
        'conservative_split_groups': len(group_members), 'largest_group_records': max(map(len, group_members.values())),
        'draft_candidate_splits': dict(Counter(r['proposed_split'] for r in candidates)),
        'smoke_exposure_groups_forced_train': len(exposed_groups),
        'draft_audit_status': audit['status'], 'draft_audit_failures': len(audit['failures']),
        'records_sha256': sha(output/'records.jsonl'), 'relations_sha256': sha(output/'duplicate_pairs.jsonl'),
        'training_ready': 0, 'new_api_calls': 0,
        'limitations': ['No images deleted; similarity is not duplicate or patient confirmation.',
                        'Conservative grouping can overmerge different patients; splits remain drafts.',
                        'Border crops/mirroring handled heuristically; arbitrary panel crops/rotations and cross-paper patients can still be missed.',
                        'No PHI deidentification, final dataset approval, model pretraining contamination audit or clinical accuracy validation.']}
    save(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


def audit(args):
    dataset = args.dataset.resolve()
    records = rows(dataset)
    relations = []
    if args.relations:
        sibling_summary = args.relations.parent/'summary.json'
        if sibling_summary.exists():
            recorded = json.loads(sibling_summary.read_text(encoding='utf-8'))
            if recorded.get('relations_sha256') and sha(args.relations) != recorded['relations_sha256']:
                raise ValueError('Relations file changed')
            if dataset.parent == args.relations.parent and recorded.get('records_sha256') and sha(dataset) != recorded['records_sha256']:
                raise ValueError('Dedup dataset changed')
        relations = [r for r in rows(args.relations) if r.get('conservative_group_edge', True)]
    integrity = []
    for row in records:
        metadata = row.get('metadata', {})
        if row.get('image') and isinstance(row['image'], str):
            assets = [{'path': row['image'], 'sha256': metadata.get('image_sha256')}]
            root = dataset.parent
        else:
            assets = row.get('assets', row.get('images', []))
            root = ROOT
        checked_assets = []
        for asset in assets:
            path = safe_path(root, asset['path'])
            if not path.is_file() or (asset.get('sha256') and sha(path) != asset['sha256']):
                integrity.append({'type': 'image_integrity_failure', 'sample_id': row['sample_id']})
            else:
                checked_assets.append({'sha256': sha(path), 'pixel_sha256': decoded_pixel_hash(path)})
        if assets: row['dedup_asset_fingerprints'] = checked_assets
    exposure = json.loads(args.exposure.read_text(encoding='utf-8'))
    report = audit_records(records, relations, args.split_field, exposure)
    report['failures'].extend(integrity)
    if integrity: report['status'] = 'failed'
    report['dataset_sha256'] = sha(dataset)
    report['exposure_sha256'] = sha(args.exposure)
    report['relations_provided'] = bool(args.relations)
    if not args.relations:
        report['warnings'].append({'type': 'near_duplicate_relations_not_supplied'})
    save(args.report.resolve(), report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'failures','warnings'}}, indent=2))
    print(f'Failures: {len(report["failures"])}; warnings: {len(report["warnings"])}; report: {args.report}')
    if report['failures']: raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--source', type=Path, default=ROOT/'data/datasets/formal_candidates_v1/inventory.jsonl')
    b.add_argument('--output', type=Path, default=ROOT/'data/datasets/dedup_v1_1')
    b.add_argument('--exposure', type=Path, default=ROOT/'configs/known_model_exposure.json')
    b.add_argument('--decisions', type=Path)
    b.add_argument('--phash-distance', type=int, default=6)
    b.add_argument('--caption-jaccard', type=float, default=.85)
    b.add_argument('--seed', type=int, default=20261026)
    a = sub.add_parser('audit')
    a.add_argument('--dataset', type=Path, required=True)
    a.add_argument('--relations', type=Path)
    a.add_argument('--exposure', type=Path, default=ROOT/'configs/known_model_exposure.json')
    a.add_argument('--split-field', choices=['split', 'proposed_split'], default='split')
    a.add_argument('--report', type=Path, default=ROOT/'reports/dataset_leakage_audit.json')
    args = ap.parse_args()
    if args.command == 'build':
        if not 0 <= args.phash_distance <= 12 or not .5 <= args.caption_jaccard <= 1:
            ap.error('Invalid similarity thresholds')
        build(args)
    else:
        audit(args)


if __name__ == '__main__':
    main()
