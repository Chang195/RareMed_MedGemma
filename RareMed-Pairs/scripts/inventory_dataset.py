"""Inventory all image-caption candidates before formal dataset selection. No model/API calls."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from medcpt_local import ROOT, sha
from preview_dataset import ANIMAL, NONPATIENT, local_path

MRI = re.compile(r'\b(?:MRI|MRIs|magnetic\s+resonance|FLAIR|DWI|ADC|SWI|T[12](?:WI|W)?|susceptibility.weighted)\b', re.I)
PANELS = re.compile(r'\([A-Za-z](?:\s*[,–-]\s*[A-Za-z])*\)|\bmulti.panel\b|\bpanels?\s+[A-Za-z]\b', re.I)
SERIAL = re.compile(r'\b(?:serial|follow.up|longitudinal|over\s+time|at\s+ages|baseline\s+and)\b', re.I)
TARGET_RISK = re.compile(r'\b(?:CADASIL|NOTCH3|genetic|diagnos\w*|encephalopathy)\b', re.I)


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def dhash(image):
    from PIL import Image
    pixels = list(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return f'{value:016x}'


def screen(row, assets):
    link = row.get('proposed_linkage') or {}
    text = row['caption'] + ' ' + (link.get('case_label') or '') + ' ' + (link.get('diagnosis_name') or '')
    flags = []
    if ANIMAL.search(text): flags.append('animal_text_signal')
    if NONPATIENT.search(row['caption']): flags.append('nonpatient_plot_or_diagram_text_signal')
    if not MRI.search(row['caption']): flags.append('no_explicit_mri_caption_cue')
    if len(PANELS.findall(row['caption'])) >= 2 or re.search(r'\bmulti.panel\b', row['caption'], re.I):
        flags.append('multi_panel_caption_signal')
    if SERIAL.search(row['caption']): flags.append('serial_comparison_caption_signal')
    if TARGET_RISK.search(row['caption']): flags.append('caption_may_contain_nonvisual_or_diagnostic_target')
    if '\ufffd' in row['caption']: flags.append('caption_unicode_replacement_character')
    if re.search(r'\b(?:arrow|arrowhead)\w*\b', row['caption'], re.I): flags.append('caption_mentions_annotations')
    if not row['caption'].strip(): flags.append('empty_caption')
    single = link.get('link_status') in {'explicit', 'contextual'} and bool(link.get('case_label'))
    if not link: flags.append('no_validated_extraction_proposal')
    elif not single: flags.append('single_case_not_established')
    if assets and any(a.get('quality_error') for a in assets): flags.append('image_integrity_or_decode_problem')
    if not assets: flags.append('no_image_asset')
    if assets and any(min(a.get('size', [0, 0])) < 192 for a in assets): flags.append('small_image_dimension')
    if any(f in flags for f in ['image_integrity_or_decode_problem', 'no_image_asset', 'empty_caption']):
        queue = 'held_file_or_caption_problem'
    elif 'animal_text_signal' in flags:
        queue = 'held_animal_signal'
    elif 'nonpatient_plot_or_diagram_text_signal' in flags:
        queue = 'held_nonpatient_figure_signal'
    elif 'no_explicit_mri_caption_cue' in flags:
        queue = 'held_modality_uncertain'
    elif not single:
        queue = 'held_case_linkage_uncertain'
    elif any(f in flags for f in ['multi_panel_caption_signal', 'serial_comparison_caption_signal']):
        queue = 'candidate_composite_or_serial_caption_task'
    else:
        queue = 'candidate_single_figure_caption_task'
    return queue, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT/'data/releases/candidate-v0.4/candidates.jsonl')
    ap.add_argument('--output', type=Path, default=ROOT/'data/datasets/formal_candidates_v1')
    ap.add_argument('--near-distance', type=int, default=4, help='dHash Hamming threshold; candidate pairs only, never auto-delete/group')
    args = ap.parse_args()
    if not 0 <= args.near_distance <= 8:
        ap.error('near-distance must be 0..8')
    args.source, args.output = args.source.resolve(), args.output.resolve()
    upstream = json.loads((args.source.parent/'manifest.json').read_text(encoding='utf-8'))
    if sha(args.source) != upstream['candidate_sha256']:
        raise ValueError('Source manifest checksum mismatch')
    source = read_rows(args.source)
    if len({r['sample_id'] for r in source}) != len(source):
        raise ValueError('Duplicate sample IDs')
    args.output.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    output, images, exact = [], {}, defaultdict(list)
    for i, row in enumerate(source, 1):
        assets = []
        for asset in row['assets']:
            path = local_path(asset['path'])
            info = {'path': asset['path'], 'expected_sha256': asset['sha256']}
            try:
                actual = sha(path)
                info['actual_sha256'] = actual
                if actual != asset['sha256']:
                    raise ValueError('image_sha256_mismatch')
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    info.update(size=list(image.size), format=image.format, dhash64=dhash(image))
                images.setdefault(actual, {'sample_id': row['sample_id'], **info})
                exact[actual].append(row['sample_id'])
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                info['quality_error'] = type(exc).__name__ + ': image could not be verified'
            assets.append(info)
        queue, flags = screen(row, assets)
        link = row.get('proposed_linkage') or {}
        single = link.get('link_status') in {'explicit', 'contextual'} and bool(link.get('case_label'))
        provisional_case = row['article_id'] + ':' + link['case_label'] if single and 'animal_text_signal' not in flags else None
        output.append({**row, 'inventory_queue': queue, 'inventory_flags': flags, 'asset_checks': assets,
                       'provisional_case_key': provisional_case, 'case_identity_verified': False,
                       'split_group_id': row['article_id'], 'split': 'unassigned', 'training_ready': False})
        if i % 100 == 0 or i == len(source):
            print(f'Images inventoried: {i}/{len(source)}', flush=True)
    # Near-image hashes are deliberately NOT patient identity or split-group decisions.
    unique = list(images.values())
    pairs = []
    for i, a in enumerate(unique):
        for b in unique[i+1:]:
            distance = (int(a['dhash64'], 16) ^ int(b['dhash64'], 16)).bit_count()
            if distance <= args.near_distance:
                pairs.append({'sample_a': a['sample_id'], 'sample_b': b['sample_id'], 'hamming_distance': distance,
                              'status': 'visual_similarity_candidate_not_duplicate_confirmation'})
    # Article groups are merged only by exact file duplicates; similar MRI anatomy is not enough.
    parent = {r['article_id']: r['article_id'] for r in source}
    article_for_sample = {r['sample_id']: r['article_id'] for r in source}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for members in exact.values():
        for member in members[1:]:
            a, b = find(article_for_sample[members[0]]), find(article_for_sample[member])
            if a != b: parent[max(a, b)] = min(a, b)
    for r in output:
        r['split_group_id'] = find(r['article_id'])
        r['exact_duplicate_peers'] = sorted({sample for a in r['asset_checks'] for sample in exact.get(a.get('actual_sha256'), []) if sample != r['sample_id']})
    target = args.output/'inventory.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output), encoding='utf-8')
    columns = ['sample_id', 'article_id', 'figure_id', 'queue', 'flags', 'case_label', 'diagnosis', 'diagnosis_status', 'link_status', 'caption', 'image_paths', 'split_group_id']
    with (args.output/'inventory.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in sorted(output, key=lambda r: (not r['inventory_queue'].startswith('candidate_'), r['inventory_queue'], r['article_id'])):
            link = r.get('proposed_linkage') or {}
            writer.writerow({**{k:r[k] for k in ['sample_id', 'article_id', 'figure_id', 'caption', 'split_group_id']},
                'queue': r['inventory_queue'], 'flags': '; '.join(r['inventory_flags']), 'case_label': link.get('case_label'),
                'diagnosis': link.get('diagnosis_name'), 'diagnosis_status': link.get('diagnosis_status'),
                'link_status': link.get('link_status'), 'image_paths': '; '.join(a['path'] for a in r['assets'])})
    (args.output/'near_duplicate_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs), encoding='utf-8')
    (args.output/'exact_duplicate_groups.json').write_text(json.dumps({h:v for h,v in exact.items() if len(v)>1}, indent=2), encoding='utf-8')
    candidates = [r for r in output if r['inventory_queue'].startswith('candidate_')]
    summary = {'records': len(output), 'source_articles': len(parent), 'queues': dict(Counter(r['inventory_queue'] for r in output)),
               'caption_task_candidates': len(candidates), 'candidate_source_articles': len({r['article_id'] for r in candidates}),
               'provisional_within_article_case_labels': len({r['provisional_case_key'] for r in candidates if r['provisional_case_key']}),
               'verified_independent_patients': None,
               'candidate_diagnosis_statuses': dict(Counter(r['proposed_linkage']['diagnosis_status'] for r in candidates)),
               'exact_duplicate_groups': sum(len(v)>1 for v in exact.values()), 'near_image_candidate_pairs': len(pairs),
               'split_groups_after_exact_image_linkage': len({r['split_group_id'] for r in output}),
               'training_ready': 0, 'new_api_calls': 0, 'source_sha256': sha(args.source), 'inventory_sha256': sha(target),
               'script_sha256': sha(Path(__file__)),
               'note': 'Text screening and dHash are heuristics. Near pairs are not confirmed duplicates; article/case labels are not verified patient counts. No final SFT selection or split yet.'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
