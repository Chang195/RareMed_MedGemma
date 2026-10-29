"""Prepare diagnosis-specific duplicate groups and draft train/validation/test manifests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
from medcpt_local import ROOT, sha
from dedup_leakage import rows, write_rows, save
from dataset_leakage import audit_records


def diagnosis_family(name):
    # Only recognize positive CADASIL at the start; never match 'CADASIL excluded'.
    text = (name or '').strip()
    uncertain = re.search(r'\b(?:suspected|possible|not confirmed|excluded|unlikely|differential|versus|vs\.?|rule out)\b', text, re.I)
    if uncertain:
        return 'needs_label_review'
    if re.match(r'^CADASIL\b|^cerebral autosomal[ -]dominant arteriopathy', text, re.I):
        return 'CADASIL_author_reported'
    if re.search(r'\bsporadic\b', text, re.I) and re.search(r'small[ -]vessel', text, re.I):
        return 'sporadic_CSVD_author_reported'
    return 'other_reported_diagnosis_needs_control_selection'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'data/datasets/diagnosis_split_v1')
    ap.add_argument('--decisions', type=Path, help='Optional reviewed duplicate decisions from this run')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    output = args.output.resolve()
    source = ROOT/'data/datasets/diagnosis_candidates_v1/all_figures.jsonl'
    upstream = json.loads((source.parent/'summary.json').read_text(encoding='utf-8'))
    if sha(source) != upstream['records_sha256']:
        raise ValueError('Diagnosis candidate source changed')
    original = rows(source)
    # Keep all available images in the duplicate search, including held figures:
    # they can connect two otherwise separate candidate groups.
    inventory = []
    for r in original:
        if not r['assets']:
            continue
        selected = r['selection_status'] == 'single_case_author_confirmed_mri_candidate'
        inventory.append({**r, 'inventory_queue': 'candidate_diagnosis_task' if selected else r['selection_status'],
                          'diagnosis_family': diagnosis_family((r.get('proposed_linkage') or {}).get('diagnosis_name')),
                          'case_identity_verified': False})
    prepared = output/'input'
    prepared.mkdir(parents=True, exist_ok=True)
    write_rows(prepared/'inventory.jsonl', inventory)
    save(prepared/'summary.json', {'inventory_sha256': sha(prepared/'inventory.jsonl'),
                                 'diagnosis_source_sha256': sha(source)})
    if args.prepare_only:
        print('Prepared inventory; no duplicate search or split export executed.')
        return
    command = [sys.executable, str(ROOT/'scripts/dedup_leakage.py'), 'build',
               '--source', str(prepared/'inventory.jsonl'), '--output', str(output/'dedup')]
    if args.decisions:
        command += ['--decisions', str(args.decisions.resolve())]
    subprocess.run(command, check=True)
    records = rows(output/'dedup/records.jsonl')
    candidates = [r for r in records if r['inventory_queue'] == 'candidate_diagnosis_task']
    kept = [r for r in candidates if r['canonical_image_sample_id'] == r['sample_id']]
    relations = rows(output/'dedup/duplicate_pairs.jsonl')
    exposure = json.loads((ROOT/'configs/known_model_exposure.json').read_text(encoding='utf-8'))
    # Audit all records so duplicate paths through held figures are preserved.
    audit = audit_records(records, [p for p in relations if p['conservative_group_edge']], 'proposed_split', exposure)
    save(output/'leakage_audit.json', audit)
    for split in ['train', 'validation', 'test']:
        write_rows(output/(split+'.jsonl'), [r for r in kept if r['proposed_split'] == split])
    write_rows(output/'held_figures.jsonl', [r for r in original if r['selection_status'] != 'single_case_author_confirmed_mri_candidate'])
    write_rows(output/'redundant_image_records.jsonl', [r for r in candidates if r not in kept])
    counts = {s: dict(Counter(r['diagnosis_family'] for r in kept if r['proposed_split'] == s)) for s in ['train','validation','test']}
    summary = {'original_figures': len(original), 'images_searched': len(inventory),
               'diagnosis_candidates': len(candidates), 'canonical_candidate_records': len(kept),
               'split_counts': {s: sum(v.values()) for s,v in counts.items()}, 'split_diagnosis_families': counts,
               'audit_status': audit['status'], 'audit_failures': len(audit['failures']),
               'source_sha256': sha(source), 'new_api_calls': 0, 'training_ready': 0,
               'split_status': 'draft_grouped_manifests_not_model_ready_SFT',
               'limitations': ['Approximate 80/10/10 group hash split, not stratified; inspect class counts before freezing.',
                 'Other diagnoses are not automatically valid negative controls. Original diagnosis strings and evidence retained.',
                 'Context is not yet assigned or screened as model input. Do not feed full source_text to model.',
                 'Perceptual similarity can overmerge groups; review dedup/review.html and group sizes.',
                 'No PHI or rights clearance, clinical verification, or unknown cross-paper patient guarantee.']}
    save(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if audit['failures']:
        raise SystemExit('Leakage audit failed; do not train on these drafts.')


if __name__ == '__main__':
    main()
