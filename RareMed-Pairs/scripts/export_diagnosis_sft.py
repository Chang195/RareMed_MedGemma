"""Export a portable image-only, source-diagnosis SFT research dataset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from PIL import Image
from medcpt_local import ROOT, sha
from dedup_leakage import rows, write_rows, save, safe_path
from build_caption_pilot import make_messages
from dataset_leakage import audit_records

QUESTION = 'What is the most likely diagnosis suggested by this brain MRI figure? Give the diagnosis only.'


def training_target(row):
    if row['diagnosis_family'] == 'needs_label_review':
        return None
    if row['diagnosis_family'] == 'CADASIL_author_reported':
        return 'CADASIL'
    return row['proposed_linkage']['diagnosis_name'].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT/'data/datasets/diagnosis_split_v1')
    ap.add_argument('--output', type=Path, default=ROOT/'data/datasets/diagnosis-sft-v0.1')
    args = ap.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Output already exists; use a new version directory to preserve the frozen package.')
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary['audit_failures']:
        raise ValueError('Source split leakage check failed')
    ds = json.loads((source/'dedup/summary.json').read_text(encoding='utf-8'))
    if sha(source/'dedup/records.jsonl') != ds['records_sha256'] or sha(source/'dedup/duplicate_pairs.jsonl') != ds['relations_sha256']:
        raise ValueError('Source deduplication files changed')
    canonical = {r['sample_id']: r for r in rows(source/'dedup/records.jsonl')
                 if r['inventory_queue'] == 'candidate_diagnosis_task' and r['canonical_image_sample_id'] == r['sample_id']}
    staging, held = [], []
    seen = set()
    for split in ['train', 'validation', 'test']:
        for row in rows(source/(split+'.jsonl')):
            sid = row['sample_id']
            if sid in seen or canonical.get(sid) != row or row['proposed_split'] != split:
                raise ValueError('Split manifest identity mismatch')
            seen.add(sid)
            target = training_target(row)
            if not target:
                held.append({'sample_id': sid, 'reason': 'diagnosis_name_requires_review', 'source_split': split})
                continue
            if len(row['assets']) != 1:
                held.append({'sample_id': sid, 'reason': 'multiple_assets_require_explicit_image_selection', 'source_split': split})
                continue
            staging.append((split, row, target))
    if seen != canonical.keys():
        raise ValueError('Split manifests do not cover canonical source')
    (output/'images').mkdir(parents=True)
    exported, provenance = [], []
    for split, row, target in staging:
        asset = row['assets'][0]
        original = safe_path(ROOT, asset['path'])
        if sha(original) != asset['sha256']:
            raise ValueError('Image changed')
        with Image.open(original) as im:
            im.verify()
        with Image.open(original) as im:
            suffix = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tif', 'WEBP': '.webp'}.get(im.format)
            if not suffix:
                raise ValueError('Unsupported original image format')
        relative = 'images/'+row['sample_id']+suffix
        shutil.copyfile(original, safe_path(output, relative))
        record = {'sample_id': row['sample_id'], 'image': relative,
            'messages': make_messages(QUESTION, target),
            'metadata': {'split': split, 'article_id': row['article_id'], 'split_group_id': row['split_group_id'],
                'image_sha256': asset['sha256'], 'pixel_sha256': row['dedup_asset_fingerprints'][0]['pixel_sha256'],
                'diagnosis_family': row['diagnosis_family'], 'target_origin': 'unreviewed_model_extracted_author_report',
                'source_diagnosis_name': row['proposed_linkage']['diagnosis_name'],
                'training_ready': False, 'research_experiment_candidate': True}}
        exported.append(record)
        provenance.append({'sample_id': row['sample_id'], 'source_url': row['source_url'],
            'license': row['license'], 'license_evidence': row['license_evidence'],
            'original_caption': row['caption'], 'source_text': row.get('source_text'),
            'proposed_linkage': row['proposed_linkage'], 'source_input_sha256': row.get('source_input_sha256')})
    exposure = json.loads((ROOT/'configs/known_model_exposure.json').read_text(encoding='utf-8'))
    # Classification labels can legitimately repeat across sets. Audit grouping on
    # message-free copies; audit answer-in-input on the actual fixed prompt below.
    grouping_audit = audit_records([{k:v for k,v in r.items() if k != 'messages'} for r in exported],
        [p for p in rows(source/'dedup/duplicate_pairs.jsonl') if p['conservative_group_edge']], exposure=exposure)
    input_checks = all(r['messages'] == make_messages(QUESTION, r['messages'][1]['content'][0]['text']) for r in exported)
    if grouping_audit['failures'] or not input_checks:
        raise ValueError('Export audit failed')
    files = {}
    for split in ['train', 'validation', 'test']:
        path = output/(split+'.jsonl')
        write_rows(path, [r for r in exported if r['metadata']['split'] == split])
        files[path.name] = sha(path)
    write_rows(output/'provenance.jsonl', provenance)
    save(output/'held.json', held)
    save(output/'audit.json', {'grouping': grouping_audit, 'fixed_input_prompt_only': input_checks,
        'note': 'Repeated class labels are expected, not evidence of contamination. Pixel text/OCR, PHI and model pretraining contamination not checked.'})
    manifest = {'dataset_version': output.name, 'task': 'image_only_source_reported_diagnosis',
        'question': QUESTION, 'counts': dict(Counter(r['metadata']['split'] for r in exported)),
        'diagnosis_families': {s: dict(Counter(r['metadata']['diagnosis_family'] for r in exported if r['metadata']['split']==s)) for s in ['train','validation','test']},
        'held': len(held), 'files_sha256': files, 'source_records_sha256': ds['records_sha256'],
        'script_sha256': sha(Path(__file__)), 'training_ready': False, 'new_api_calls': 0,
        'limitations': ['Image-only diagnoses may not be identifiable without clinical context; labels are source diagnoses, not image-proven truth.',
        'Other diagnoses retain their reported names; this is not an adjudicated CADASIL vs sporadic-CSVD benchmark.',
        'Small draft evaluation groups; no robust diagnostic performance claim.',
        'Source text and captions are in provenance only; do not concatenate them into model inputs.',
        'Rights, image PHI/OCR, panel attribution and clinical review remain pending.']}
    save(output/'manifest.json', manifest)
    (output/'README.md').write_text('''# Diagnosis SFT research package

Copy this entire folder to the training computer. Use train.jsonl for SFT,
validation.jsonl for development, and keep test.jsonl fixed for baseline/post-SFT comparison.

Each row contains an image path and user/assistant messages. Open the image as PIL RGB
before applying the MedGemma chat template. Train loss on assistant tokens only;
mask user, padding and image tokens. Use the same prompt/preprocessing before and after SFT.
Metadata and provenance are NOT model inputs. The test assistant answer is a reference
only: remove it before generation. Do not train on validation or test rows.

CADASIL aliases are normalized to CADASIL; other source-reported diagnoses retain
their original extracted strings. This is an experimental multiclass package, not
a clinically validated binary benchmark. Source diagnoses may require non-image tests.
Check manifest.json, audit.json and held.json before use. Test images retain published
annotations; no OCR deidentification, rights clearance or independent clinical audit done.
Do not upload the source images publicly before completing rights/privacy review.
''', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
