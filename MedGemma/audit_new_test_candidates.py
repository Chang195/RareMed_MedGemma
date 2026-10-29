"""Local review export and image-overlap candidates. No clinical adjudication or split."""
import csv
import json
from pathlib import Path
import sys
from collections import Counter
import run_original_test_pipeline_v2 as runner


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').split('\n') if s.strip()]


def main():
    runner.prepare()
    root = runner.RUN
    sys.path.insert(0, str(root / 'scripts'))
    from dedup_leakage import fingerprints, near_score
    data = rows(root / 'data/datasets/diagnosis_candidates_v1/confirmed_mri_candidates.jsonl')
    eligibility = runner.read(root / 'local_run/extraction_eligibility.json')
    allowed = {r['sample_id'] for r in rows(root / eligibility['eligible_manifest'])}
    if any(r['sample_id'] not in allowed for r in data):
        raise ValueError('User-excluded candidate reached review export')
    prior = runner.read(root / 'local_run/historical_exclusions.json')
    if any(r['article_id'] in prior['article_ids'] for r in data):
        raise ValueError('Previously searched article reached review export')
    out = root / 'data/datasets/independent_test_review_v1'
    out.mkdir(exist_ok=False)
    # Current originals are accessed via immutable SHA-named copies, not old machine paths.
    old_files = {}
    for folder in [runner.HOME / 'RareMed_CADASIL_SFT_Mixture_Experiments/images', runner.HOME / 'caption-pilot-v0.1/images']:
        for p in folder.rglob('*'):
            if p.is_file():
                old_files.setdefault(runner.sha(p), p)
    profiles = {}
    for i, (digest, p) in enumerate(old_files.items(), 1):
        profiles[digest] = fingerprints(p)
        if i % 200 == 0:
            print(f'Historical image fingerprints: {i}/{len(old_files)}', flush=True)
    new_profiles = {}
    pairs = []
    for i, r in enumerate(data, 1):
        for asset in r['assets']:
            p = root / asset['path']
            digest = runner.sha(p)
            if digest != asset['sha256']:
                raise ValueError('New asset hash mismatch')
            profile = new_profiles.setdefault(digest, fingerprints(p))
            for old_sha, old in profiles.items():
                reason, distance, dhash = None, None, None
                if digest == old_sha:
                    reason = 'exact_file'
                elif profile['pixel_sha256'] == old['pixel_sha256']:
                    reason = 'exact_decoded_pixels'
                elif not profile['low_information'] and not old['low_information']:
                    ra = profile['size'][0]/profile['size'][1]
                    rb = old['size'][0]/old['size'][1]
                    if max(ra, rb)/min(ra, rb) <= 1.8:
                        distance, _, _ = near_score(profile, old)
                        dhash = (int(profile['dhash64'],16)^int(old['dhash64'],16)).bit_count()
                        if distance <= 6 or dhash <= 4:
                            reason = 'perceptual_candidate_requires_review'
                if reason:
                    pairs.append({'new_sample_id':r['sample_id'], 'new_sha256':digest,
                        'old_sha256':old_sha, 'old_local_image':str(old_files[old_sha]),
                        'reason':reason, 'phash_distance':distance, 'dhash_distance':dhash,
                        'review_decision':'', 'review_reason':''})
        if i % 20 == 0:
            print(f'New candidate comparisons: {i}/{len(data)}', flush=True)
    fields = ['new_sample_id','new_sha256','old_sha256','old_local_image','reason','phash_distance','dhash_distance','review_decision','review_reason']
    with (out/'cross_history_image_matches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(pairs)
    table = []
    for r in data:
        matches = [p for p in pairs if p['new_sample_id']==r['sample_id']]
        table.append({'sample_id':r['sample_id'],'article_id':r['article_id'],
            'source_diagnosis':r['proposed_linkage']['diagnosis_name'],
            'source_case_label':r['proposed_linkage'].get('case_label'),
            'source_url':r['source_url'],'image_path':str(root/r['assets'][0]['path']),
            'history_exact_overlap':any(p['reason'] in {'exact_file','exact_decoded_pixels'} for p in matches),
            'history_near_match_count':sum(p['reason']=='perceptual_candidate_requires_review' for p in matches),
            'review_brain_mri':'','review_same_case_linkage':'','review_source_diagnosis':'',
            'review_appropriate_control':'','review_pixel_label_leakage':'',
            'review_license':'','review_cross_article_patient_overlap':'',
            'include_final_test':'','reviewer':'','review_notes':''})
    with (out/'candidate_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    positive = [r for r in data if 'CADASIL' in r['proposed_linkage']['diagnosis_name'].upper()]
    report={'candidate_images':len(data),'source_articles':len({r['article_id'] for r in data}),
        'cadasil_name_candidates':len(positive),'cadasil_source_articles':len({r['article_id'] for r in positive}),
        'historical_unique_images_checked':len(profiles),'new_unique_images':len(new_profiles),
        'cross_history_pair_types':dict(Counter(p['reason'] for p in pairs)),
        'matched_new_samples':len({p['new_sample_id'] for p in pairs}),
        'within_new_exact_duplicate_sets':sum(n>1 for n in Counter(a['sha256'] for r in data for a in r['assets']).values()),
        'historical_article_overlap':0,'clinical_review_complete':False,'test_ready':0,
        'independent_patients':None,'new_api_calls':0,'model_evaluation_performed':False,
        'source_summary_sha256':runner.sha(root/'data/datasets/diagnosis_candidates_v1/summary.json'),
        'dedup_implementation_sha256':runner.sha(root/'scripts/dedup_leakage.py'),
        'limitations':['Perceptual matches are review candidates, not proven duplicates.',
            'Comparison covers available local historical images, not all previously searched articles or model pretraining.',
            'Original fingerprint code uses crop/mirror hypotheses in memory for comparison only; original images unchanged.',
            'Old image caption aliases and cross-article patient identity still require review.']}
    (out/'readiness_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
