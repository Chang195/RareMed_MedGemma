"""MultiCaRe metadata audit and review queues. Does not invent diagnoses or train."""
import os
from pathlib import Path
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import re
from collections import Counter,defaultdict
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.eval-deps'))
LOCAL=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local')))))
SOURCE=LOCAL/'data/multicare-source-3.0.1'
OUT=LOCAL/'runs/multicare-subset-v0.1'
RECORD='20416562'

def hash_file(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def fetch(names):
    SOURCE.mkdir(parents=True,exist_ok=True)
    response=requests.get('https://zenodo.org/api/records/'+RECORD,timeout=60); response.raise_for_status()
    record=response.json()
    assert record['metadata']['version']=='3.0.1'
    write(SOURCE/'zenodo-record.json',record)
    files={f['key']:f for f in record['files']}
    for name in names:
        spec=files[name]; destination=SOURCE/name
        if not destination.exists():
            temp=destination.with_suffix(destination.suffix+'.partial')
            # Downloads only the pinned official metadata. No third-party mirror/remote code.
            url=spec['links'].get('self') or spec['links']['content']
            with requests.get(url,stream=True,timeout=(30,90)) as r:
                r.raise_for_status(); downloaded=0; threshold=0
                with temp.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk:
                            f.write(chunk); downloaded+=len(chunk)
                            if downloaded>=threshold:
                                print(name,downloaded,'/',spec['size'],'bytes',flush=True); threshold=downloaded+20*1024*1024
            assert temp.stat().st_size==spec['size']
            algorithm,expected=spec['checksum'].split(':')
            assert hash_file(temp,algorithm)==expected
            temp.rename(destination)
        algorithm,expected=spec['checksum'].split(':')
        assert hash_file(destination,algorithm)==expected
        print('Verified official checksum:',name,flush=True)
    write(SOURCE/'download-inventory.json',{p.name:dict(bytes=p.stat().st_size,sha256=hash_file(p)) for p in SOURCE.iterdir() if p.is_file() and p.suffix in ['.csv','.parquet']})

def inspect():
    import pyarrow.parquet as pq
    for name in ['data_dictionary.csv','captions_and_labels.csv']:
        with (SOURCE/name).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); print(name,'columns:',reader.fieldnames,flush=True)
            for _,row in zip(range(2),reader): print(json.dumps(row,ensure_ascii=False)[:4000],flush=True)
    for name in ['metadata.parquet','cases.parquet','case_images.parquet']:
        if not (SOURCE/name).exists(): continue
        p=pq.ParquetFile(SOURCE/name)
        print(name,'rows',p.metadata.num_rows,'schema',str(p.schema_arrow),flush=True)
        print(json.dumps(next(p.iter_batches(batch_size=1)).to_pylist()[0],ensure_ascii=False,default=str)[:3000],flush=True)

def audit_candidates():
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True,exist_ok=True)
    excluded_articles=set()
    source_splits={}
    for split in ['train','validation','test']:
        p=LOCAL/'data/diagnosis-sft-v0.1'/(split+'.jsonl')
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        articles={r['metadata']['article_id'] for r in rows}
        excluded_articles|=articles; source_splits[split]=sorted(articles)
    metadata={r['article_id']:r['article_metadata'] for r in pq.read_table(SOURCE/'metadata.parquet').to_pylist()}
    cases={}
    keyword_articles=set()
    pattern=re.compile(r'\bcadasil\b|\bnotch\s*3\b|cerebral autosomal dominant arteriopathy',re.I)
    for row in pq.read_table(SOURCE/'cases.parquet').to_pylist():
        for case in row['cases'] or []:
            cases[case['case_id']]=case
            if pattern.search(case['case_text'] or ''): keyword_articles.add(row['article_id'])
        if pattern.search(json.dumps(metadata.get(row['article_id'],{}),ensure_ascii=False)):
            keyword_articles.add(row['article_id'])
    figure_links=defaultdict(list)
    for row in pq.read_table(SOURCE/'case_images.parquet').to_pylist():
        for case in row['case_images'] or []:
            for figure in case['case_image_list'] or []:
                figure_links[figure['image_id']].append((case['case_id'],figure))
    with (SOURCE/'captions_and_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        images=list(csv.DictReader(f))
    for r in images:
        if pattern.search(r['caption'] or ''): keyword_articles.add(r['file'].split('_')[0])
    counts=Counter(); pools=defaultdict(list); exclusions=[]
    for r in images:
        article=r['file'].split('_')[0]; counts['total_images']+=1
        if r['image_subtype']=='mri' and r['radiology_region']=='head': counts['raw_head_mri']+=1
        reasons=[]
        if article in excluded_articles: reasons.append('existing_dataset_article_overlap')
        if article in keyword_articles: reasons.append('CADASIL_NOTCH3_keyword_article_hold_not_a_diagnosis')
        if r['license'] not in ['CC BY','CC0']: reasons.append('license_outside_conservative_initial_allowlist')
        article_license=metadata.get(article,{}).get('license')
        if article_license not in ['CC BY','CC0']: reasons.append('article_license_missing_or_outside_initial_allowlist')
        if r['image_type']!='radiology': reasons.append('outside_initial_radiology_scope')
        if r['image_component'].lower()!='undivided': reasons.append('upstream_panel_crop_needs_review')
        if not 30<=len(r['caption'].strip())<=1500: reasons.append('caption_length_outside_initial_limits')
        cid=r['patient_id']; case=cases.get(cid)
        links=figure_links.get(r['main_image'],[])
        unique_cases={x[0] for x in links}
        if case is None or not links or unique_cases!={cid}: reasons.append('missing_or_ambiguous_case_figure_link')
        if reasons:
            counts.update(reasons)
            exclusions.append(dict(file_id=r['file_id'],article_id=article,reasons=reasons))
            continue
        pool='brain_mri_review' if r['image_subtype']=='mri' and r['radiology_region']=='head' else 'general_radiology_caption_review'
        filename=r['file']; upstream=f'{filename[:4]}/{filename[:5]}/{filename}'
        record=dict(candidate_id=r['file_id'],article_id=article,case_id=cid,source_version='3.0.1',zenodo_record=RECORD,
            image_file=filename,upstream_archive=filename[:4]+'.zip',upstream_archive_path=upstream,
            main_image=r['main_image'],upstream_image_component=r['image_component'],
            source_url=f'https://pmc.ncbi.nlm.nih.gov/articles/{article}/',image_license=r['license'],article_license=article_license,
            article_title=metadata.get(article,{}).get('title'),image_subtype=r['image_subtype'],region=r['radiology_region'],
            caption_verbatim=r['caption'],case_text_verbatim=case['case_text'],
            figure_text_references=[ref for _,fig in links for ref in (fig['text_references'] or [])],
            pool=pool,diagnosis_target=None,caption_target_candidate=r['caption'],
            training_ready=False,review_status='pending_image_and_source_review',
            review_required=['image_decode_and_hash','image_caption_alignment','brain_vs_other_head_anatomy',
                'figure_case_diagnosis_evidence','image_only_answerability','upstream_processing','PHI_OCR','license_terms'],
            no_automatic_negative_label=True)
        pools[pool].append(record)
    selected={}; used=set()
    for pool,rows in sorted(pools.items()):
        chosen=[]
        for r in sorted(rows,key=lambda r:hashlib.sha256(('42'+r['candidate_id']).encode()).hexdigest()):
            if r['article_id'] in used: continue
            used.add(r['article_id']); chosen.append(r)
            if len(chosen)==300: break
        selected[pool]=chosen
        path=OUT/(pool+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen),encoding='utf-8')
    (OUT/'exclusions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exclusions),encoding='utf-8')
    write(OUT/'article-exclusion-list.json',dict(existing_dataset=source_splits,target_keyword_articles=sorted(keyword_articles)))
    review=[]
    for pool,rows in selected.items():
        for r in rows[:30]:
            review.append(dict(candidate_id=r['candidate_id'],pool=pool,article_id=r['article_id'],case_id=r['case_id'],
                source_url=r['source_url'],image_file=r['image_file'],caption_verbatim=r['caption_verbatim'],
                image_caption_match='',brain_MRI_confirmed='',author_diagnosis_verbatim='',diagnosis_evidence_quote='',
                image_only_answerable='',license_review='',PHI_OCR_review='',include_exclude_hold='',reviewer='',notes=''))
    with (OUT/'first-60-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(review[0])); writer.writeheader(); writer.writerows(review)
    result=dict(stage='METADATA_AUDIT_COMPLETE_VISUAL_AND_CLINICAL_REVIEW_PENDING',source_record=RECORD,source_version='3.0.1',
        counts=dict(counts),source_articles=len(metadata),source_cases=len(cases),existing_excluded_articles=len(excluded_articles),
        target_keyword_articles=len(keyword_articles),eligible_pool_counts={k:len(v) for k,v in pools.items()},
        selected_candidate_counts={k:len(v) for k,v in selected.items()},first_review_rows=len(review),
        selected_articles=len(used),training_ready_samples=0,images_downloaded=0,
        policy=dict(seed=42,max_selected_per_article=1,pool_cap=300,
            initial_licenses=['CC BY','CC0'],exclude_all_old_dataset_articles=True,
            exclude_keyword_articles_from_generic_ablation_pool=True,require_undivided_upstream=True,
            no_diagnosis_inference_from_taxonomy=True),
        limitations=['Image taxonomy head/MRI is not a diagnosis or proof of brain anatomy.',
            'No visual review, decoded-pixel duplicate check, OCR, clinical adjudication or license clearance yet.',
            'Keyword absence does not prove non-CADASIL; these are candidates, not validated negatives.',
            'No SFT JSONL exported; diagnosis_target remains null; no generated or rewritten medical targets.'])
    write(OUT/'audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['fetch','inspect','audit'])
    p.add_argument('--all-metadata',action='store_true'); a=p.parse_args()
    if a.stage=='fetch':
        names=['data_dictionary.csv','captions_and_labels.csv','metadata.parquet']
        if a.all_metadata: names+=['cases.parquet','case_images.parquet']
        fetch(names)
    elif a.stage=='inspect': inspect()
    else: audit_candidates()

if __name__=='__main__': main()
