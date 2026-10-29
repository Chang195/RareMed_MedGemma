"""Version a machine-collected candidate set without claiming reviewed diagnoses."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from collect import ROOT, dump, digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='fulltext_batch_66_v4')
    ap.add_argument('--version', default='candidate-v0.1')
    ap.add_argument('--links',type=Path,help='Optional validated, unreviewed model proposal manifest')
    args=ap.parse_args()
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]+', v) for v in (args.run,args.version)):
        raise ValueError('Invalid version or run')
    source=ROOT/'data/processed'/(args.run+'_figures.jsonl')
    receipt_path=ROOT/'reports'/(args.run+'_receipt.json')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    link_map={}
    if args.links:
        from link_cases import validate
        for line in args.links.read_text(encoding='utf-8').split('\n'):
            if not line.strip():
                continue
            link=json.loads(line)
            key=(link['article_id'],link['figure_id'])
            if key in link_map:
                raise ValueError('Duplicate model linkage entry')
            response_path=(ROOT/link['response_path']).resolve()
            if not response_path.is_relative_to(ROOT) or digest(response_path)!=link['response_sha256']:
                raise ValueError('Model linkage response provenance mismatch')
            inp=ROOT/'data/model_inputs'/link['article_id']/'input.json'
            if digest(inp)!=link['input_sha256']:
                raise ValueError('Model linkage input changed')
            request=json.loads(inp.read_text(encoding='utf-8'))
            response=json.loads(response_path.read_text(encoding='utf-8'))
            errors,_=validate(response,request)
            matching=[r for r in response['figures'] if r['figure_id']==link['figure_id']]
            if errors or len(matching)!=1 or matching[0]!=link['proposed_linkage']:
                raise ValueError('Unvalidated or altered model linkage')
            link_map[key]=link
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').split('\n') if line.strip()]
    folder=ROOT/'data/releases'/args.version
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        raise ValueError('Release version already exists; choose a new --version')
    seen_ids=set()
    groups={}
    for row in rows:
        if row['sample_id'] in seen_ids:
            raise ValueError('Repeated sample ID')
        seen_ids.add(row['sample_id'])
        for asset in row['assets']:
            path=(ROOT/asset['path']).resolve()
            if not path.is_relative_to(ROOT) or digest(path)!=asset['sha256']:
                raise ValueError('Image hash/path mismatch')
            groups.setdefault(asset['sha256'],[]).append(row['sample_id'])
    output=[]
    for row in rows:
        if not row['assets']:
            continue
        record=dict(row)
        record.update({'dataset_version':args.version,'human_review':'skipped_by_user',
                       'disease_label':None,'disease_linkage_status':'not_batch_model_verified',
                       'training_ready':False,'split':'unassigned',
                       'patient_identity_status':'unresolved',
                       'exact_duplicate_groups':[a['sha256'] for a in row['assets']
                                                 if len(groups[a['sha256']])>1]})
        # Only attach existing extraction results if their input and evidence remain valid.
        model_dir=ROOT/'data/model_inputs'/row['article_id']
        response_path=model_dir/'response.json'
        if response_path.exists() and response_path.stat().st_size:
            from link_cases import validate
            request=json.loads((model_dir/'input.json').read_text(encoding='utf-8'))
            response=json.loads(response_path.read_text(encoding='utf-8-sig'))
            errors,_=validate(response,request)
            requested=next((f for f in request['candidate_figures'] if f['figure_id']==row['figure_id']),None)
            if not errors and requested and requested['caption']==row['caption']:
                match=next(f for f in response['figures'] if f['figure_id']==row['figure_id'])
                record['proposed_linkage']=match
                record['proposed_linkage_response_sha256']=digest(response_path)
                record['disease_linkage_status']='model_proposed_unreviewed'
        link=link_map.get((row['article_id'],row['figure_id']))
        if link:
            record['proposed_linkage']=link['proposed_linkage']
            record['proposed_linkage_response_sha256']=link['response_sha256']
            record['disease_linkage_status']='model_proposed_unreviewed'
        output.append(record)
    target=folder/'candidates.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    dump(folder/'exact_duplicate_groups.json',{k:v for k,v in groups.items() if len(v)>1})
    manifest={'dataset_version':args.version,'created_at':datetime.now(timezone.utc).isoformat(),
              'source_manifest_sha256':digest(source),'acquisition_receipt_sha256':digest(receipt_path),
              'articles_attempted':receipt['attempted_articles'],
              'article_outcomes':dict(Counter(r['status'] for r in receipt['results'])),
              'articles_with_images':len({r['article_id'] for r in output}),
              'figure_candidates':len(rows),'image_caption_records':len(output),
              'unique_image_hashes':len(groups),'exact_duplicate_groups':sum(len(v)>1 for v in groups.values()),
              'model_proposed_records':sum(r['disease_linkage_status']=='model_proposed_unreviewed' for r in output),
              'human_reviewed':0,'training_ready':0,'independent_patient_count':None,
              'candidate_sha256':digest(target),
              'proposal_manifest_sha256':digest(args.links) if args.links else None,
              'limitations':['MRI and topic signals are textual candidates, not verified image diagnoses.',
                             'Only archived validated model proposals are attached; packaging makes no model calls.',
                             'No patient-level split, synthesis, clinical gold labels or completed privacy review.',
                             'Image paths reference project raw files; release is a local index, not a portable image archive.',
                             'Exact byte deduplication does not resolve visually altered images or duplicate patients.']}
    dump(folder/'manifest.json',manifest)
    (folder/'README.md').write_text('# RareMed-Pairs candidate dataset\n\n'
        'Machine-collected image-caption candidates; human review intentionally skipped.\n'
        'Not a diagnosis benchmark or a training-ready release.\n\n'
        'Read manifest.json for counts and limitations. Disease labels remain null.\n'
        'Where available, proposed_linkage preserves unreviewed model extraction separately.\n'
        'Original article, DOI and license evidence are retained in each record; figure-level rights remain unresolved.\n'
        'Raw image paths are project-relative. Exact duplicate hashes group identical files without deleting provenance.\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
