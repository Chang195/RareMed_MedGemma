"""Count MRI/head source labels; never equate head labels to pure brain MRI."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from medcpt_local import ROOT, sha

def main():
    folder=ROOT/'data/benchmarks/four_disease/sources'
    path=folder/'multicare_captions_and_labels.csv'
    record=json.loads((folder/'multicare_record.json').read_text(encoding='utf-8'))
    expected=next(f for f in record['files'] if f['key']=='captions_and_labels.csv')
    if path.stat().st_size!=expected['size']:raise ValueError('Download incomplete')
    checksum=expected.get('checksum','')
    if checksum.startswith('md5:') and hashlib.md5(path.read_bytes()).hexdigest()!=checksum.split(':')[1]:raise ValueError('Source checksum mismatch')
    total=0;types=Counter();regions=Counter();head=[];mri=[];brainlabels=0
    with path.open(encoding='utf-8',newline='') as h:
        for r in csv.DictReader(h):
            total+=1;types[r['image_subtype']]+=1
            if r['image_subtype']!='mri':continue
            mri.append(r);regions[r['radiology_region']]+=1
            labels=(r.get('ml_labels_for_supervised_classification') or '')+' '+(r.get('gt_labels_for_semisupervised_classification') or '')
            if 'brain' in labels.lower():brainlabels+=1
            if r['radiology_region']=='head':head.append(r)
    def count(records):
        return {'image_or_subimage_rows':len(records),'unique_file_ids':len({r['file_id'] for r in records}),
            'source_main_images':len({r['main_image'] for r in records}),
            'cases':len({r['patient_id'] for r in records}),
            'articles':len({r['patient_id'].split('_')[0] for r in records})}
    result={'source':'https://zenodo.org/records/14994046','all_rows':total,'all_mri':count(mri),
        'head_mri':count(head),'mri_regions':dict(regions),'mri_rows_with_brain_label':brainlabels,
        'definition':'Source image_subtype=mri and radiology_region=head. Head is broader than brain; image rows may be subimages. Source main_image IDs can include case identity and are not guaranteed globally deduplicated physical figures.',
        'labels':'Dataset source classifications, not newly independently expert-verified brain labels.',
        'sha256':sha(path),'checksum_verified':checksum}
    out=ROOT/'reports/multicare_mri_counts.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
