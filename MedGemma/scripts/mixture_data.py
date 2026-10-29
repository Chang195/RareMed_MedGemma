"""Independent received-package audit and compatibility with proven MedGemma masks."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from pipeline import LOCAL,conversation,encode,write
from diagnosis_data import MODEL,REVISION,sha,read_jsonl,target
from PIL import Image,features
from transformers import AutoProcessor

PACKAGE=LOCAL/'data/RareMed_CADASIL_SFT_Mixture_Experiments'
OUT=LOCAL/'runs/mixture-experiments/preflight'
ARMS={'A':('dataset_A_20pct',20),'B':('dataset_B_50pct',50),'C':('dataset_C_80pct',80)}

def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def load_messages(row,folder,package,answer=False):
    file=(folder/row['image']).resolve()
    assert file.parent==(package/'images').resolve(), 'Image must stay in package/images'
    # The shared image alias is not a second image. Only messages and one PIL RGB enter the model.
    messages=conversation(row,folder)
    return messages if answer else messages[:1]

def audit(package,out):
    package=package.resolve(); artifacts=json.loads((package/'manifests/build_artifact_hashes.json').read_text(encoding='utf-8'))
    for rel,digest in artifacts.items():
        path=(package/rel).resolve()
        assert path.is_relative_to(package) and sha(path)==digest, 'Artifact hash mismatch: '+rel
    manifest_rows=csv_rows(package/'manifests/sample_manifest.csv')
    manifest={r['sample_id']:r for r in manifest_rows}
    assert len(manifest)==len(manifest_rows)
    ocr={r['sample_id']:r for r in csv_rows(package/'manifests/pixel_answer_leakage.csv')}
    question=json.loads((package/'manifests/build_config.json').read_text())['question']
    image_info={}
    for image in sorted((package/'images').iterdir()):
        if not image.is_file(): continue
        assert re.fullmatch(r'[0-9a-f]{64}\.[a-z]+',image.name)
        digest=sha(image); assert digest==image.stem
        with Image.open(image) as im: im.verify()
        with Image.open(image) as im:
            im.load(); rgb=im.convert('RGB'); size=rgb.size
            pixel=hashlib.sha256(str(size).encode()+rgb.tobytes()).hexdigest()
        image_info[image.name]=dict(sha256=digest,pixel_sha256=pixel,size=size)
    pixel_differences={}
    for row in manifest_rows:
        filename=Path(row['image']).name
        info=image_info[filename]
        assert info['sha256']==row['image_sha256']
        if info['pixel_sha256']!=row['pixel_sha256']:
            pixel_differences[row['sample_id']]=dict(sample_id=row['sample_id'],filename=filename,
                source_pixel_sha256=row['pixel_sha256'],local_pixel_sha256=info['pixel_sha256'],file_sha256_verified=True)
    training={}; shared=defaultdict(set); all_rows={}; summaries={}; selected_ids=set()
    global_owners={field:defaultdict(set) for field in ['sample_id','source_case_id','source_article_id','split_group_id','image_sha256','pixel_sha256','local_decoded_pixel_sha256']}
    for arm,(dirname,pct) in ARMS.items():
        folder=package/dirname; all_rows[arm]={}; training[arm]={}; summaries[arm]={}
        for split in ['train','val','test']:
            file=folder/(split+'.jsonl'); rows=read_jsonl(file); all_rows[arm][split]=rows
            assert len(rows)=={'train':100,'val':67,'test':68}[split]
            assert len({r['sample_id'] for r in rows})==len(rows)
            if split!='train': shared[split].add(sha(file))
            for row in rows:
                raw=manifest[row['sample_id']]
                assert row['image']==row['image_path'] and row['split']==row['metadata']['split']==('validation' if split=='val' else split)
                path=(folder/row['image']).resolve()
                assert path.parent==(package/'images').resolve()
                info=image_info[path.name]
                assert info['sha256']==row['metadata']['image_sha256']==raw['image_sha256']
                assert row['metadata']['pixel_sha256']==raw['pixel_sha256']
                selected_ids.add(row['sample_id'])
                assert [m['role'] for m in row['messages']]==['user','assistant']
                assert row['messages'][0]['content']==[{'type':'image'},{'type':'text','text':question}]
                assert target(row)==row['binary_target'] and row['binary_target'] in ['CADASIL','non-CADASIL']
                assert ocr[row['sample_id']]['diagnosis_term_flag']=='False'
                assert row['metadata']['training_ready_clinically_adjudicated'] is False
                for field in global_owners:
                    value=row.get(field) if field in ['sample_id','source_case_id','source_article_id'] else info['pixel_sha256'] if field=='local_decoded_pixel_sha256' else row['metadata'][field]
                    if value: global_owners[field][value].add(split)
                if row.get('source_article_id')=='PMC12675991': assert split=='train'
                if split=='train':
                    assert raw['original_split'] not in ['val','validation','test','dev']
                    training[arm][row['sample_id']]=row['binary_target']
            positives=[r for r in rows if r['binary_target']=='CADASIL']
            if split=='train': assert len(positives)==pct
            summaries[arm][split]=dict(records=len(rows),CADASIL=len(positives),non_CADASIL=len(rows)-len(positives),
                unique_images=len({r['metadata']['image_sha256'] for r in rows}),
                CADASIL_source_articles=len({r['source_article_id'] for r in positives if r['source_article_id']}),
                CADASIL_provisional_case_keys=len({r['source_case_id'] for r in positives}),
                CADASIL_split_groups=len({r['metadata']['split_group_id'] for r in positives}),
                source_counts=dict(Counter(r['source_dataset'] for r in rows)))
    assert all(len(x)==1 for x in shared.values()), 'Heldout JSONLs differ across arms'
    overlaps={field:[key for key,splits in values.items() if len(splits)>1] for field,values in global_owners.items()}
    assert not any(overlaps.values()), 'Cross-split overlaps detected'
    for label,order in [('CADASIL',['A','B','C']),('non-CADASIL',['C','B','A'])]:
        sets=[{sid for sid,value in training[a].items() if value==label} for a in order]
        assert sets[0]<=sets[1]<=sets[2], 'Training subsets are not nested'
    old_heldouts={}
    for oldsplit,newsplit in [('validation','val'),('test','test')]:
        old=read_jsonl(LOCAL/'data/diagnosis-sft-v0.1'/(oldsplit+'.jsonl'))
        oldhashes={r['metadata']['image_sha256'] for r in old}
        newhashes={r['metadata']['image_sha256'] for r in all_rows['A'][newsplit]}
        assert oldhashes<=newhashes, 'Old heldout images not retained'
        assert not oldhashes&{r['metadata']['image_sha256'] for arm in ARMS for r in all_rows[arm]['train']}
        old_heldouts[oldsplit]=dict(original_records=len(old),retained_image_hashes=len(oldhashes))
    write(out/'pixel-hash-differences.json',list(pixel_differences.values()))
    result=dict(status='PASS_STRUCTURAL_CHECKS_WITH_PIXEL_HASH_WARNING' if pixel_differences else 'PASS_LOCAL_TECHNICAL_AUDIT',package=str(package),artifact_hashes_verified=len(artifacts),
        exported_images_verified=len(image_info),arms=summaries,shared_heldout_hashes={k:list(v)[0] for k,v in shared.items()},
        old_heldout_retention=old_heldouts,unique_used_records=len(selected_ids),
        pixel_hash_differences=dict(all_exported=len(pixel_differences),selected_used=len(set(pixel_differences)&selected_ids),
            formats=dict(Counter(Path(x['filename']).suffix for x in pixel_differences.values())),
            local_Pillow=Image.__version__,jpeg_library=features.version('jpg'),
            explanation='Raw bytes verified; upstream/local decoded RGB hash differs. Decoder variation is a hypothesis, not confirmed. Originals not modified.'),
        global_cross_split_overlaps=overlaps,nested_subsets=True,question=question,
        source_file_sha256={str((package/dirname/(split+'.jsonl')).relative_to(package)):sha(package/dirname/(split+'.jsonl'))
            for dirname,_ in ARMS.values() for split in ['train','val','test']},
        limitations=['Clinical labels, OCR sensitivity and legal rights not independently adjudicated here.',
            'Source is confounded with training class: RareMed positives, MedPix negatives.',
            'Previously evaluated heldouts are not untouched confirmation cohorts. Do not reuse old free-diagnosis baseline under this new binary prompt.'])
    write(out/'audit.json',result)
    return all_rows,result

def processor_check(package,out,all_rows):
    processor=AutoProcessor.from_pretrained(LOCAL/'runs/diagnosis-night-01/processor',use_fast=False,local_files_only=True)
    summary=[]
    for arm,(dirname,_) in ARMS.items():
        for label in ['CADASIL','non-CADASIL']:
            row=next(r for r in all_rows[arm]['train'] if r['binary_target']==label)
            messages=load_messages(row,package/dirname,package,True)
            batch,_,n=encode(processor,messages)
            ids=batch['input_ids'][0]; labels=batch['labels'][0]
            text=processor.tokenizer.decode(ids[labels!=-100],skip_special_tokens=False)
            assert text==label
            assert len(load_messages(row,package/dirname,package,False))==1
            image_id=processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')
            entry=dict(arm=arm,sample_id=row['sample_id'],target=label,supervised_tokens=int((labels!=-100).sum()),
                prefix_tokens=n,image_tokens=int((ids==image_id).sum()),input_tokens=len(ids),
                pixel_values_shape=list(batch['pixel_values'].shape),no_history_or_metadata_in_inputs=True)
            summary.append(entry)
            write(out/'masks'/f'{arm}-{label}.json',dict(**entry,input_ids=ids.tolist(),labels=labels.tolist()))
    write(out/'processor-check.json',dict(status='PASS',model=MODEL,revision=REVISION,samples=summary,
        image_processing=processor.image_processor.to_dict(),template=processor.chat_template,optimizer_steps=0))
    print('Actual processor and assistant-only masks PASS:',json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=PACKAGE); p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--processor',action='store_true'); a=p.parse_args()
    rows,report=audit(a.data,a.out)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if a.processor: processor_check(a.data,a.out,rows)

if __name__=='__main__': main()
