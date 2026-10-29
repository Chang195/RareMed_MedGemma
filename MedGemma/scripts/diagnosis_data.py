"""Audit diagnosis package and reuse the proven pilot processor/mask implementation.

Never generates predictions or updates weights. References are not model inputs.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
from collections import defaultdict, Counter
from pipeline import LOCAL, conversation, encode, write
import torch
from PIL import Image
from transformers import AutoProcessor

MODEL='google/medgemma-1.5-4b-it'
REVISION='91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def target(row):
    content=row['messages'][1]['content']
    if isinstance(content,str): return content
    assert all(x['type']=='text' for x in content)
    return ''.join(x['text'] for x in content)

def model_messages(row,root,include_answer=False):
    # Proven PIL RGB loader; metadata/provenance never enter this return value.
    messages=conversation(row,root)
    return messages if include_answer else messages[:1]

def audit(root,out):
    root=root.resolve()
    for name in ['images','train.jsonl','validation.jsonl','test.jsonl','manifest.json','audit.json','provenance.jsonl','held.json','README.md']:
        assert (root/name).exists(), f'Missing {name}'
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    provenance=read_jsonl(root/'provenance.jsonl')
    provenance_ids=[x['sample_id'] for x in provenance]
    assert len(provenance_ids)==len(set(provenance_ids)), 'Duplicate provenance IDs'
    held=json.loads((root/'held.json').read_text(encoding='utf-8-sig'))
    assert len(held)==manifest['held']==6
    splits={}; summaries={}; images=[]; seen_ids=set()
    owner={key:defaultdict(set) for key in ['article_id','split_group_id','image_sha256','stored_pixel_sha256','decoded_rgb_sha256']}
    for split in ['train','validation','test']:
        path=root/(split+'.jsonl')
        assert sha(path)==manifest['files_sha256'][path.name], f'{split} JSONL hash mismatch'
        rows=read_jsonl(path); splits[split]=rows
        assert len(rows)==manifest['counts'][split]
        for row in rows:
            sid=row['sample_id']; meta=row['metadata']
            assert sid not in seen_ids, f'Duplicate sample ID {sid}'
            seen_ids.add(sid)
            assert sid in provenance_ids, f'Missing provenance {sid}'
            assert meta['split']==split
            assert meta['training_ready'] is False
            assert [m['role'] for m in row['messages']]==['user','assistant']
            user=row['messages'][0]['content']
            assert sum(c.get('type')=='image' for c in user)==1
            assert all(c.get('type') in ['image','text'] for c in user)
            assert all(set(c)=={'type'} for c in user if c['type']=='image'), 'Expected image placeholder only'
            question=''.join(c['text'] for c in user if c['type']=='text')
            assert question==manifest['question'], 'Unexpected prompt variation'
            label=target(row)
            assert label.strip() and label==label.strip()
            if meta['diagnosis_family']=='CADASIL_author_reported': assert label=='CADASIL'
            else: assert label==meta['source_diagnosis_name'], f'Changed source label {sid}'
            rel=Path(row['image']); path=(root/rel).resolve()
            assert not rel.is_absolute() and '..' not in rel.parts and path.is_relative_to(root)
            digest=sha(path)
            assert digest==meta['image_sha256'], f'Image hash mismatch {sid}'
            with Image.open(path) as im: im.verify()
            with Image.open(path) as im:
                im.load(); size=im.size; mode=im.mode; rgb=im.convert('RGB')
                pixel=hashlib.sha256(rgb.tobytes()).hexdigest()
            for key in ['article_id','split_group_id','image_sha256']: owner[key][meta[key]].add(split)
            owner['stored_pixel_sha256'][meta['pixel_sha256']].add(split)
            owner['decoded_rgb_sha256'][pixel].add(split)
            images.append(dict(sample_id=sid,split=split,image=row['image'],sha256=digest,
                               decoded_rgb_sha256=pixel,size=size,source_mode=mode))
        cases=[x for x in rows if x['metadata']['diagnosis_family']=='CADASIL_author_reported']
        counts=dict(Counter(x['metadata']['diagnosis_family'] for x in rows))
        assert counts==manifest['diagnosis_families'][split]
        summaries[split]=dict(images=len(rows),articles=len({r['metadata']['article_id'] for r in rows}),
            split_groups=len({r['metadata']['split_group_id'] for r in rows}),CADASIL_images=len(cases),
            CADASIL_articles=len({r['metadata']['article_id'] for r in cases}),
            CADASIL_split_groups=len({r['metadata']['split_group_id'] for r in cases}),diagnosis_families=counts)
    overlaps={key:[v for v,sets in values.items() if len(sets)>1] for key,values in owner.items()}
    assert not any(overlaps.values()), f'Cross-split overlap: {overlaps}'
    assert set(provenance_ids)==seen_ids, 'Unexpected unmatched provenance IDs'
    smoke={s:[r['sample_id'] for r in rows if r['metadata']['article_id']=='PMC12675991'] for s,rows in splits.items()}
    assert smoke['train'] and not smoke['validation'] and not smoke['test']
    held_ids={r.get('sample_id') for r in held}
    assert not (held_ids&seen_ids), 'Held record allocated'
    result=dict(status='PASS',dataset=str(root),splits=summaries,held=len(held),provenance_records=len(provenance),
        smoke_case_locations=smoke,cross_split_overlaps=overlaps,images=images,
        package_file_sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()},
        stored_pixel_hash_note='Stored pixel hashes checked for cross-split overlap only: upstream hashing recipe not supplied. Image file SHA256 fully verified. Independently decoded RGB bytes hashed for every image.',
        limits='Groups/articles are not verified patient counts; author diagnosis labels are not clinical gold standards; no OCR/PHI or rights adjudication performed')
    write(out/'data-audit.json',result)
    return splits,result

def processor_check(rows,root,out):
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,use_fast=False,local_files_only=True)
    selected=[]
    for positive in [True,False]:
        selected.extend([r for r in rows if (r['metadata']['diagnosis_family']=='CADASIL_author_reported')==positive][:2])
    longest=max(rows,key=lambda r:len(target(r)))
    if longest['sample_id'] not in {r['sample_id'] for r in selected}: selected.append(longest)
    summaries=[]
    for row in selected:
        messages=model_messages(row,root,include_answer=True)
        batch,prompt,n=encode(processor,messages)
        ids=batch['input_ids'][0]; labels=batch['labels'][0]
        supervised=processor.tokenizer.decode(ids[labels!=-100],skip_special_tokens=False)
        assert supervised==target(row), 'Supervised text differs from reference'
        # Generation messages contain exactly one user turn, never the reference.
        inference=model_messages(row,root)
        assert len(inference)==1 and inference[0]['role']=='user'
        image_id=processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')
        summary=dict(sample_id=row['sample_id'],input_tokens=len(ids),user_prefix_tokens=n,
            image_tokens=int((ids==image_id).sum()),supervised_tokens=int((labels!=-100).sum()),
            supervised_text=supervised,pixel_values_shape=list(batch['pixel_values'].shape),
            generation_roles=[m['role'] for m in inference],metadata_in_model_inputs=False)
        summaries.append(summary)
        write(out/'first-batch'/f"{row['sample_id']}-mask.json",dict(**summary,
            input_ids=ids.tolist(),labels=labels.tolist(),attention_mask=batch['attention_mask'][0].tolist()))
    # Real multimodal batch with unequal answer lengths: preserve image handling.
    batch_messages=[model_messages(r,root,True) for r in selected[:2]]
    # Include the longest target to guarantee unequal sequence lengths if needed.
    batch_messages[1]=model_messages(longest,root,True)
    full=processor.apply_chat_template(batch_messages,tokenize=True,add_generation_prompt=False,
                                      return_dict=True,return_tensors='pt',padding=True,truncation=False)
    assert (full['attention_mask']==0).any(), 'Padding test needs unequal lengths'
    combined=torch.full_like(full['input_ids'],-100)
    for i,m in enumerate(batch_messages):
        single,_,_=encode(processor,m)
        positions=torch.nonzero(full['attention_mask'][i],as_tuple=True)[0]
        assert torch.equal(full['input_ids'][i,positions],single['input_ids'][0])
        combined[i,positions]=single['labels'][0]
    assert (combined[full['attention_mask']==0]==-100).all()
    write(out/'processor-check.json',dict(status='PASS',model=MODEL,revision=REVISION,train_samples=summaries,
        template=processor.chat_template,image_processing=processor.image_processor.to_dict(),
        multimodal_padding_mask_checked=True,padding_tokens=int((full['attention_mask']==0).sum()),
        tokenizer_padding_side=processor.tokenizer.padding_side,training=False))
    return summaries

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=LOCAL/'data/diagnosis-sft-v0.1')
    p.add_argument('--out',type=Path,default=LOCAL/'runs/diagnosis-v0.1/preflight')
    p.add_argument('--processor',action='store_true')
    a=p.parse_args(); splits,report=audit(a.data,a.out)
    print(json.dumps(dict(status=report['status'],splits=report['splits'],held=report['held']),indent=2),flush=True)
    if a.processor:
        summaries=processor_check(splits['train'],a.data,a.out)
        print(json.dumps(dict(processor='PASS',train_samples=summaries),indent=2),flush=True)
    write(a.out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        free_total_vram=list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None,
        packages={d.metadata['Name']:d.version for d in md.distributions()},model=MODEL,revision=REVISION,
        smoke_adapter_loaded=False,weight_updates_performed=False))

if __name__=='__main__': main()
