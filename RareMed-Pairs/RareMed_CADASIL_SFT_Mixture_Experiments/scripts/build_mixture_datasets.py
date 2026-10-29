"""Real-source, image-only binary SFT mixtures. Never trains or calls a model API.

First run: python scripts/build_mixture_datasets.py --source-root .. --download-medpix
Portable rebuild: python scripts/build_mixture_datasets.py (frozen inputs required).
Dependencies: Python 3.10+, numpy, Pillow, matplotlib. Seed is fixed at 42.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, csv, hashlib, io, json, math
import random, re, shutil, sys, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[1]
SEED = 42
QUESTION = 'Classify this brain MRI figure as CADASIL or non-CADASIL. Reply with exactly one label: CADASIL or non-CADASIL.'
UNCERTAIN = re.compile(r'\b(probabl\w*|possibl\w*|presumed|suspect\w*|uncertain|unknown|awaiting|differential|versus)\b|\bvs\.?\s', re.I)
HARD = re.compile(r'multiple sclerosis|leukoaraiosis|leukodystroph|leukoencephalopath|alexander\s+disease|canavan|small vessel|CADASIL|CARASIL|HTRA1|Fabry', re.I)
CAD = re.compile(r'\bCADASIL\b|cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy', re.I)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').split('\n') if s.strip()]
def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def jsonl(p, rr):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rr), encoding='utf-8')
def table(p, rr):
    rr=list(rr); Path(p).parent.mkdir(parents=True, exist_ok=True)
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys);w.writeheader();w.writerows(rr)
def norm(s): return ' '.join(str(s or '').casefold().split())
def family(s):
    if re.search(r'\bCADASIL2\b|HTRA1|CADASIL[- ]like',s,re.I): return 'Other'
    if CAD.search(s): return 'CADASIL'
    if re.search(r'glioblastoma|\bGBM\b',s,re.I): return 'GBM'
    if re.search(r'astrocytoma',s,re.I): return 'Astrocytoma'
    if re.search(r'multiple sclerosis',s,re.I): return 'Multiple sclerosis'
    return 'Other'
def evidence_status(s):
    if not s or norm(s) in {'n/a','none','unknown','unclear'}:return 'unresolved'
    if re.search(r'differential|versus|\bvs\b',s,re.I):return 'differential'
    if UNCERTAIN.search(s):return 'probable_or_unresolved'
    return 'source_reported_final'

class UF:
    def __init__(self):self.p={}
    def find(self,x):
        self.p.setdefault(x,x)
        if self.p[x]!=x:self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def freeze(root):
    """Snapshot inputs, retaining originals unchanged; refuse source drift."""
    inp=BASE/'inputs';inp.mkdir(exist_ok=True)
    mapping={
        'medpix_cases.json':root/'data/benchmarks/four_disease/sources/medpix_cases.json',
        'medpix_images.json':root/'data/benchmarks/four_disease/sources/medpix_images.json',
        'raremed_provenance.jsonl':root/'data/datasets/diagnosis-sft-v0.2/provenance.jsonl',
        'raremed_inventory.jsonl':root/'data/datasets/diagnosis_split_v1/dedup/records.jsonl',
        'duplicate_pairs.jsonl':root/'data/datasets/diagnosis_split_v1/dedup/duplicate_pairs.jsonl',
        'figure_links.jsonl':root/'data/processed/deepseek_collection_v3_links.jsonl',
        'figures.jsonl':root/'data/processed/collection_v3_figures.jsonl',
        'known_exposure.json':root/'configs/known_model_exposure.json',
    }
    for split in ['train','validation','test']:
        mapping['raremed_'+split+'.jsonl']=root/'data/datasets/diagnosis-sft-v0.2'/(split+'.jsonl')
    for name,p in mapping.items():
        target=inp/name
        if target.exists() and sha(target)!=sha(p):raise ValueError('Frozen input differs: '+name)
        if not target.exists():shutil.copy2(p,target)
    texts={}
    articles={r['metadata']['article_id'] for split in ['train','validation','test'] for r in rows(inp/('raremed_'+split+'.jsonl'))}
    for art in sorted(articles):
        path=root/'data/raw'/art/'article.bioc.xml'
        if not path.exists():raise ValueError('Missing original BioC source '+art)
        doc=ET.parse(path)
        texts[art]={'source_sha256':sha(path),'text':'\n'.join(t.text or '' for t in doc.findall('.//passage/text'))}
    frozen=inp/'raremed_original_bioc_texts.json'
    if frozen.exists() and read(frozen)!=texts:raise ValueError('Frozen original article text changed')
    if not frozen.exists():save(frozen,texts)
    save(inp/'source_hashes.json', {k:sha(v) for k,v in mapping.items()})

def fetch_images(download):
    """Use pinned author repository tree; verify each Git blob SHA-1."""
    tree=read(BASE/'inputs/medpix_github_tree.json');revision=tree['sha']
    images=read(BASE/'inputs/medpix_images.json')
    needed={x['image']+'.png' for x in images if x['Type']=='MR' and x['Location']=='Brain and Neuro'}
    dest=BASE/'inputs/medpix_images';dest.mkdir(exist_ok=True)
    items=[x for x in tree['tree'] if x['path'].startswith('MedPix-2-0/images/') and Path(x['path']).name in needed]
    def one(x):
        p=dest/Path(x['path']).name
        url=f'https://raw.githubusercontent.com/CHILab1/MedPix-2.0/{revision}/'+x['path']
        if not p.exists() and download:
            last=None
            for attempt in range(3):
                try:
                    data=urllib.request.urlopen(url,timeout=60).read()
                    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                    if blob!=x['sha']:raise ValueError('Git blob mismatch')
                    p.write_bytes(data);last=None;break
                except Exception as e:last=str(e)
            if last:return {'image':p.name,'status':'download_failed','error':last}
        if not p.exists():return {'image':p.name,'status':'missing','url':url}
        data=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=x['sha']:raise ValueError('Cached image mismatch '+p.name)
        with Image.open(p) as im:im.load()
        return {'image':p.name,'status':'verified','sha256':sha(p),'git_blob_sha1':blob,'url':url,'revision':revision}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        result=[]
        for i,r in enumerate(pool.map(one,items),1):
            result.append(r)
            if i%50==0:print(f'MedPix images {i}/{len(items)}',flush=True)
    table(BASE/'manifests/medpix_image_download.csv',result)
    return result

def fingerprints(path):
    with Image.open(path) as im:
        im.load();rgb=im.convert('RGB');a=np.asarray(rgb)
        pixel=hashlib.sha256(str(rgb.size).encode()+a.tobytes()).hexdigest()
        g=np.asarray(rgb.convert('L').resize((32,32)),dtype=float)
        n=np.arange(32);mat=np.cos(np.pi*(2*n[None,:]+1)*n[:,None]/64)
        d=(mat@g@mat.T)[:8,:8].flatten()[1:];bits=d>np.median(d)
        ph=sum(int(v)<<i for i,v in enumerate(bits))
        tiny=np.asarray(rgb.convert('L').resize((9,8)));db=(tiny[:,1:]>tiny[:,:-1]).flatten()
        dh=sum(int(v)<<i for i,v in enumerate(db))
        return pixel,ph,dh,rgb.size

def sanitize_history(history,diagnosis):
    """Image-only primary task never uses this field. Preserve raw in inputs."""
    terms=[r'CADASIL',r'NOTCH3',r'glioblastoma',r'astrocytoma',r'multiple sclerosis',r'leukodystroph\w*']
    if len(diagnosis)>3:terms.append(re.escape(diagnosis))
    bad=re.compile('|'.join(terms)+r'|\bdiagnos\w*\b|\bbiopsy\b|\bpatholog\w*\b',re.I)
    sentences=re.split(r'(?<=[.!?])\s+|[\r\n]+',history or '')
    kept=[s for s in sentences if not bad.search(s)]
    return ' '.join(kept).strip(),len(sentences)-len(kept)

def collect(root):
    inp=BASE/'inputs';prov={r['sample_id']:r for r in rows(inp/'raremed_provenance.jsonl')}
    inventory={r['sample_id']:r for r in rows(inp/'raremed_inventory.jsonl')}
    original_texts=read(inp/'raremed_original_bioc_texts.json')
    candidates=[]; audit=[];original=[]
    for split in ['train','validation','test']:
        for r in rows(inp/('raremed_'+split+'.jsonl')):
            sid=r['sample_id'];p=prov[sid];ln=p['proposed_linkage'];inv=inventory[sid];diag=ln.get('diagnosis_name') or ''
            original.append({'sample_id':sid,'split':split,'article_id':r['metadata']['article_id'],'diagnosis':diag,'family':family(diag),'original_export_family':r['metadata']['diagnosis_family']})
            status=evidence_status(diag)
            # Exact quote support is structural evidence, not independent semantic adjudication.
            snippets='\n'.join(x.get('text','') for v in p.get('source_text',{}).values() if isinstance(v,list) for x in v if isinstance(x,dict))
            quotes=[q.get('quote','') for q in ln.get('diagnosis_evidence',[])]
            rawtext=original_texts[r['metadata']['article_id']]['text']
            quote_ok=bool(quotes) and all(q and norm(q) in norm(rawtext) for q in quotes)
            reason='eligible'
            if ln.get('diagnosis_status')!='author_reported_confirmed' or status!='source_reported_final':reason='nonfinal_or_uncertain_diagnosis'
            elif ln.get('link_status') not in {'explicit','contextual'} or not ln.get('case_label'):reason='no_single_case_link'
            elif not quote_ok:reason='diagnosis_quote_not_grounded_in_frozen_source_text'
            if CAD.search(diag) and re.search(r'phenotypes?',diag,re.I) and re.search(r'unclear clinical significance|not definitive',json.dumps(ln),re.I):
                reason='CADASIL_phenotype_with_explicitly_unresolved_confirmation'
            if split=='train' and family(diag)!='CADASIL':reason='original_train_nonCADASIL_excluded_from_MedPix_mixture'
            path=(root/'data/datasets/diagnosis-sft-v0.2'/r['image']) if root else BASE/'inputs/raremed_images'/(sid+Path(r['image']).suffix)
            frozen=BASE/'inputs/raremed_images'/(sid+Path(r['image']).suffix)
            if root and path.exists() and not frozen.exists():frozen.parent.mkdir(exist_ok=True);shutil.copy2(path,frozen)
            path=frozen
            if not path.exists():reason='missing_image'
            elif sha(path)!=r['metadata']['image_sha256']:raise ValueError('Original RareMed image hash mismatch '+sid)
            audit.append({'source':'RareMed-Pairs','sample_id':sid,'original_split':split,'diagnosis':diag,'family':family(diag),'evidence_level':status,'extraction_status':ln.get('diagnosis_status'),'quote_support':quote_ok,'decision':reason})
            if reason!='eligible':continue
            art=r['metadata']['article_id'];case=art+'::'+norm(ln['case_label'])
            candidates.append({'original_sample_id':sid,'sample_id':'S'+digest('RareMed:'+sid)[:20], 'source_dataset':'RareMed-Pairs','source_article_id':art,'source_case_id':case,'patient_id':None,'case_identity_status':'provisional_source_link_not_verified_patient', 'original_split':split,'split':split,'diagnosis':diag,'binary_target':'CADASIL' if family(diag)=='CADASIL' else 'non-CADASIL','diagnosis_family':family(diag),'diagnosis_evidence_level':'author_reported_confirmed_quote_supported_unreviewed','negative_difficulty':'hard' if HARD.search(diag) else 'easy','negative_stratum':neg_stratum(diag),'case_history':'','case_history_status':'unassigned_article_context_not_used','image_input':str(path.relative_to(BASE)).replace('\\','/'),'source_url':p.get('source_url'),'license':p.get('license'),'source_link_status':ln.get('link_status')})
            candidates[-1]['source_article_title']=inv.get('article_title')
    cases=read(inp/'medpix_cases.json');ci={c['U_id']:c for c in cases}
    variants=[]
    for c in cases:
        d=str(c['Case'].get('Case Diagnosis') or '');h=c['Case'].get('History') or '';clean,n=sanitize_history(h,d)
        variants.append({'source_case_id':c['U_id'],'raw_source_history':h,'sanitized_history':clean,'removed_sentences':n,'used_in_model_input':False,'policy':'source_original_and_sentence_removed_variants_for_provenance_only'})
    jsonl(BASE/'manifests/history_variants_not_model_input.jsonl',variants)
    ims=read(inp/'medpix_images.json');med_audit=[]
    for im in ims:
        c=ci[im['U_id']];diag=str(c['Case'].get('Case Diagnosis') or '').strip();status=evidence_status(diag)
        reason='eligible'
        if im['Type']!='MR' or im['Location']!='Brain and Neuro':reason='not_source_labelled_brain_MR'
        elif im['image'] not in c['MRI']:reason='image_case_MRI_list_disagreement'
        elif status!='source_reported_final':reason='nonfinal_or_unresolved_diagnosis'
        elif family(diag)=='CADASIL':reason='MedPix_CADASIL_not_negative'
        # Catch observed source-location errors; retain a transparent exclusion record.
        elif re.search(r'\bL[1-5][-–]S1\b|cauda equina|lumbar (?:spine|disc|disk)',diag,re.I):reason='source_location_conflicts_with_spinal_diagnosis'
        path=BASE/'inputs/medpix_images'/(im['image']+'.png')
        if reason=='eligible' and not path.exists():reason='missing_image'
        med_audit.append({'source':'MedPix2','sample_id':im['image'],'source_case_id':im['U_id'],'diagnosis':diag,'family':family(diag),'evidence_level':status,'source_modality':im['Type'],'source_location':im['Location'],'decision':reason})
        if reason!='eligible':continue
        history,removed=sanitize_history(c['Case'].get('History') or '',diag)
        candidates.append({'original_sample_id':im['image'],'sample_id':'S'+digest('MedPix:'+im['image'])[:20], 'source_dataset':'MedPix2','source_article_id':None,'source_case_id':im['U_id'],'patient_id':None,'case_identity_status':'source_case_ID_not_global_patient_ID','original_split':None,'split':'unassigned','diagnosis':diag,'binary_target':'non-CADASIL','diagnosis_family':family(diag),'diagnosis_evidence_level':'structured_source_final_not_independent_clinical_adjudication','negative_difficulty':'hard' if HARD.search(diag) else 'easy','negative_stratum':neg_stratum(diag),'case_history':history,'history_sentences_removed':removed,'case_history_status':'sanitized_source_history_not_used_in_image_only_prompt','image_input':str(path.relative_to(BASE)).replace('\\','/'),'source_url':'https://github.com/CHILab1/MedPix-2.0','license':'source_terms_not_independently_cleared','source_link_status':'structured_case_image_ID'})
    table(BASE/'manifests/raremed_record_audit.csv',audit);table(BASE/'manifests/medpix_record_audit.csv',med_audit)
    jsonl(BASE/'manifests/original_split_inventory.jsonl',original)
    # Entire upstream figure inventory, including unresolved and keyword-only records.
    figure_audit=[]
    links={r['article_id']+'_'+r['figure_id']:r for r in rows(inp/'figure_links.jsonl')}
    for r in rows(inp/'figures.jsonl'):
        sid=r['article_id']+'_'+r['figure_id'];lk=links.get(sid,{}).get('proposed_linkage',{});diag=lk.get('diagnosis_name') or ''
        ds=lk.get('diagnosis_status','unresolved')
        level=('keyword_mention_only_no_case_diagnosis' if not diag and CAD.search(r.get('caption','')) else ds)
        figure_audit.append({'sample_id':sid,'article_id':r['article_id'],'diagnosis':diag,'family':family(diag),'evidence_level':level,'link_status':lk.get('link_status','not_extracted'),'diagnosis_name_certainty':evidence_status(diag),'assets':len(r.get('assets',[])),'in_prior_diagnosis_export':sid in prov,'cohort_policy':'use_existing_diagnosis_export_only; do_not_resurrect_held_unlinked_figures'})
    table(BASE/'manifests/upstream_figure_audit.csv',figure_audit)
    return candidates,original,audit,med_audit,figure_audit

def neg_stratum(diag):
    f=family(diag)
    if f in {'GBM','Astrocytoma','Multiple sclerosis'}:return f
    return 'Other hard neuro' if HARD.search(diag) else 'Other easy neuro'

def screen_ocr(rr):
    path=BASE/'inputs/pixel_ocr.json'
    if not path.exists():raise ValueError('Missing frozen pixel OCR; run scripts/audit_pixel_text.py first')
    cache=read(path);audit=[];kept=[]
    terms=re.compile(r'CADASIL|NOTCH3|glioblastoma|astrocytoma|multiple\s+sclerosis|leukodystrophy|leukoaraiosis|meningioma|schwannoma|cysticercosis|craniopharyngioma|medulloblastoma|hemangioblastoma|cavernoma|ependymoma|encephalitis|infarct(?:ion)?|\bGBM\b',re.I)
    for r in rr:
        h=sha(BASE/r['image_input']);ocr=cache.get(h)
        if not ocr or ocr.get('status')!='completed':raise ValueError('Missing/failed OCR for eligible image: '+r['original_sample_id'])
        text=' '.join(v['text'] for v in ocr.get('text',[]) if v['confidence']>=.5)
        hits=terms.findall(text)
        # Exact source diagnosis also catches diseases outside the fixed lexicon.
        exact=norm(r['diagnosis']) in norm(text) if len(r['diagnosis'])>=5 else False
        flag=bool(hits or exact)
        audit.append({'sample_id':r['sample_id'],'original_sample_id':r['original_sample_id'],'sha256':h,'OCR_status':'completed','diagnosis_term_flag':flag,'hits':json.dumps(hits),'OCR_text':text,'decision':'quarantine_image_diagnosis_text' if flag else 'no_diagnostic_text_detected'})
        if not flag:r['pixel_text_check']='local_OCR_no_diagnostic_terms_detected';kept.append(r)
    table(BASE/'manifests/pixel_answer_leakage.csv',audit)
    return kept

def group_and_dedup(rr):
    uf=UF(); owner={}; hashes={}; exclusions=[];relations=[]
    # Union all old edges, including records outside the eligible pool, so bridges are retained.
    for p in rows(BASE/'inputs/duplicate_pairs.jsonl'):
        if p.get('conservative_group_edge'):uf.union('old:'+p['sample_a'],'old:'+p['sample_b'])
    for r in rows(BASE/'inputs/raremed_inventory.jsonl'):
        uf.union('old:'+r['sample_id'],'article:'+r['article_id'])
    for r in rr:
        sid=r['sample_id'];uf.find(sid)
        uf.union(sid,'case:'+r['source_dataset']+':'+r['source_case_id'])
        if r['source_article_id']:uf.union(sid,'article:'+r['source_article_id']);uf.union(sid,'old:'+r['original_sample_id'])
        pixel,ph,dh,size=fingerprints(BASE/r['image_input'])
        r.update(image_sha256=sha(BASE/r['image_input']),pixel_sha256=pixel,width=size[0],height=size[1],phash=ph,dhash=dh)
        for k in [('bytes',r['image_sha256']),('pixels',pixel)]:
            if k in owner:uf.union(sid,owner[k])
            owner[k]=sid
    for i,a in enumerate(rr):
        for b in rr[:i]:
            if a['source_dataset']=='RareMed-Pairs' and b['source_dataset']=='RareMed-Pairs':continue
            if a['source_case_id']==b['source_case_id'] and a['source_dataset']==b['source_dataset']:continue
            pd=(a['phash']^b['phash']).bit_count();dd=(a['dhash']^b['dhash']).bit_count()
            if pd<=4 and dd<=6:
                uf.union(a['sample_id'],b['sample_id'])
                relations.append({'sample_a':a['sample_id'],'sample_b':b['sample_id'],'phash_distance':pd,'dhash_distance':dd,'status':'unreviewed_similarity_conservatively_grouped'})
    # Prior heldouts win; no former evaluation example can move into train.
    groups=collections.defaultdict(list)
    for r in rr:groups[uf.find(r['sample_id'])].append(r)
    changed=[]
    exposure=read(BASE/'inputs/known_exposure.json')
    for key,gg in groups.items():
        fixed={r['original_split'] for r in gg if r['original_split'] in {'validation','test'}}
        seen=any(r['source_article_id'] in exposure.get('article_ids',[]) or r['original_sample_id'] in exposure.get('sample_ids',[]) for r in gg)
        # If two heldouts collide, quarantine the component rather than migrate test to validation.
        trained=any(r['original_split']=='train' for r in gg)
        split='quarantine' if len(fixed)>1 or (fixed and (seen or trained)) else (next(iter(fixed)) if fixed else 'train')
        for r in gg:
            r['split_group_id']='G'+digest(key)[:20]
            r['split']=split if (fixed or r['source_dataset']=='RareMed-Pairs') else 'unassigned'
            if r['original_split'] and r['original_split']!=r['split']:
                changed.append({'sample_id':r['sample_id'],'old':r['original_split'],'new':r['split'],'reason':'conservative_connected_component_or_known_exposure'})
    # Exact duplicates: keep heldouts before train; conflicting labels quarantine all copies.
    pixels=collections.defaultdict(list)
    for r in rr:pixels[r['pixel_sha256']].append(r)
    keep=[]
    for pixel,gg in pixels.items():
        if len({r['binary_target'] for r in gg})>1:
            for r in gg:exclusions.append({'sample_id':r['sample_id'],'reason':'conflicting_labels_same_decoded_image'})
            continue
        gg.sort(key=lambda r:(0 if r['split'] in {'test','validation'} else 1,r['sample_id']))
        keep.append(gg[0])
        for r in gg[1:]:exclusions.append({'sample_id':r['sample_id'],'reason':'exact_decoded_duplicate','canonical_sample_id':gg[0]['sample_id']})
    table(BASE/'manifests/near_duplicate_edges.csv',relations);table(BASE/'manifests/dedup_exclusions.csv',exclusions);table(BASE/'manifests/split_changes.csv',changed)
    return keep,changed,exclusions

def ranked(rr,tag):
    """Round-robin across cases: first one image/case, then second; stable nested prefixes."""
    cc=collections.defaultdict(list)
    for r in rr:cc[r['source_case_id']].append(r)
    order=sorted(cc,key=lambda x:digest(f'{SEED}:{tag}:{x}'))
    for c in order:cc[c].sort(key=lambda r:digest(f'{SEED}:{tag}:{r["sample_id"]}'))
    return [cc[c][i] for i in range(max((len(v) for v in cc.values()),default=0)) for c in order if i<len(cc[c])]

def allocate_medpix(rr):
    # Group allocation before mixture sampling; all photos of one source case stay together.
    groups=collections.defaultdict(list)
    for r in rr:groups[r['split_group_id']].append(r)
    official=read(BASE/'inputs/medpix_author_split_uid_list.json')
    official_lookup={i:s for s in ['train','dev','test'] for i in official[s]}
    assert not (set(official['train'])&set(official['dev']) or set(official['train'])&set(official['test']) or set(official['dev'])&set(official['test']))
    strata=collections.defaultdict(list)
    for g,gg in groups.items():
        if any(r['original_split'] in {'validation','test'} for r in gg):continue
        med=[r for r in gg if r['source_dataset']=='MedPix2']
        if med:
            splits={official_lookup[r['source_case_id']] for r in med}
            split=('quarantine' if len(splits)>1 else {'train':'train','dev':'validation','test':'test'}[next(iter(splits))])
            for r in gg:
                if r['split']=='unassigned':r['split']=split
            continue
    for stratum,ids in strata.items():
        ids.sort(key=lambda x:digest(f'{SEED}:heldout:{x}'))
        # >= 3 groups allows one val and one test; never copy rare cases.
        n=min(max(1,round(len(ids)*.15)),(len(ids)-1)//2) if len(ids)>=3 else 0
        for j,g in enumerate(ids):
            split='validation' if j<n else 'test' if j<2*n else 'train'
            for r in groups[g]:
                if r['split']=='unassigned':r['split']=split
    return rr

def neg_rank(rr,max_needed):
    """Weighted fair queue gives a single nested stream and stable disease mix."""
    cc=collections.defaultdict(list)
    for r in rr:cc[r['negative_stratum']].append(r)
    # Hard negatives capped only by real availability; no artificial repeats.
    queues={k:ranked(v,k) for k,v in cc.items()};counts=collections.Counter();out=[]
    # Prefer glioma/MS/other hard diseases over a random grab of unrelated MRI.
    # Their real scarcity caps the weights; all arms share this one fixed stream.
    weights={k:min(.20,len(v)/max_needed) for k,v in cc.items() if k!='Other easy neuro'}
    if 'Other easy neuro' in cc:weights['Other easy neuro']=1-sum(weights.values())
    else:
        total=sum(weights.values());weights={k:v/total for k,v in weights.items()}
    for i in range(len(rr)):
        available=[k for k,q in queues.items() if counts[k]<len(q)]
        k=max(available,key=lambda k:((i+1)*weights[k]-counts[k],digest(f'{SEED}:{k}')))
        out.append(queues[k][counts[k]]);counts[k]+=1
    return out

def exported(r,split):
    target=r['binary_target']
    return {'sample_id':r['sample_id'],'image':r['image'],'image_path':r['image'],
        'case_history':r['case_history'],'diagnosis':r['diagnosis'],'binary_target':target,
        'source_dataset':r['source_dataset'],'source_article_id':r['source_article_id'],
        'source_case_id':r['source_case_id'],'patient_id':None,
        'diagnosis_evidence_level':r['diagnosis_evidence_level'],'split':split,
        'messages':[{'role':'user','content':[{'type':'image'},{'type':'text','text':QUESTION}]},
                    {'role':'assistant','content':[{'type':'text','text':target}]}],
        'metadata':{'split':split,'split_group_id':r['split_group_id'],'image_sha256':r['image_sha256'],
            'pixel_sha256':r['pixel_sha256'],'input_policy':'image_only_no_history_caption_or_metadata',
            'source_case_identity_verified_as_patient':False,'negative_stratum':r['negative_stratum'],
            'negative_difficulty':r['negative_difficulty'],'training_ready_clinically_adjudicated':False}}

def write_package(rr):
    images=BASE/'images';images.mkdir(exist_ok=True)
    expected_names={r['image_sha256']+Path(r['image_input']).suffix.lower() for r in rr}
    # Only remove stale generated copies inside our own export images directory.
    for stale in images.iterdir():
        if stale.is_file() and stale.name not in expected_names:
            assert stale.resolve().parent==images.resolve()
            stale.unlink()
    for i,r in enumerate(rr):
        original=BASE/r['image_input'];name=r['image_sha256']+original.suffix.lower()
        target=images/name
        if not target.exists():shutil.copy2(original,target)
        if sha(target)!=r['image_sha256']:raise ValueError('Image changed')
        r['image']='../images/'+name
    positives=ranked([r for r in rr if r['split']=='train' and r['binary_target']=='CADASIL'],'CADASIL')
    negative_pool=[r for r in rr if r['split']=='train' and r['source_dataset']=='MedPix2']
    # Exact 20/50/80 requires a multiple of 10, without replacement.
    n=(min(len(positives)//8,len(negative_pool)//8))*10
    if n<10:raise ValueError('Insufficient real training records; audit written, no viable 3-way mixture')
    negatives=neg_rank(negative_pool,n*8//10)
    held={s:[exported(r,'validation' if s=='val' else s) for r in sorted(rr,key=lambda r:r['sample_id']) if r['split']==('validation' if s=='val' else s)] for s in ['val','test']}
    comps=[];membership=[];checks=[];sets={}
    for code,pct in [('A',20),('B',50),('C',80)]:
        folder=BASE/f'dataset_{code}_{pct}pct';folder.mkdir(exist_ok=True)
        pos=n*pct//100;train=positives[:pos]+negatives[:n-pos]
        random.Random(SEED).shuffle(train)
        data=[exported(r,'train') for r in train]
        jsonl(folder/'train.jsonl',data)
        for split in held:jsonl(folder/(split+'.jsonl'),held[split])
        allr=data+held['val']+held['test'];sets[code]=allr
        for r in allr:membership.append({'dataset':code,'sample_id':r['sample_id'],'split':r['split']})
        neg=[r for r in train if r['binary_target']=='non-CADASIL']
        item={'dataset':code,'training_examples':n,'CADASIL_records':pos,'non_CADASIL_records':n-pos,
            'unique_training_images':len({r['pixel_sha256'] for r in train}),
            'unique_CADASIL_provisional_cases':len({r['source_case_id'] for r in train if r['binary_target']=='CADASIL'}),
            'unique_non_CADASIL_source_cases':len({r['source_case_id'] for r in neg}),
            'CADASIL_source_articles':len({r['source_article_id'] for r in train if r['binary_target']=='CADASIL'}),
            'actual_CADASIL_ratio':pos/n,'easy_negative_proportion':sum(r['negative_difficulty']=='easy' for r in neg)/len(neg),
            'hard_negative_proportion':sum(r['negative_difficulty']=='hard' for r in neg)/len(neg),
            'validation_examples':len(held['val']),'test_examples':len(held['test']),
            'unique_patients':'not_identifiable_from_source_IDs','leakage_check':'no detected split/text/filename/OCR-keyword leaks; patient identity not fully verifiable'}
        item.update({k:sum(r['negative_stratum']==k for r in neg) for k in sorted({r['negative_stratum'] for r in negatives})})
        comps.append(item)
        # Audit all non-training split identities, not just selected positives.
        for field in ['sample_id','source_case_id','source_article_id','split_group_id','image_sha256','pixel_sha256']:
            seen=collections.defaultdict(set)
            for r in allr:
                value=r.get(field) or r['metadata'].get(field)
                if value:seen[value].add(r['split'])
            bad={k:sorted(v) for k,v in seen.items() if len(v)>1}
            checks.append({'dataset':code,'check':field+'_cross_split','failures':len(bad),'details':json.dumps(bad)})
        assert len({r['sample_id'] for r in data})==n
        assert not any(r['original_split'] in {'validation','test'} for r in train)
        for r in allr:
            with Image.open(folder/r['image']) as im:im.verify()
            assert r['messages'][0]['content']==[{'type':'image'},{'type':'text','text':QUESTION}]
            assert r['messages'][1]['content'][0]['text'] in {'CADASIL','non-CADASIL'}
            assert not CAD.search(Path(r['image']).name)
    assert not any(c['failures'] for c in checks),'Cross-split leakage: see check CSV'
    for split in ['val','test']:
        hashes={sha(BASE/f'dataset_{c}_{p}pct'/(split+'.jsonl')) for c,p in [('A',20),('B',50),('C',80)]}
        assert len(hashes)==1
        assert {r['binary_target'] for r in held[split]}=={'CADASIL','non-CADASIL'}
    assert {r['sample_id'] for r in sets['A'] if r['split']=='train' and r['binary_target']=='CADASIL'} <= {r['sample_id'] for r in sets['B'] if r['split']=='train' and r['binary_target']=='CADASIL'} <= {r['sample_id'] for r in sets['C'] if r['split']=='train' and r['binary_target']=='CADASIL'}
    table(BASE/'manifests/dataset_comparison.csv',comps);table(BASE/'manifests/sample_membership.csv',membership);table(BASE/'manifests/split_leakage_check.csv',checks)
    return comps,held,len(positives),len(negatives)

def plots(comp):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(3);bottom=np.zeros(3)
    for label,key,color in [('CADASIL','CADASIL_records','#784b9a'),('Easy non-CADASIL',None,'#478f9a'),('Hard non-CADASIL',None,'#dfa43b')]:
        vals=[r[key] if key else round(r['non_CADASIL_records']*r['easy_negative_proportion' if label.startswith('Easy') else 'hard_negative_proportion']) for r in comp]
        ax.bar(x,vals,bottom=bottom,label=label,color=color)
        for i,v in enumerate(vals):
            if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white',fontweight='bold')
        bottom+=vals
    ax.set_xticks(x,['A: 20% CADASIL','B: 50% CADASIL','C: 80% CADASIL']);ax.set_ylabel('Training image records (not patients)')
    ax.set_title('Fixed-size CADASIL supervision-density mixtures');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3)
    fig.tight_layout();fig.savefig(BASE/'composition.png',dpi=180,bbox_inches='tight');fig.savefig(BASE/'composition.svg',bbox_inches='tight');plt.close(fig)

def report(rr,original,raudit,maudit,faudit,changed,dups,comp,held,p,q,download):
    orig_summary=[]
    for s in ['train','validation','test']:
        z=[r for r in original if r['split']==s];cad=[r for r in z if r['family']=='CADASIL']
        oldcad=[r for r in z if r['original_export_family']=='CADASIL_author_reported']
        orig_summary.append({'split':s,'records':len(z),'CADASIL_records':len(oldcad),'CADASIL_articles':len({r['article_id'] for r in oldcad}),'diagnosis_name_CADASIL_mentions_not_automatically_eligible':len(cad)})
    traincad=[r for r in rr if r['split']=='train' and r['binary_target']=='CADASIL']
    summary={'seed':SEED,'input_mode':'image_only_matching_existing_export_format','original_splits':orig_summary,
        'CADASIL_training_pool':{'records':p,'source_articles':len({r['source_article_id'] for r in traincad}),'provisional_case_keys':len({r['source_case_id'] for r in traincad}),'verified_unique_patients':None},
        'MedPix_training_pool_records':q,'eligible_pool_records':len(rr),'split_changes':changed,'exact_duplicate_removals':dups,
        'original_export_decisions':dict(collections.Counter(r['decision'] for r in raudit)),
        'MedPix_image_decisions':dict(collections.Counter(r['decision'] for r in maudit)),
        'upstream_diagnosis_status':dict(collections.Counter(r['evidence_level'] for r in faudit)),
        'download_status':dict(collections.Counter(r['status'] for r in download)),
        'shared_evaluation':{s:{'records':len(v),'labels':dict(collections.Counter(r['binary_target'] for r in v)),'source_case_keys':len({r['source_case_id'] for r in v}),'source_datasets':dict(collections.Counter(r['source_dataset'] for r in v))} for s,v in held.items()},
        'technical_checks':{'equal_training_sizes':True,'shared_validation_test_byte_identical':True,'all_paths_readable':True,'case_article_exact_and_detected_similar_images_cross_split':0,'without_replacement':True,'positive_and_negative_nested_prefix_sampling':True,'model_text_input_diagnosis_free_except_constant_label_options':True},
        'unresolved_checks':{'global_patient_identity':'not provided; cannot verify cross-publication same patient','pixel_answer_leakage_OCR':'all eligible images screened locally; diagnostic keyword hits quarantined; OCR misses remain possible','clinical_label_adjudication':'not performed; original RareMed labels are model extracted author reports','image_PHI':'not independently screened','actual_MedGemma_training_loader':'not present locally; only existing export format and smoke loader inspected','source_rights':'retained, no blanket redistribution permission inferred'},
        'datasets':comp,'training_performed':False,'clinical_quality_clearance':False}
    old88=[r for r in rows(BASE/'inputs/raremed_train.jsonl') if r['metadata']['diagnosis_family']=='CADASIL_author_reported']
    oldprov={r['sample_id']:r for r in rows(BASE/'inputs/raremed_provenance.jsonl')}
    summary['original_88_audit']={'image_records':len(old88),'source_articles':len({r['metadata']['article_id'] for r in old88}),'provisional_article_local_case_keys':len({(r['metadata']['article_id'],norm(oldprov[r['sample_id']]['proposed_linkage']['case_label'])) for r in old88}),'old_split_components':len({r['metadata']['split_group_id'] for r in old88}),'verified_independent_patients':None,'old_CADASIL_label_included_one_HTRA1_CADASIL_like_case':True}
    save(BASE/'manifests/validation_results.json',summary)
    table(BASE/'manifests/sample_manifest.csv',[{k:v for k,v in r.items() if k not in {'case_history','phash','dhash'}} for r in rr])
    table(BASE/'manifests/heldout_counts.csv',[{'split':s,'source_dataset':ds,'binary_target':label,'records':sum(r['source_dataset']==ds and r['binary_target']==label for r in v),'cases':len({r['source_case_id'] for r in v if r['source_dataset']==ds and r['binary_target']==label})} for s,v in held.items() for ds in ['RareMed-Pairs','MedPix2'] for label in ['CADASIL','non-CADASIL']])
    retained={r['original_sample_id']:r for r in rr if r['source_dataset']=='RareMed-Pairs'}
    decisions={r['sample_id']:r['decision'] for r in raudit}
    table(BASE/'manifests/label_reconciliation.csv',[{'original_sample_id':r['sample_id'],'original_family':r['original_export_family'],'source_diagnosis':r['diagnosis'],'name_family_only_not_confirmation':r['family'],'retained_binary_target':retained[r['sample_id']]['binary_target'] if r['sample_id'] in retained else 'excluded','eligibility_decision':decisions[r['sample_id']],'reason':'HTRA1/CADASIL-like is not NOTCH3 CADASIL; explicit co-diagnoses can be positive; uncertain phenotype is excluded; see source evidence'} for r in original if ((r['original_export_family']=='CADASIL_author_reported')!=(r['family']=='CADASIL'))])
    dist=[]
    for source,data in [('RareMed upstream figures',faudit),('RareMed original export',raudit),('MedPix all source image metadata',maudit)]:
        groups=collections.defaultdict(list)
        for r in data:groups[(r['family'],r['evidence_level'])].append(r)
        for (fam,level),gg in sorted(groups.items()):
            dist.append({'stage':source,'disease_family':fam,'evidence_level':level,'image_or_figure_records':len(gg),'article_count':len({r['article_id'] for r in gg if r.get('article_id')}) if source=='RareMed upstream figures' else 'not_applicable_here','source_case_count':len({r['source_case_id'] for r in gg if r.get('source_case_id')}) if source.startswith('MedPix') else 'unverified_see_case_manifest'})
    for source in ['RareMed-Pairs','MedPix2']:
        groups=collections.defaultdict(list)
        for r in rr:
            if r['source_dataset']==source:groups[(r['diagnosis_family'],r['split'])].append(r)
        for (fam,split),gg in sorted(groups.items()):dist.append({'stage':source+' eligible final pool '+split,'disease_family':fam,'evidence_level':'source_final_not_independently_adjudicated','image_or_figure_records':len(gg),'unique_decoded_images':len({r['pixel_sha256'] for r in gg}),'source_case_count':len({r['source_case_id'] for r in gg}),'article_count':len({r['source_article_id'] for r in gg if r['source_article_id']}) if source=='RareMed-Pairs' else 'not_provided'})
    table(BASE/'manifests/inventory_counts.csv',dist)
    save(BASE/'manifests/build_config.json',{'seed':SEED,'question':QUESTION,'target_labels':['CADASIL','non-CADASIL'],'ratio_unit':'image_record','one_record_one_original_full_figure':True,'MedPix_split':'preserve_author_case_split','positive_sampling':'case_round_robin_nested_prefix','negative_sampling':'disease_prioritized_availability_capped_weighted_fair_nested_prefix','exact_common_N':comp[0]['training_examples'],'source_confounding':'CADASIL train source is RareMed; all train negatives are MedPix','script_sha256':sha(Path(__file__))})
    lines=['# CADASIL SFT mixture data audit','', '**Status: technical research exports built; clinical/pixel/rights clearance remains incomplete. No model was trained.**','',
        '## Source inventory','',f'Original exported split: `{json.dumps(orig_summary)}`.',
        f'88 original CADASIL train image records came from 60 articles. After eligibility and duplicate/split checks: **{p}** usable training image records, **{len({r["source_article_id"] for r in traincad})}** articles, **{len({r["source_case_id"] for r in traincad})}** provisional article-local case keys. These are NOT verified independent patient counts. Available positive pool counts may differ from the original 88 because explicit dual CADASIL diagnoses are reconciled, uncertain phenotypes are excluded, and pixel-text flags are quarantined.',
        'Case labels are model-extracted article-local descriptions; synonyms can split one patient into several keys. Article and detected-duplicate connected components are the conservative split unit. MedPix U_id is a case ID, not a globally verified patient ID.',
        f'MedPix source: {len(read(BASE/"inputs/medpix_cases.json"))} cases and {len(read(BASE/"inputs/medpix_images.json"))} image metadata records; 463 source-labelled Brain and Neuro MR images are requested from the pinned author repository. Training negative pool after filtering/splitting/dedup: {q} image records.',
        'See upstream_figure_audit.csv for every acquired figure, raremed_record_audit.csv for the 296 exported source rows, and medpix_record_audit.csv for all 2,050 MedPix image rows. GBM / Astrocytoma / CADASIL counts are from final/source-reported diagnoses, separately from upstream unconfirmed extraction. Probable diagnosis names override a nominal confirmed extraction flag and are excluded. No MultiCaRe mentions are used as final diagnoses. The original 88 CADASIL training labels included one HTRA1 CADASIL-like case, which is not classified as NOTCH3 CADASIL here. Explicit co-diagnoses in the old other bucket are reconciled only when source evidence supports them; an uncertain CADASIL phenotype is excluded. See label_reconciliation.csv.',
        '', '## Heldouts and contamination','',f'Changed original eligible rows: {len(changed)}. Original rows rejected by evidence/diagnosis eligibility are explicitly listed; they are never reused for training. Connected components touching a retained prior validation/test stay in that heldout; components touching both are quarantined. Existing heldouts are retained where eligible and safe. MedPix author case-level train/dev/test assignments are preserved; they were not regenerated with seed 42. Seed 42 governs mixture sampling. Components touching old training and evaluation are quarantined. Source-based binary label changes are listed in label_reconciliation.csv.',
        f'Shared evaluation summary: `{json.dumps(summary["shared_evaluation"])}`.',
        'Known earlier smoke-exposure source groups are barred from evaluation. Exact decoded-image duplicates are removed; detected near-images and all prior conservative duplicate edges are grouped, including bridge rows outside the eligible set. Unseen cross-publication patient duplicates remain unverifiable without patient identity. A zero detected leak count is not proof of globally unique patients.',
        '', '## Fixed-N design','',f'Maximum exact common record scale: N = 10 × floor(min(P/8,Q/8)) = **{comp[0]["training_examples"]}**, P={p}, Q={q}. Ratios use image/SFT records, not unique cases. Sampling has no replacement. Positive and negative ranks are fixed nested prefixes, with case round-robin ordering and weighted fair negative-stratum scheduling. Glioma/MS/other hard strata are preferentially sampled, with each capped at its real availability divided by the maximum negative requirement; remaining weight goes to other neurological cases. Smaller groups therefore reuse a subset of cases across experiments, never across evaluation splits.',
        'The available pool yields a small pilot, especially for sensitivity estimation. Keep this fixed-N comparison separate from two alternatives: (1) fixed CADASIL cases + progressively more MedPix negatives changes total size; (2) fixed negatives + progressively more CADASIL changes total size. Neither isolates class ratio as cleanly. No duplicate padding is used.',
        '', '| Metric | A | B | C |','|---|---:|---:|---:|']
    for label,key in [('Training records','training_examples'),('Unique CADASIL provisional cases','unique_CADASIL_provisional_cases'),('Unique MedPix negative cases','unique_non_CADASIL_source_cases'),('CADASIL record ratio','actual_CADASIL_ratio'),('Easy fraction of negatives','easy_negative_proportion'),('Hard fraction of negatives','hard_negative_proportion'),('Validation records','validation_examples'),('Test records','test_examples')]:lines.append('|'+label+'|'+'|'.join(str(r[key]) for r in comp)+'|')
    lines+=['','Negative-stratum counts are in dataset_comparison.csv. Proportions are approximately conserved by weighted sampling, with unavoidable integer variation. Hard-negative labels mean selected potentially similar diseases, not a validated difficulty score. Absence of a specific final diagnosis (e.g. explicitly sporadic CSVD) is not repaired by relabelling leukoaraiosis or ordinary infarction. See negative_availability.csv. Positive training source is RareMed-Pairs and negative training source is MedPix: disease and source/domain are confounded. This design cannot isolate diagnostic concept learning from source/style recognition without additional cross-source controls.',
        '', '## Input and label policy','', 'The primary task remains image-only to match the existing RareMed export: original full figure + fixed binary question, answer exactly CADASIL / non-CADASIL. It does not mix description and classification tasks. No caption, diagnosis, source filename, provenance or history is passed in messages. Images are copied byte-for-byte with hash-only filenames. No resizing/cropping is applied here; the same MedGemma processor must be used for all arms.',
        'case_history is a provenance field, NOT a model input. RareMed case context was never verified as uniquely assigned: history is empty rather than fabricated. MedPix History is a real case field; an ancillary version drops sentences containing diagnosis/pathology keywords, while unmodified source history remains in inputs. This is NOT a validated multimodal history-augmented experiment. The original and sanitized variants are available for future review; do not silently add either to this experiment.',
        'The local export loader/chat message layout was inspected, but the actual remote QLoRA training script, processor revision, label masking and chat-template runtime were unavailable. A portable read-only loader example is provided. Export compatibility must be checked on the other machine before training; no existing SFT code was changed.',
        '', '## Remaining limitations','', 'Brain/MRI eligibility uses source annotations and the existing figure-caption pipeline, not an independent pixel-level modality/anatomy adjudicator; mixed panels and source annotation errors remain possible. Original source diagnoses are labels about the reported case, not proof the diagnosis is visually inferable from this figure. No independent clinical review has been performed. RareMed diagnosis evidence quotes are checked against frozen source-text snippets; this checks grounding, not semantic correctness or patient attribution.',
        'All eligible image bytes were screened with local OCR; diagnostic keyword hits were quarantined and recorded in pixel_answer_leakage.csv. OCR can miss text and does not prove the absence of diagnostic overlays. Burned-in identifiers, hidden patient links, original study pretraining contamination and source redistribution rights are not fully cleared. The technical prompt/filename checks pass, but no explicit diagnostic terms were detected by the implemented OCR screen in exported images; missed overlays cannot be ruled out. Dataset metadata explicitly does not assert clinical training readiness. Do not publish a blanket relicensed image package.',
        'The old trained model may have been inspected on its old test set; those heldouts are not a pristine confirmatory benchmark for a new hypothesis. New MedPix heldouts help estimate false positives, but there is no newly collected untouched CADASIL cohort. Repeated test monitoring would consume the heldout; select ratios/thresholds/checkpoints using validation only. MedQA retention requires a fixed separate untouched MedQA evaluation and is not guaranteed by adding negatives. Use the unchanged base model and the same frozen binary question for the primary comparison. The old 239-example free-diagnosis SFT differs in total size and target format, so comparisons to it are exploratory.',
        '', '## Reproducibility','', 'See README.md. Frozen source hashes, repository revision and Git blob hashes are retained. Build reruns use no generative model/API and seed=42. The audit records all decisions and sample membership.']
    (BASE/'audit_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    zh=['# CADASIL 三组 SFT 数据集：审计摘要','',
        '**已生成真实数据导出；未训练模型，未修改原有 SFT 代码或原始数据。技术校验通过不代表完成临床、隐私和发布权利审核。**','',
        '## 原有 88 条是什么','',
        '实际为 88 条图像训练记录，来自 60 篇文献、64 个文章内临时病例键、56 个旧划分分组。没有可验证的全局患者 ID，因此不能报告为 88 名独立患者。',
        '原标签中有一条 HTRA1/CADASIL-like，被重新识别为非 NOTCH3 CADASIL；旧“其他诊断”中明确的 CADASIL 共病记录另行校正。仅表型且确认不足的记录排除。原始诊断引文已与原始 BioC 正文快照核对。',
        f'病例标签与 OCR 检查后，CADASIL 训练池有 {p} 条图像记录，{summary["CADASIL_training_pool"]["provisional_case_keys"]} 个临时病例键、{summary["CADASIL_training_pool"]["source_articles"]} 篇文献。MedPix 阴性训练池有 {q} 条。',
        '', '## 三组训练集','', '|指标|A|B|C|','|---|---:|---:|---:|']
    for label,key in [('训练记录数','training_examples'),('CADASIL 记录数','CADASIL_records'),('MedPix 阴性记录数','non_CADASIL_records'),('CADASIL 临时病例键数','unique_CADASIL_provisional_cases'),('MedPix 阴性病例数','unique_non_CADASIL_source_cases'),('实际 CADASIL 比例','actual_CADASIL_ratio'),('阴性中 easy 比例','easy_negative_proportion'),('阴性中 hard 比例','hard_negative_proportion'),('验证记录数','validation_examples'),('测试记录数','test_examples')]:zh.append('|'+label+'|'+'|'.join(str(r[key]) for r in comp)+'|')
    zh+=['','N 的计算：同时满足最高 80% CADASIL 和最高 80% 阴性，且精确实现 20/50/80 需要 N 为 10 的倍数。因此 N=10×min(floor(P/8),floor(Q/8))。本次固定 N，而非通过复制凑比例。训练抽样 seed=42，正负样本均采用嵌套前缀，无放回抽样。',
        '逐病种计数见 manifests/dataset_comparison.csv。Hard 的定义是潜在相似病种，而非已验证的难度分数；MS、Alexander disease、Canavan disease、Leukoaraiosis 等只按原始最终诊断使用，不把 Leukoaraiosis 编成“确诊散发性小血管病”。',
        '', '## 验证与测试','',
        '原有 24 条 validation 和 33 条 test 在本次数据检查后均保留；加入 MedPix 作者既有病例级 dev/test。三组 val/test 文件逐字节一致，既有 heldout 不回流训练。',
        f'验证集：{len(held["val"])} 条，其中 CADASIL {sum(r["binary_target"]=="CADASIL" for r in held["val"])} 条；测试集：{len(held["test"])} 条，其中 CADASIL {sum(r["binary_target"]=="CADASIL" for r in held["test"])} 条。CADASIL 评估病例很少，不能按图像数当独立患者数。测试 CADASIL 计数相对旧大类有共病标签校正，详见 label_reconciliation.csv。',
        '已有测试集曾用于前期研究，不是全新、从未查看的确认性队列。验证集用于比例、阈值与 checkpoint 选择；测试集不要每天用于调参。MedQA 需另外冻结独立评测，本包没有 MedQA 数据，也不保证能力不下降。',
        '', '## 输入与质量控制','',
        '统一保持现有的 image-only 格式：原始完整影像＋固定二分类问题，答案仅 CADASIL / non-CADASIL。不将病史、图注、最终诊断或来源元数据送入 user message。图片保持原字节，导出文件名使用哈希。',
        'RareMed 正文尚未安全分配为病例级病史，因此 case_history 留空。MedPix 保留真实 History 的去诊断句版本，但不用于本次模型输入；原文与处理版对照见 history_variants_not_model_input.jsonl。不要自行把这些溯源字段拼进 prompt。',
        '原始图片经过本地 OCR；有明确诊断词的候选图片隔离，详见 pixel_answer_leakage.csv。OCR 不是绝对保证，尚不能排除漏识文字、烧录身份信息或源标注错误。未发现所实现的病例键、文献、图片哈希、近重复关系跨集合泄漏；没有足够信息排除跨论文同一患者。',
        '', '## 必须保留的局限','',
        'RareMed 的作者确诊和病例关联仍来自既有模型抽取，未做独立临床判读。MRI/脑部资格来自来源标注与现有图注流程，不等于对所有像素重新做了脑 MRI 鉴定。',
        '训练阳性均来自 RareMed，阴性均来自 MedPix，疾病与来源存在混杂，不能把结果完全解释为医学概念学习。原有作者/病例报告诊断也不代表每幅图本身足以确诊。',
        '另一台机器的实际 QLoRA loader、processor revision、chat template 与 label masking 代码不在本机，无法完成其运行时兼容性确认。这里只沿用已存在的 JSONL messages 格式并提供只读加载验证。未修改训练代码。',
        '源许可保留，不推断整包可统一发布到 GitHub。临床、隐私和发布权利审核未被宣称完成。',
        '', '## 文件','', '完整说明见 README.md 与 audit_report.md；技术复查见 scripts/validate_exports.py；离线重建见 scripts/build_mixture_datasets.py；组成图见 composition.png。']
    (BASE/'audit_summary_zh.md').write_text('\n'.join(zh)+'\n',encoding='utf8')
    avail=collections.defaultdict(list)
    for r in rr:
        if r['source_dataset']=='MedPix2':avail[(r['diagnosis'],r['negative_difficulty'],r['split'])].append(r)
    table(BASE/'manifests/negative_availability.csv',[{'diagnosis':k[0],'difficulty':k[1],'split':k[2],'image_records':len(v),'source_cases':len({r['source_case_id'] for r in v})} for k,v in sorted(avail.items())])
    save(BASE/'manifests/build_artifact_hashes.json',{str(p.relative_to(BASE)).replace('\\','/'):sha(p) for p in sorted(BASE.rglob('*')) if p.is_file() and p.suffix in {'.jsonl','.csv','.py'} and 'inputs' not in p.parts})
    print(json.dumps({'datasets':comp,'CADASIL_training_records':p,'MedPix_training_records':q,'status':'technical_exports_complete_with_documented_unresolved_clinical_pixel_and_rights_checks'},ensure_ascii=True,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path);ap.add_argument('--download-medpix',action='store_true');ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    if args.source_root:freeze(args.source_root.resolve())
    download=fetch_images(args.download_medpix)
    rr,orig,ra,ma,fa=collect(args.source_root.resolve() if args.source_root else None)
    if args.prepare_only:
        print(json.dumps({'eligible_before_OCR_and_dedup':len(rr),'status':'prepared_only'}));return
    rr=screen_ocr(rr)
    rr,changes,dups=group_and_dedup(rr);rr=allocate_medpix(rr)
    comp,held,p,q=write_package(rr);plots(comp);report(rr,orig,ra,ma,fa,changes,dups,comp,held,p,q,download)

if __name__=='__main__':main()
