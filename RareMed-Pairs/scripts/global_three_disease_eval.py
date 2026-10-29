"""Full-corpus three-disease extraction evaluation; no article/sample cap.

Uses all local public-source metadata, not a new PMC crawl. Original text-only
extraction prompt/schema/runner are frozen. Batches limit request size, never
the number of evaluated records. Paid inference requires explicit --allow-api.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import random
import re
import time
import csv
import pyarrow.parquet as pq
import run_deepseek
from dedup_leakage import rows, write_rows, save
from medcpt_local import ROOT, sha
from task_benchmark_suite import text_scores

SRC=ROOT/'data/benchmarks/four_disease/sources'
OUT=ROOT/'data/benchmarks/global_three_disease_v1'
DISEASES={'CADASIL':r'\bCADASIL\b|cerebral autosomal dominant arteriopathy',
          'MS':r'\bmultiple sclerosis\b',
          'NMOSD':r'neuromyelitis optica|\bNMOSD\b|\bDevic(?:[’\x27]s)?\b'}
SOURCE_FILES=['medpix_cases.json','medpix_images.json','multicare_cases.parquet','multicare_images.parquet']

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def norm(s):return ' '.join(re.findall(r'\w+',s.lower()))
def mentions(text):
    found=[d for d,p in DISEASES.items() if re.search(p,text,re.I)]
    if 'MS' not in found and re.search(r'\bMS\b',text):found.append('MS')
    return sorted(found)

def parquet_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        yield from batch.to_pylist()

def fingerprint():
    return {'sources':{n:sha(SRC/n) for n in SOURCE_FILES},
        'pipeline':{n:sha(ROOT/'scripts'/n) for n in ['run_deepseek.py','link_cases.py','extraction_format.py']},
        'prompt':sha(ROOT/'configs/linkage_prompt.txt'),'schema':sha(ROOT/'configs/linkage_schema.json'),
        'retrieval':DISEASES,'MS_abbreviation':'uppercase standalone MS is retrieval-only, never a diagnosis label',
        'figures_per_request':8,'source_view':'all source case narratives plus original whole-figure captions; not full PMC acquisition',
        'script':sha(Path(__file__))}

def scan():
    OUT.mkdir(parents=True,exist_ok=True)
    identity=fingerprint()
    if (OUT/'catalog_manifest.json').exists() and load(OUT/'catalog_manifest.json')['identity']!=identity:
        raise ValueError('Sources/code changed after freeze; use a new version directory')
    docs=[];scanned=Counter();counts=defaultdict(Counter)
    for r in parquet_rows(SRC/'multicare_cases.parquet'):
        scanned['MultiCaRe_articles']+=1;scanned['MultiCaRe_cases']+=len(r['cases'])
        hit={c['case_id']:mentions(c['case_text']) for c in r['cases']}
        hit={k:v for k,v in hit.items() if v}
        if hit:
            docs.append({'dataset':'MultiCaRe','document_id':r['article_id'],'cases':r['cases'],
                'case_mention_strata':hit,'mention_strata':sorted({d for v in hit.values() for d in v})})
            for d in {d for v in hit.values() for d in v}:counts['MultiCaRe_document_mentions'][d]+=1
    for r in load(SRC/'medpix_cases.json'):
        scanned['MedPix_cases']+=1;hit=mentions(json.dumps(r['Case'],ensure_ascii=False))
        if hit:
            docs.append({'dataset':'MedPix2','document_id':r['U_id'],'raw_case':r,'mention_strata':hit})
            for d in hit:counts['MedPix_case_mentions'][d]+=1
    docs.sort(key=lambda x:(x['dataset'],x['document_id']))
    write_rows(OUT/'catalog.jsonl',docs)
    save(OUT/'catalog_manifest.json',{'identity':identity,'scanned':dict(scanned),'selected_documents':len(docs),
        'selected_by_dataset':dict(Counter(d['dataset'] for d in docs)),'disease_mention_counts':{k:dict(v) for k,v in counts.items()},
        'catalog_sha256':sha(OUT/'catalog.jsonl'),'selection':'ALL matching records; no per-disease or document limit',
        'labels':'Mention strata are retrieval only. Final diagnoses are references only when independently available.'})
    print(json.dumps(load(OUT/'catalog_manifest.json'),ensure_ascii=True,indent=2),flush=True)

def source_passages(cases):
    result=[];offset=0
    for i,c in enumerate(cases,1):
        text=f'Case {i}\n'+c['text']
        # Do not silently truncate long case narratives.
        result.append({'passage_id':f'p{i-1:05d}','section':'CASE','source_offset':offset,'text':text})
        offset+=len(text)+1
    return result,offset

def prepare():
    if not (OUT/'catalog_manifest.json').exists():scan()
    catalog_m=load(OUT/'catalog_manifest.json')
    if catalog_m['identity']!=fingerprint() or sha(OUT/'catalog.jsonl')!=catalog_m['catalog_sha256']:raise ValueError('Frozen catalog changed')
    catalog=rows(OUT/'catalog.jsonl')
    selected={d['document_id']:d for d in catalog if d['dataset']=='MultiCaRe'}
    fixtures={r['article_id']:r['case_images'] for r in parquet_rows(SRC/'multicare_images.parquet') if r['article_id'] in selected}
    jobs=[];references=[];held=[];input_hashes={};prompt_hashes={};coverage=Counter()
    template=(ROOT/'configs/linkage_prompt.txt').read_text(encoding='utf-8')
    schema=(ROOT/'configs/linkage_schema.json').read_text(encoding='utf-8')
    def add_document(uid,dataset,cases,figures,strata):
        if not figures:
            held.append({'document_id':uid,'dataset':dataset,'reason':'no_source_figure_associations'});return
        passages,offset=source_passages(cases)
        for batch_no,start in enumerate(range(0,len(figures),8),1):
            group=figures[start:start+8]
            job=uid+f'_B{batch_no:04d}'
            request={'article_id':job,'passages':list(passages),'candidate_figures':[]}
            off=offset
            for f in group:
                request['candidate_figures'].append({'figure_id':f['figure_id'],'caption':f['caption'],'caption_source_offset':off})
                request['passages'].append({'passage_id':f'p{len(request["passages"]):05d}','section':'FIG','source_offset':off,'text':f['caption']})
                off+=len(f['caption'])+1
            prompt=template.replace('{{SCHEMA}}',schema).replace('{{ARTICLE_INPUT}}',json.dumps(request,ensure_ascii=False,indent=2))
            if len(prompt)>120000:
                held.append({'document_id':uid,'dataset':dataset,'job':job,'reason':'full_context_over_120000_chars_no_truncation',
                    'figure_ids':[f['figure_id'] for f in group]});continue
            folder=OUT/'work/data/model_inputs'/job;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'input.json').exists() and load(folder/'input.json')!=request:raise ValueError('Existing input changed: '+job)
            save(folder/'input.json',request);(folder/'prompt.txt').write_text(prompt,encoding='utf-8')
            input_hashes[job]=sha(folder/'input.json');prompt_hashes[job]=sha(folder/'prompt.txt')
            jobs.append({'job':job,'dataset':dataset,'source_document':uid,'figures':len(group)})
            for f in group:
                references.append({'job':job,'dataset':dataset,'source_document':uid,'figure_id':f['figure_id'],
                    'source_case_ids':f['case_ids'],'reference_case_labels':f['case_labels'],
                    'source_image_ids':f['image_ids'],'source_caption':f['caption'],
                    'mention_strata':strata,'reference_diagnosis':f.get('diagnosis'),
                    'case_passage_identity':{f'p{i:05d}':c['case_id'] for i,c in enumerate(cases)},
                    'case_identities':{f'Case {i+1}':c['case_id'] for i,c in enumerate(cases)},
                    'diagnosis_reference_type':'structured_final_field' if dataset=='MedPix2' else 'not_available_do_not_score_generated_labels'})
            coverage[dataset+'_figure_trials']+=len(group)
    for index,d in enumerate([d for d in catalog if d['dataset']=='MultiCaRe'],1):
        uid=d['document_id'];cases=[{'case_id':c['case_id'],'text':c['case_text']} for c in d['cases']]
        labels={c['case_id']:f'Case {i+1}' for i,c in enumerate(cases)}
        figures={}
        for case in fixtures.get(uid,[]):
            for im in case['case_image_list']:
                tag=im['tag'];caption=im['caption'] or ''
                if not re.fullmatch(r'[A-Za-z0-9_-]+',tag or ''):
                    held.append({'dataset':'MultiCaRe','document_id':uid,'reason':'unsafe_or_missing_figure_tag','tag':tag});continue
                if tag in figures and norm(figures[tag]['caption'])!=norm(caption):
                    # Same tag may have case-specific caption variants. Do not
                    # choose a variant using predictions or silently merge it.
                    figures[tag]['conflict']=True
                f=figures.setdefault(tag,{'figure_id':tag,'caption':caption,'case_ids':[],'case_labels':[],'image_ids':[]})
                if case['case_id'] not in labels:
                    held.append({'dataset':'MultiCaRe','document_id':uid,'reason':'orphan_source_case','case_id':case['case_id']});continue
                f['case_ids'].append(case['case_id']);f['case_labels'].append(labels[case['case_id']]);f['image_ids'].append(im['image_id'])
        clean=[]
        for f in figures.values():
            if f.get('conflict'):
                held.append({'dataset':'MultiCaRe','document_id':uid,'reason':'conflicting_source_captions_same_tag','tag':f['figure_id']});continue
            for k in ['case_ids','case_labels','image_ids']:f[k]=sorted(set(f[k]))
            if f['case_ids']:clean.append(f)
        add_document(uid,'MultiCaRe',cases,clean,d['mention_strata'])
        if index%25==0:print(f'Prepared all-source MultiCaRe documents {index}/{len(selected)}',flush=True)
    # All matching MedPix cases, with two distractors per group. Figure batches
    # preserve the same three candidate cases; no gold UID in model input.
    med=sorted([d for d in catalog if d['dataset']=='MedPix2'],key=lambda d:hashlib.sha256(d['document_id'].encode()).hexdigest())
    all_med=load(SRC/'medpix_cases.json');images=load(SRC/'medpix_images.json');rng=random.Random(20261027)
    selected_ids={d['document_id'] for d in med}
    for groupindex,start in enumerate(range(0,len(med),3),1):
        target=med[start:start+3];raw=[d['raw_case'] for d in target]
        used={r['U_id'] for r in raw}
        raw+=[c for c in all_med if c['U_id'] not in used and c['U_id'] not in selected_ids][:3-len(raw)]
        rng.shuffle(raw);labels={c['U_id']:f'Case {i+1}' for i,c in enumerate(raw)}
        cases=[{'case_id':c['U_id'],'text':'\n'.join(k+': '+v for k,v in c['Case'].items() if isinstance(v,str) and v.strip() not in {'','N/A'})} for c in raw]
        target_ids={d['document_id'] for d in target};ims=[im for im in images if im['U_id'] in target_ids];rng.shuffle(ims)
        cmap={c['U_id']:c for c in raw}
        figs=[{'figure_id':f'F{i+1:04d}','caption':im['Description'].get('Caption') or '',
            'case_ids':[im['U_id']],'case_labels':[labels[im['U_id']]],'image_ids':[im['image']],
            'diagnosis':cmap[im['U_id']]['Case'].get('Case Diagnosis','')} for i,im in enumerate(ims)]
        add_document(f'MEDPIX_ALL_{groupindex:04d}','MedPix2',cases,figs,sorted({s for d in target for s in d['mention_strata']}))
    write_rows(OUT/'jobs.jsonl',jobs);write_rows(OUT/'references.jsonl',references);write_rows(OUT/'held.jsonl',held)
    save(OUT/'prepared_manifest.json',{'identity':catalog_m['identity'],'catalog_sha256':catalog_m['catalog_sha256'],
        'jobs_sha256':sha(OUT/'jobs.jsonl'),'references_sha256':sha(OUT/'references.jsonl'),
        'inputs':input_hashes,'prompts':prompt_hashes,'counts':dict(coverage),'jobs':len(jobs),
        'held':dict(Counter(r['reason'] for r in held)),
        'scope':'All matching local public-source documents, all source-associated figure tags, no modality prefilter. Eight figures/request is batching, not sampling.',
        'no_diagnosis_reference_policy':'MultiCaRe association scored by retrieval strata only. No full-corpus diagnosis accuracy without independent source diagnosis references.'})
    score()

def check():
    m=load(OUT/'prepared_manifest.json')
    if m['identity']!=fingerprint():raise ValueError('Frozen source/pipeline changed')
    for name,key in [('catalog.jsonl','catalog_sha256'),('jobs.jsonl','jobs_sha256'),('references.jsonl','references_sha256')]:
        if sha(OUT/name)!=m[key]:raise ValueError('Frozen file changed: '+name)
    for key,file in [('inputs','input.json'),('prompts','prompt.txt')]:
        for j,h in m[key].items():
            if sha(OUT/'work/data/model_inputs'/j/file)!=h:raise ValueError('Prepared input changed: '+j)

def output_folder(job):
    folder=OUT/'work/data/model_inputs'/job
    payload=run_deepseek.build_request(folder,8192)
    h=hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode('utf-8')).hexdigest()
    return OUT/'work/data/model_outputs'/run_deepseek.MODEL/job/h[:16]

def infer(allow_api):
    check();jobs=rows(OUT/'jobs.jsonl');run_deepseek.ROOT=OUT/'work'
    if not allow_api:
        total=sum(len((OUT/'work/data/model_inputs'/j['job']/'prompt.txt').read_text(encoding='utf-8')) for j in jobs)
        print(json.dumps({'dry_run':True,'jobs':len(jobs),'figures':sum(j['figures'] for j in jobs),
            'prompt_characters':total,'max_output_tokens_per_request':8192,'no_sample_limit':True},indent=2));return
    key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key:raise ValueError('Set DEEPSEEK_API_KEY in this terminal; never send it in chat')
    for index,j in enumerate(jobs,1):
        job=j['job'];out=output_folder(job)
        if (out/'receipt.json').exists():
            print(f'[{index}/{len(jobs)}] Existing attempt retained: {job}',flush=True);continue
        print(f'[{index}/{len(jobs)}] {j["dataset"]} {job} ({j["figures"]} figures)',flush=True)
        ok=run_deepseek.call_one(job,key,8192)
        if not ok:
            r=load(out/'receipt.json') if (out/'receipt.json').exists() else {}
            if r.get('status')!='validation_failed':
                print('Stopped on provider/account/network error; no automatic retry.',flush=True);break
        if index%10==0:score(check_files=False)
        time.sleep(.25)
    score(check_files=False)

def predicted_cases(link,source,reference):
    # No reference target IDs used for mapping: only all candidate case labels
    # and exact evidence membership in candidate narratives.
    if not link:return []
    if link.get('link_status')=='multiple_cases':return []
    label=link.get('case_label') or ''
    match=re.fullmatch(r'case\s*(\d+)',label.strip(),re.I)
    mapping=reference['case_identities']
    if match and 'Case '+match[1] in mapping:return [mapping['Case '+match[1]]]
    found=set()
    for key in ['link_evidence','diagnosis_evidence','basis_evidence']:
        for e in link.get(key,[]):
            q=norm(e.get('quote',''))
            hits=[p['passage_id'] for p in source['passages'] if p['section']=='CASE' and len(q)>=25 and q in norm(p['text'])]
            if len(hits)==1:found.add(reference['case_passage_identity'][hits[0]])
    return sorted(found)

def relation_metrics(data):
    tp=sum(len(set(r['predicted_case_ids'])&set(r['source_case_ids'])) for r in data)
    fp=sum(len(set(r['predicted_case_ids'])-set(r['source_case_ids'])) for r in data)
    fn=sum(len(set(r['source_case_ids'])-set(r['predicted_case_ids'])) for r in data)
    def div(a,b):return a/b if b else None
    return {'figures':len(data),'source_cases':len({i for r in data for i in r['source_case_ids']}),
        'TP':tp,'FP':fp,'FN':fn,'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),'f1':div(2*tp,2*tp+fp+fn),
        'exact_case_set_accuracy':div(sum(set(r['predicted_case_ids'])==set(r['source_case_ids']) for r in data),len(data)),
        'validated_figures':sum(r['inference_status']=='validated' for r in data)}

def score(check_files=True):
    if check_files:check()
    refs=rows(OUT/'references.jsonl');jobs=rows(OUT/'jobs.jsonl');cache={};states=Counter();usage=Counter()
    for j in jobs:
        name=j['job'];out=output_folder(name);linkmap={};state='not_run'
        if (out/'receipt.json').exists():
            receipt=load(out/'receipt.json');state=receipt['status']
            usage.update({k:v for k,v in receipt.get('usage',{}).items() if isinstance(v,int)})
        if (out/'api_response.json').exists():
            parsed,errors,_=run_deepseek.check_completion(load(out/'api_response.json'),load(OUT/'work/data/model_inputs'/name/'input.json'))
            state='validation_failed' if errors else 'validated'
            if not errors:linkmap={r['figure_id']:r for r in parsed['figures']}
        states[state]+=1;cache[name]=(state,linkmap,load(OUT/'work/data/model_inputs'/name/'input.json'))
    details=[]
    for ref in refs:
        state,mapping,source=cache[ref['job']];link=mapping.get(ref['figure_id'])
        details.append({**ref,'inference_status':state,'predicted_case_ids':predicted_cases(link,source,ref),
            'prediction':link,'diagnosis_text_scores':text_scores((link or {}).get('diagnosis_name') or '',ref['reference_diagnosis'])
                if ref['reference_diagnosis'] else None})
    write_rows(OUT/'prediction_details.jsonl',details)
    metrics=[]
    for ds in ['MedPix2','MultiCaRe']:
        for group in ['ALL']+list(DISEASES):
            subset=[r for r in details if r['dataset']==ds and (group=='ALL' or group in r['mention_strata'])]
            if not subset:continue
            valid=[r for r in subset if r['inference_status']=='validated']
            metrics.append({'dataset':ds,'stratum':group,'stratum_type':'document disease MENTION; not confirmed diagnosis',
                'all_planned_missing_as_no_association':relation_metrics(subset),
                'validated_only':relation_metrics(valid) if valid else None,
                'coverage':len(valid)/len(subset)})
    meddiag=[r for r in details if r['reference_diagnosis'] and r['inference_status']=='validated']
    diagnosis={}
    for group in ['ALL']+list(DISEASES):
        subset=[r for r in meddiag if group=='ALL' or group in mentions(r['reference_diagnosis'])]
        diagnosis[group]={'n':len(subset),'cases':len({i for r in subset for i in r['source_case_ids']}),
            'scores':{k:sum(float(r['diagnosis_text_scores'][k]) for r in subset)/len(subset) for k in subset[0]['diagnosis_text_scores']} if subset else None,
            'reference_type':'MedPix original final-diagnosis field; no disease-positive score when n=0'}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'catalog':load(OUT/'catalog_manifest.json'),
        'prepared':load(OUT/'prepared_manifest.json'),'inference_states':dict(states),'usage':dict(usage),
        'association_metrics':metrics,'MedPix_diagnosis_name_by_final_disease':diagnosis,
        'complete':bool(jobs) and sum(states[s] for s in ['validated','validation_failed'])==len(jobs),
        'note':'Only original public case associations and MedPix final diagnosis fields are scored. Mention-stratum metrics are not disease-confirmation accuracy. MultiCaRe full-corpus diagnosis scoring dropped: no independent labels. Not-run and invalid outputs explicitly retained. No aggregate end-to-end score.'}
    save(OUT/'summary.json',result)
    with (OUT/'association_metrics.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=['dataset','stratum','coverage','figures','source_cases','TP','FP','FN','precision','recall','f1','exact_case_set_accuracy','validated_figures']);w.writeheader()
        for r in metrics:w.writerow({'dataset':r['dataset'],'stratum':r['stratum'],'coverage':r['coverage'],**r['all_planned_missing_as_no_association']})
    html='<!doctype html><meta charset="utf-8"><title>全量三病来源评测</title><style>body{font-family:Arial,"Microsoft YaHei";margin:32px;line-height:1.7}pre{white-space:pre-wrap}</style><h1>全量 CADASIL / MS / NMOSD 来源评测</h1><p>遍历所有本地公开病例记录，不设每病/文档/图片抽样上限。8图一批仅控制单次请求长度。病例—图注为文本输入，未下载或分析图像像素。</p><p>候选按疾病提及检索，不把提及当最终诊断。MedPix诊断评分按最终诊断字段分组；MultiCaRe仅按候选文档提及分层评估来源病例关联，缺诊断真值不评分。</p><p><a href="association_metrics.csv">关联指标CSV</a> · <a href="held.jsonl">无法准备的记录及原因</a> · <a href="prediction_details.jsonl">全部逐图记录</a></p><pre>'+escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>'
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'jobs':len(jobs),'figures':len(details),'states':dict(states),'held':result['prepared']['held'],
        'complete':result['complete'],'report':str(OUT/'report.html')},ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['scan','prepare','infer','score','run'])
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.command=='scan':scan()
    elif args.command=='prepare':prepare()
    elif args.command=='infer':infer(args.allow_api)
    elif args.command=='score':score()
    else:
        if not (OUT/'prepared_manifest.json').exists():prepare()
        infer(args.allow_api)
