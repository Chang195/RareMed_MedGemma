"""Four-disease evaluation entry point using frozen existing extraction functions.

coverage -> prepare -> infer (explicit paid opt-in) -> score (verified references).
PMC acquisition is isolated from training data. MedPix is a structured-source adapter,
not an end-to-end PMC acquisition test. No gold labels are passed to DeepSeek.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import pyarrow.parquet as pq
import collect
import link_cases
import run_deepseek
from inventory_dataset import MRI
from preview_dataset import ANIMAL, NONPATIENT
from medcpt_local import ROOT, sha
from dedup_leakage import write_rows, rows, save
from dataset_leakage import normalize
from evaluate_curation import metrics

BASE=ROOT/'data/benchmarks/four_disease'
CONFIG=ROOT/'configs/four_disease_eval_v1.json'


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def disease_matches(text, config):
    return [k for k,p in config['diseases'].items() if re.search(p,text,re.I|re.S)]


def catalog(config):
    docs=[]
    pf=pq.ParquetFile(BASE/'sources/multicare_cases.parquet')
    for batch in pf.iter_batches(batch_size=256):
        for r in batch.to_pylist():
            hits=sorted({d for c in r['cases'] for d in disease_matches(c['case_text'],config)})
            if hits: docs.append({'dataset':'MultiCaRe','document_id':r['article_id'],'strata':hits,'cases':r['cases']})
    for r in load(BASE/'sources/medpix_cases.json'):
        # Search case only, not generic Topic/Disease Discussion.
        hits=disease_matches(json.dumps(r['Case'],ensure_ascii=False),config)
        if hits: docs.append({'dataset':'MedPix2','document_id':r['U_id'],'strata':hits,'raw_case':r})
    return docs


def coverage(args):
    print('Scanning public case metadata for the three fixed disease strata...',flush=True)
    config=load(CONFIG); docs=catalog(config)
    selected={}
    counts={}
    for ds in ['MultiCaRe','MedPix2']:
        counts[ds]={}
        for disease in config['diseases']:
            bucket=[d for d in docs if d['dataset']==ds and disease in d['strata']]
            counts[ds][disease]=len(bucket)
            bucket.sort(key=lambda r:hashlib.sha256((config['seed']+r['document_id']).encode()).hexdigest())
            for r in bucket[:config['per_disease_per_dataset']]: selected[(ds,r['document_id'])]=r
    out=BASE/'v1';out.mkdir(parents=True,exist_ok=True)
    selection=list(selected.values())
    identity={'config_sha256':sha(CONFIG),'source_sha256':{p.name:sha(p) for p in (BASE/'sources').iterdir() if p.suffix in {'.json','.parquet'}}}
    if (out/'selection_manifest.json').exists() and load(out/'selection_manifest.json')['identity']!=identity:
        raise ValueError('Frozen sources/config changed; use a new benchmark version')
    write_rows(out/'selected_documents.jsonl',selection)
    save(out/'selection_manifest.json',{'identity':identity,'coverage_document_mentions':counts,
        'selected_documents':len(selection),'selection_sha256':sha(out/'selected_documents.jsonl'),
        'note':config['selection_note'],'new_api_calls':0})
    print(json.dumps({'coverage_document_mentions':counts,'selected_documents':len(selection),'output':str(out)},indent=2))


def fixture_lookup(selection):
    selected={d['document_id'] for d in selection if d['dataset']=='MultiCaRe'}
    result={}
    for batch in pq.ParquetFile(BASE/'sources/multicare_images.parquet').iter_batches(batch_size=256):
        for r in batch.to_pylist():
            if r['article_id'] in selected: result[r['article_id']]=r['case_images']
    return result


def native_medpix(doc):
    r=doc['raw_case'];uid=doc['document_id']
    # No stable patient ID or per-image source label is supplied to extractor.
    passages=[];offset=0
    for heading,text in r['Case'].items():
        if not isinstance(text,str) or text.strip() in {'','N/A'}: continue
        content=heading+': '+text
        passages.append({'offset':offset,'text':content,'metadata':{'section_type':'CASE'}})
        offset+=len(content)+1
    figures=[]
    for img in load(BASE/'sources/medpix_images.json'):
        if img['U_id']!=uid:continue
        caption=img['Description'].get('Caption') or ''
        fid=img['image'];capt_off=offset
        passages.append({'offset':offset,'text':caption,'metadata':{'section_type':'FIG','type':'fig_caption','id':fid}})
        offset+=len(caption)+1
        figures.append({'sample_id':uid+'_'+fid,'article_id':uid,'figure_id':fid,'caption':caption,
            'caption_evidence':{'offset':capt_off},'assets':[], 'native_source':img})
    return passages,figures


def prepare(args):
    out=BASE/'v1'; manifest=load(out/'selection_manifest.json')
    if sha(out/'selected_documents.jsonl')!=manifest['selection_sha256']: raise ValueError('Selection changed')
    selection=rows(out/'selected_documents.jsonl');fixtures=fixture_lookup(selection)
    # Isolate caches and model inputs; leave original project artifacts untouched.
    work=out/'work';work.mkdir(parents=True,exist_ok=True)
    collect.ROOT=work
    all_figures=[];references=[];jobs=[];statuses=[]
    dl=collect.Downloader(offline=args.cached_only)
    for i,doc in enumerate(selection,1):
        uid=doc['document_id'];folder=work/'data/model_inputs'/uid
        print(f'Preparing {i}/{len(selection)}: {doc["dataset"]} {uid}',flush=True)
        try:
            if doc['dataset']=='MultiCaRe':
                # All figures retained in universe; original mri_only gate is
                # evaluated separately. Pre-filtering would hide false negatives.
                figs,receipt=collect.collect_article(uid,dl,mri_only=False,license_policy='exclude_by_nc_nd')
                if not figs:
                    statuses.append({'document_id':uid,'status':receipt['status']});continue
                article=load(work/'data/processed/articles'/(uid+'.json'))
                passages=[{'passage_id':f'p{j:04d}','section':p['metadata'].get('section_type'),
                           'source_offset':p['offset'],'text':p['text']} for j,p in enumerate(article['passages']) if p['metadata'].get('section_type')!='REF']
                for f in figs:
                    matches=[(c['case_id'],im) for c in fixtures.get(uid,[]) for im in c['case_image_list'] if im['tag']==f['figure_id']]
                    references.append({'sample_id':f['sample_id'],'dataset':doc['dataset'],'strata':doc['strata'],
                        'source_case_ids':sorted({x[0] for x in matches}),
                        'source_captions':[x[1]['caption'] for x in matches],
                        'source_text_references':[t for x in matches for t in x[1]['text_references']],
                        'case_source':doc['cases'],'reference_status':'silver_unverified'})
            else:
                raw,figs=native_medpix(doc)
                passages=[{'passage_id':f'p{j:04d}','section':p['metadata']['section_type'],'source_offset':p['offset'],'text':p['text']} for j,p in enumerate(raw)]
                for f in figs:
                    native=f.pop('native_source')
                    references.append({'sample_id':f['sample_id'],'dataset':doc['dataset'],'strata':doc['strata'],
                        'source_case_ids':[uid], 'source_captions':[f['caption']], 'source_modality':native['Type'],
                        'source_location':native.get('Location'), 'source_location_category':native.get('Location Category'),
                        'source_diagnosis':doc['raw_case']['Case'].get('Case Diagnosis'),
                        'source_diagnosis_basis':doc['raw_case']['Case'].get('Diagnosis By'),
                        'reference_status':'structured_source_not_confirmed_status_gold',
                        'note':'Single-case structured adapter; case linkage easier than full multi-case literature.'})
            for f in figs:f.update(benchmark_dataset=doc['dataset'],sampling_strata=doc['strata'])
            request={'article_id':uid,'passages':passages,'candidate_figures':[
                {'figure_id':f['figure_id'],'caption':f['caption'],'caption_source_offset':f['caption_evidence']['offset']} for f in figs]}
            folder.mkdir(parents=True,exist_ok=True)
            template=(ROOT/'configs/linkage_prompt.txt').read_text(encoding='utf-8')
            prompt=template.replace('{{SCHEMA}}',(ROOT/'configs/linkage_schema.json').read_text(encoding='utf-8')).replace('{{ARTICLE_INPUT}}',json.dumps(request,ensure_ascii=False,indent=2))
            if (folder/'input.json').exists() and load(folder/'input.json')!=request: raise ValueError('Existing prepared source changed')
            save(folder/'input.json',request);(folder/'prompt.txt').write_text(prompt,encoding='utf-8')
            jobs.append(uid);all_figures.extend(figs)
            statuses.append({'document_id':uid,'dataset':doc['dataset'],'status':'prepared','figures':len(figs)})
        except (OSError,ValueError,RuntimeError,KeyError) as exc:
            statuses.append({'document_id':uid,'dataset':doc['dataset'],'status':'preparation_failed','error':str(exc)})
    write_rows(out/'figures.jsonl',all_figures);write_rows(out/'silver_references.jsonl',references)
    save(out/'jobs.json',jobs);save(out/'preparation_status.json',statuses)
    gold=out/'gold_review.csv'
    previous={r['sample_id']:r for r in csv.DictReader(gold.open(encoding='utf-8-sig'))} if gold.exists() else {}
    fields=['sample_id','verified','brain_mri','case_ids','caption_aliases_json','diagnosis_aliases_json','diagnosis_status','evidence','reviewer']
    with gold.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for f in all_figures:w.writerow(previous.get(f['sample_id'],{'sample_id':f['sample_id'],'verified':'false'}))
    from html import escape
    refmap={r['sample_id']:r for r in references}
    cards=[]
    for f in all_figures:
        cards.append('<article><h2>'+escape(f['sample_id'])+'</h2><h3>原图注</h3><pre>'+escape(f['caption'])+'</pre><h3>来源参考（尚未独立审核）</h3><pre>'+escape(json.dumps(refmap.get(f['sample_id']),ensure_ascii=False,indent=2))+'</pre><h3>原文输入</h3><pre>'+escape(json.dumps(load(work/'data/model_inputs'/f['article_id']/'input.json')['passages'],ensure_ascii=False,indent=2))+'</pre></article>')
    (out/'reference_review.html').write_text('<!doctype html><meta charset="utf-8"><title>三病参考审核</title><style>body{font-family:Arial;margin:30px}pre{white-space:pre-wrap}article{border-bottom:2px solid #aaa}</style><h1>参考审核（不显示预测）</h1><p>先核验来源并填写 gold_review.csv，来源标签不自动作为真值。图片需按来源另行查看，MedPix 本步骤仅准备文本，不声称核验图片像素。</p>'+''.join(cards),encoding='utf-8')
    fingerprints={n:sha(ROOT/'scripts'/n) for n in ['collect.py','inventory_dataset.py','preview_dataset.py','run_deepseek.py','link_cases.py','extraction_format.py']}
    save(out/'prepared_manifest.json',{'figures_sha256':sha(out/'figures.jsonl'),'jobs_sha256':sha(out/'jobs.json'),
        'frozen_script_hashes':fingerprints,'selection_identity':manifest['identity'],
        'input_sha256':{j:sha(work/'data/model_inputs'/j/'input.json') for j in jobs},
        'prompt_sha256':{j:sha(work/'data/model_inputs'/j/'prompt.txt') for j in jobs},
        'note':'No change to existing rules or prompts. MedPix input adapter differs from full PMC acquisition. No gold fields supplied to model.'})
    save(out/'preparation_summary.json',{'selected_documents':len(selection),'prepared_documents':len(jobs),'figures':len(all_figures),
        'statuses':dict(Counter(s['status'] for s in statuses)),'new_api_calls':0,'verified_gold_rows':0})
    print(json.dumps(load(out/'preparation_summary.json'),indent=2))


def check_frozen(out):
    m=load(out/'prepared_manifest.json')
    if sha(out/'figures.jsonl')!=m['figures_sha256'] or sha(out/'jobs.json')!=m['jobs_sha256']:raise ValueError('Frozen manifests changed')
    for n,h in m['frozen_script_hashes'].items():
        if sha(ROOT/'scripts'/n)!=h:raise ValueError('Pipeline changed since preparation: '+n)
    for j in load(out/'jobs.json'):
        if sha(out/'work/data/model_inputs'/j/'input.json')!=m['input_sha256'][j] or sha(out/'work/data/model_inputs'/j/'prompt.txt')!=m['prompt_sha256'][j]:raise ValueError('Prepared input changed')


def infer(args):
    out=BASE/'v1';check_frozen(out);work=out/'work'
    run_deepseek.ROOT=work
    jobs=load(out/'jobs.json')[:args.limit]
    if not args.allow_api:
        for j in jobs:run_deepseek.call_one(j,'',8192,dry_run=True)
        print('Dry run only. Add --allow-api in your terminal for paid extraction.');return
    key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key:raise ValueError('Configure DEEPSEEK_API_KEY locally in this terminal')
    for j in jobs:
        folder=work/'data/model_inputs'/j
        payload=run_deepseek.build_request(folder,8192)
        h=hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode('utf-8')).hexdigest()
        receipt_path=work/'data/model_outputs'/run_deepseek.MODEL/j/h[:16]/'receipt.json'
        if receipt_path.exists():
            receipt=load(receipt_path)
            if receipt.get('status')=='validation_failed':
                print(f'Skipped previous validation failure, no retry or charge: {j}',flush=True)
                continue
            if not receipt.get('structural_and_quote_checks_passed'):
                print(f'Stopped on previous incomplete/account/network failure; inspect receipt: {j}',flush=True)
                break
        ok=run_deepseek.call_one(j,key,8192)
        if not ok:
            receipt=load(receipt_path) if receipt_path.exists() else {}
            if receipt.get('status')=='validation_failed':
                print(f'Recorded invalid output; continuing without altering or retrying it: {j}',flush=True)
                continue
            break
    collect_predictions(out)


def collect_predictions(out):
    predictions=[];documents={}
    for f in rows(out/'figures.jsonl'):
        uid=f['article_id'];folder=out/'work/data/model_inputs'/uid
        payload=run_deepseek.build_request(folder,8192)
        request_hash=hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode('utf-8')).hexdigest()
        result=out/'work/data/model_outputs'/run_deepseek.MODEL/uid/request_hash[:16]
        link=None;status='not_run'
        if (result/'api_response.json').exists():
            response,errors,_=run_deepseek.check_completion(load(result/'api_response.json'),load(folder/'input.json'))
            status='invalid_output' if errors else 'validated_output_not_semantic_gold'
            if not errors:link=next((r for r in response['figures'] if r['figure_id']==f['figure_id']),None)
        receipt=load(result/'receipt.json') if (result/'receipt.json').exists() else {}
        documents[uid]={'document_id':uid,'dataset':f['benchmark_dataset'],
            'status':receipt.get('status','not_run'),'errors':receipt.get('errors',[]),
            'usage':receipt.get('usage',{})}
        # Match original collection MRI cue; downstream text exclusions remain frozen.
        selection=bool(collect.caption_signals(f['caption'])['mri']) and not ANIMAL.search(f['caption']) and not NONPATIENT.search(f['caption'])
        predictions.append({'sample_id':f['sample_id'],'dataset':f['benchmark_dataset'],'strata':f['sampling_strata'],
            'brain_mri_selected':bool(selection),'caption':f['caption'],'link':link,'inference_status':status})
    write_rows(out/'predictions.jsonl',predictions)
    save(out/'inference_summary.json',{'documents':len(documents),
        'document_status_counts':dict(Counter(r['status'] for r in documents.values())),
        'figure_status_counts':dict(Counter(r['inference_status'] for r in predictions)),
        'articles':list(documents.values()),'note':'Validation failure is schema-contract failure, not automatically a wrong clinical diagnosis. No failed output is silently repaired.'})
    mapping=out/'case_label_map.csv'
    previous={r['sample_id']:r for r in csv.DictReader(mapping.open(encoding='utf-8-sig'))} if mapping.exists() else {}
    with mapping.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=['sample_id','predicted_case_label','mapped_case_id','reviewer']);w.writeheader()
        for p in predictions:
            if p['link'] and p['link'].get('case_label'):
                old=previous.get(p['sample_id'],{})
                if old.get('predicted_case_label')==p['link']['case_label']:w.writerow(old)
                else:w.writerow({'sample_id':p['sample_id'],'predicted_case_label':p['link']['case_label']})
    return predictions


def relation_metrics(tp,fp,fn):
    return {'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def score(args):
    out=BASE/'v1';check_frozen(out)
    pred=collect_predictions(out)
    gold={r['sample_id']:r for r in csv.DictReader((out/'gold_review.csv').open(encoding='utf-8-sig')) if r['verified'].lower()=='true'}
    maps={r['sample_id']:r for r in csv.DictReader((out/'case_label_map.csv').open(encoding='utf-8-sig')) if r['mapped_case_id'].strip() and r['reviewer'].strip()}
    details=[]
    for p in pred:
        g=gold.get(p['sample_id'])
        if not g:continue
        if g['brain_mri'] not in {'0','1'} or not g['reviewer'].strip() or not g['evidence'].strip():raise ValueError('Verified reference requires brain_mri 0/1, reviewer and evidence')
        positive=g['brain_mri']=='1';selected=p['brain_mri_selected'];link=p['link'] or {}
        expected_cases={x.strip() for x in g['case_ids'].split(';') if x.strip()}
        # Natural-language case names need identity mapping; never score spelling
        # differences as wrong patient identities or copy expected IDs as predictions.
        assigned=maps.get(p['sample_id'],{}).get('mapped_case_id')
        case_ok=bool(assigned and assigned in expected_cases)
        caption_aliases=json.loads(g['caption_aliases_json'] or '[]')
        diagnosis_aliases=json.loads(g['diagnosis_aliases_json'] or '[]')
        caption_ok=normalize(p['caption']) in {normalize(x) for x in caption_aliases}
        diagnosis_ok=normalize(link.get('diagnosis_name') or '') in {normalize(x) for x in diagnosis_aliases} and link.get('diagnosis_status')==g['diagnosis_status']
        details.append({'sample_id':p['sample_id'],'dataset':p['dataset'],'strata':p['strata'],
            'inference_attempted':p['inference_status']!='not_run',
            'gold_positive':positive,'selected':selected,'link_pending':positive and bool(expected_cases) and bool(link.get('case_label')) and not assigned,
            'case_eligible':positive and bool(expected_cases),'case_emitted':selected and link.get('link_status') in {'explicit','contextual'} and bool(link.get('case_label')),
            'case_correct':case_ok,'caption_eligible':positive and bool(caption_aliases),'caption_emitted':selected,
            'caption_correct':caption_ok,'diagnosis_eligible':positive and bool(diagnosis_aliases) and bool(g['diagnosis_status']),
            'diagnosis_emitted':selected and bool(link.get('diagnosis_name')),'diagnosis_correct':diagnosis_ok,
            'diagnosis_gold_status':g['diagnosis_status'],'diagnosis_pred_status':link.get('diagnosis_status') or 'abstain'})
    reports={}
    for dataset in ['MultiCaRe','MedPix2']:
        reports[dataset]={}
        for stratum in ['ALL',*load(CONFIG)['diseases']]:
            subset=[r for r in details if r['dataset']==dataset and (stratum=='ALL' or stratum in r['strata'])]
            if not subset:
                reports[dataset][stratum]={'status':'no_verified_ground_truth','n':0};continue
            entry={'n':len(subset),'brain_mri_selection':metrics([r['gold_positive'] for r in subset],[r['selected'] for r in subset])}
            for task in ['case','caption','diagnosis']:
                applicable=[r for r in subset if r[task+'_eligible']]
                if task in {'case','diagnosis'} and any(not r['inference_attempted'] for r in applicable):
                    entry[task]={'status':'pending_model_inference','n':len(applicable)};continue
                if task=='case' and any(r['link_pending'] for r in applicable):
                    entry[task]={'status':'pending_case_identity_mapping','n':len(applicable)};continue
                tp=sum(bool(r[task+'_emitted'] and r[task+'_correct']) for r in applicable)
                fp=sum(bool(r[task+'_emitted'] and not r[task+'_correct']) for r in applicable)
                fn=len(applicable)-tp
                entry[task]={'status':'evaluated' if applicable else 'missing_reference_for_task','n':len(applicable),**relation_metrics(tp,fp,fn)}
            dx=[r for r in subset if r['diagnosis_eligible']]
            if dx and all(r['inference_attempted'] for r in dx):
                per_status={}
                for label in sorted({r['diagnosis_gold_status'] for r in dx}):
                    tp=sum(r['diagnosis_gold_status']==label and r['diagnosis_pred_status']==label for r in dx)
                    fp=sum(r['diagnosis_gold_status']!=label and r['diagnosis_pred_status']==label for r in dx)
                    fn=sum(r['diagnosis_gold_status']==label and r['diagnosis_pred_status']!=label for r in dx)
                    per_status[label]=relation_metrics(tp,fp,fn)
                entry['diagnosis_status']={'per_class':per_status,'macro_f1':sum(v['f1'] or 0 for v in per_status.values())/len(per_status)}
            reports[dataset][stratum]=entry
    report={'verified_figures':len(details),'all_prepared_figures':len(pred),'datasets':reports,
        'note':'Case/caption/diagnosis metrics are conditional on verified target MRI with reference for task; not full-universe endpoint precision. Caption metric is strict normalized full-caption match, not semantic or panel alignment. Strata may overlap.',
        'gold_sha256':sha(out/'gold_review.csv'),'predictions_sha256':sha(out/'predictions.jsonl'),'new_api_calls':0}
    save(out/'scores.json',report);write_rows(out/'scored_details.jsonl',details)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>三病四任务评测</title><h1>三病四任务评测</h1><p>未核验的公开参考不计成绩；空缺不是零分。此页不会触发 API。</p><pre style="white-space:pre-wrap">'+__import__('html').escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>',encoding='utf-8')
    print(json.dumps({'verified_figures':len(details),'all_prepared_figures':len(pred),'report':str(out/'report.html')},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('coverage')
    p=sub.add_parser('prepare');p.add_argument('--cached-only',action='store_true')
    p=sub.add_parser('infer');p.add_argument('--allow-api',action='store_true');p.add_argument('--limit',type=int,default=1)
    sub.add_parser('score')
    sub.add_parser('status')
    args=ap.parse_args()
    if hasattr(args,'limit') and not 1<=args.limit<=100:ap.error('limit must be 1..100')
    def status(_):
        check_frozen(BASE/'v1');collect_predictions(BASE/'v1')
        r=load(BASE/'v1/inference_summary.json')
        print(json.dumps({k:v for k,v in r.items() if k!='articles'},indent=2))
    {'coverage':coverage,'prepare':prepare,'infer':infer,'score':score,'status':status}[args.command](args)


if __name__=='__main__':main()
