"""Full-source GBM/astrocytoma/CADASIL extraction tests and auditable matrices.

No sampling caps. Paid requests require --allow-api. MRI scoring is caption-rule
selection against MultiCaRe HEAD MRI source labels, not pixel brain recognition.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import re
from urllib.request import urlopen
import global_three_disease_eval as engine
import collect
from four_disease_eval import native_medpix
from dedup_leakage import rows, write_rows, save
from medcpt_local import ROOT, sha
from evaluate_curation import predict, metrics
from task_benchmark_suite import text_scores

OUT=ROOT/'data/benchmarks/gbm_astro_cadasil_v1'
REPORT=ROOT/'reports/gbm_astro_cadasil_v1'
TARGETS={'GBM':r'\bglioblastoma\b|\bGBM\b',
         'Astrocytoma':r'\bastrocytomas?\b',
         'CADASIL':r'\bCADASIL\b|cerebral autosomal dominant arteriopathy'}
DIAG_LABELS=['GBM','Astrocytoma','CADASIL','Other','Unresolved']
ORIGINAL_FINGERPRINT=engine.fingerprint

def mentions(text):return sorted(k for k,p in TARGETS.items() if re.search(p,text,re.I))
def diagnosis_class(text):
    if not text or not text.strip() or re.search(r'\b(?:unknown|probably|probable|suspected|possible)\b',text,re.I):return 'Unresolved'
    # GBM wins over the historical umbrella "grade IV astrocytoma" whenever
    # the source explicitly also says glioblastoma. Do not reclassify by grade.
    for k in ['GBM','Astrocytoma','CADASIL']:
        if re.search(TARGETS[k],text,re.I):return k
    return 'Other'

def identity():
    return {**ORIGINAL_FINGERPRINT(),'task_driver_sha256':sha(Path(__file__)),
        'subtype_policy':'all explicitly named astrocytoma subtypes; retain original text; probable diagnoses excluded from gold',
        'matrix_policy':'case/description candidate edges within original document; diagnosis one-vs-rest plus multiclass'}

engine.OUT=OUT;engine.DISEASES=TARGETS;engine.mentions=mentions;engine.fingerprint=identity

def load(path):return json.loads(path.read_text(encoding='utf-8'))

def matrix(y,p):
    return metrics(y,p) if y else {'n':0,'positive':0,'negative':0,'TP':0,'FP':0,'FN':0,'TN':0,
        'precision':None,'recall':None,'specificity':None,'f1':None,'accuracy':None}

def bin_record(task,dataset,group,y,p,unit,note,**extra):
    return {'task':task,'dataset':dataset,'group':group,'labels':['Negative','Positive'],
        'matrix':[[sum(not a and not b for a,b in zip(y,p)),sum(not a and b for a,b in zip(y,p))],
                  [sum(a and not b for a,b in zip(y,p)),sum(a and b for a,b in zip(y,p))]],
        'metrics':matrix(y,p),'unit':unit,'note':note,**extra}

def mri_test():
    """Entire classification CSV; no disease keyword pre-filter for MRI test."""
    file=engine.SRC/'multicare_captions_and_labels.csv'
    if not file.exists():raise ValueError('Missing official MultiCaRe captions_and_labels.csv')
    rec=load(engine.SRC/'multicare_record.json')
    expected=next(r for r in rec['files'] if r['key']=='captions_and_labels.csv')
    if file.stat().st_size!=expected['size']:raise ValueError('Incomplete source label CSV')
    if expected.get('checksum','').startswith('md5:') and hashlib.md5(file.read_bytes()).hexdigest()!=expected['checksum'].split(':')[1]:raise ValueError('Source label checksum mismatch')
    data=[]
    with file.open(encoding='utf-8',newline='') as h:
        for r in csv.DictReader(h):
            gold=r['image_subtype']=='mri' and r['radiology_region']=='head'
            pred=predict(r['caption'] or '')['actual_mri_caption_gate']
            data.append({'sample_id':r['file_id'],'case_id':r['patient_id'],'gold':gold,'prediction':pred,
                'modality':r['image_subtype'],'region':r['radiology_region'],'caption':r['caption']})
    write_rows(OUT/'mri_predictions.jsonl',data)
    definitions=[('ALL',data),('MRI_only',[r for r in data if r['modality']=='mri'])]
    result=[bin_record('mri_selection','MultiCaRe',g,[r['gold'] for r in s],[r['prediction'] for r in s],
        'source image/subimage',
        'Existing caption MRI gate vs source mri+head labels. HEAD is broader than brain. Source classifications are not independent expert gold. MRI_only isolates anatomy errors.',
        source_sha256=sha(file),unique_cases=len({r['case_id'] for r in s})) for g,s in definitions]
    save(OUT/'mri_matrices.json',result)
    return result

def resolve_xml(uid,fetch):
    candidates=[ROOT/'data/raw'/uid/'article.bioc.xml',
        ROOT/'data/benchmarks/four_disease/v1/work/data/raw'/uid/'article.bioc.xml',
        OUT/'source_text'/uid/'article.bioc.xml']
    for path in candidates:
        if path.exists():return path
    if not fetch:return None
    path=candidates[-1];path.parent.mkdir(parents=True,exist_ok=True)
    url='https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/'+uid+'/unicode'
    with urlopen(url,timeout=45) as response:raw=response.read()
    # Parse before atomically keeping a cache; error/challenge pages not accepted.
    tmp=path.with_suffix('.tmp');tmp.write_bytes(raw)
    collect.parse_bioc(tmp,uid);tmp.replace(path)
    save(path.with_suffix('.source.json'),{'url':url,'sha256':sha(path)})
    return path

def descriptions(fetch):
    refs=rows(OUT/'references.jsonl');catalog=rows(OUT/'catalog.jsonl')
    bydoc=defaultdict(list)
    for r in refs:
        if r['dataset']=='MultiCaRe':bydoc[r['source_document']].append(r)
    output=[];held=[]
    for index,(uid,group) in enumerate(bydoc.items(),1):
        try:
            xml=resolve_xml(uid,fetch)
            if xml is None:
                held.append({'document':uid,'reason':'BioC_not_cached_use_fetch_text','figures':len(group)});continue
            passages=collect.parse_bioc(xml,uid);captions=defaultdict(list)
            for p in passages:
                m=p['metadata']
                if m.get('section_type')=='FIG' and m.get('type') in {'fig_title_caption','fig_caption'}:captions[m.get('id')].append(p)
            extracted={fid:next((p for p in pp if p['metadata'].get('type')=='fig_title_caption'),pp[0])['text'] for fid,pp in captions.items()}
            for r in group:
                output.append({**r,'extracted_caption':extracted.get(r['figure_id']),
                    'status':'parsed','source_sha256':sha(xml),'reference_role':'independent publication extraction vs MultiCaRe source caption'})
        except (OSError,ValueError,KeyError,StopIteration) as exc:
            held.append({'document':uid,'reason':type(exc).__name__+': '+str(exc)[:200],'figures':len(group)})
        if index%25==0:print(f'Original descriptions {index}/{len(bydoc)} documents',flush=True)
    # Exercise the existing native source adapter. Reference and input share the
    # structured source: explicitly an import-integrity check, not semantic test.
    native={}
    for doc in catalog:
        if doc['dataset']=='MedPix2':
            _,figs=native_medpix(doc)
            for f in figs:native[f['figure_id']]=f['caption']
    for r in refs:
        if r['dataset']!='MedPix2':continue
        image=r['source_image_ids'][0]
        output.append({**r,'extracted_caption':native.get(image),'status':'parsed',
            'reference_role':'structured source adapter integrity only'})
    write_rows(OUT/'description_predictions.jsonl',output);write_rows(OUT/'description_held.jsonl',held)
    print(json.dumps({'description_records':len(output),'held_documents':len(held),'fetch_text':fetch},indent=2),flush=True)

def edge_matrix(data,task,dataset,group,caption=False):
    y=[];p=[];samples=[]
    pools=defaultdict(set)
    if caption:
        for r in data:pools[r['source_document']].add(engine.norm(r['source_caption']))
    for r in data:
        if caption:
            gold={engine.norm(r['source_caption'])};pred={engine.norm(r['extracted_caption'])} if r.get('extracted_caption') is not None else set()
            universe=pools[r['source_document']]|pred
        else:
            gold=set(r['source_case_ids']);pred=set(r['predicted_case_ids']);universe=set(r['case_identities'].values())|pred
        for candidate in sorted(universe):y.append(candidate in gold);p.append(candidate in pred)
        samples.append(r)
    note=('Exact normalized image-caption association recovery; competing captions are in same document. MedPix is structured adapter integrity, not independent semantic pairing.'
          if caption else 'All candidate cases in input document; multiple-case abstention remains a miss of source edges, not a wrong clinical diagnosis. Validated outputs only.')
    return bin_record(task,dataset,group,y,p,'image-description edge' if caption else 'image-case edge',note,
        figures=len(samples),cases=len({c for r in samples for c in r['source_case_ids']}))

def group_records(data,dataset,group):
    subset=[r for r in data if r['dataset']==dataset]
    if group=='ALL':return subset
    if dataset=='MedPix2':return [r for r in subset if diagnosis_class(r.get('reference_diagnosis'))==group]
    return [r for r in subset if group in r['mention_strata']]

def score():
    engine.score()
    REPORT.mkdir(parents=True,exist_ok=True)
    records=rows(OUT/'prediction_details.jsonl');valid=[r for r in records if r['inference_status']=='validated']
    matrices=load(OUT/'mri_matrices.json') if (OUT/'mri_matrices.json').exists() else mri_test()
    statuses=Counter(r['inference_status'] for r in records)
    for ds in ['MedPix2','MultiCaRe']:
        for g in ['ALL']+list(TARGETS):
            s=group_records(valid,ds,g)
            matrices.append(edge_matrix(s,'case_association',ds,g))
    descriptions_data=rows(OUT/'description_predictions.jsonl') if (OUT/'description_predictions.jsonl').exists() else []
    caption_scores=[]
    for ds in ['MedPix2','MultiCaRe']:
        for g in ['ALL']+list(TARGETS):
            s=group_records(descriptions_data,ds,g)
            matrices.append(edge_matrix(s,'original_description',ds,g,caption=True))
            scores=[text_scores(r.get('extracted_caption') or '',r['source_caption']) for r in s]
            caption_scores.append({'dataset':ds,'group':g,'figures':len(s),
                'scores':{k:sum(float(r[k]) for r in scores)/len(scores) for k in scores[0]} if scores else None,
                'scope':'adapter_integrity_only' if ds=='MedPix2' else 'publication_text_extraction_vs_source_caption'})
    diagnosis=[r for r in valid if r['dataset']=='MedPix2' and diagnosis_class(r.get('reference_diagnosis'))!='Unresolved']
    gold=[diagnosis_class(r['reference_diagnosis']) for r in diagnosis]
    predicted=[diagnosis_class((r.get('prediction') or {}).get('diagnosis_name')) for r in diagnosis]
    cm=[[sum(a==x and b==z for a,b in zip(gold,predicted)) for z in DIAG_LABELS] for x in DIAG_LABELS]
    matrices.append({'task':'diagnosis_multiclass','dataset':'MedPix2','group':'ALL','labels':DIAG_LABELS,
        'matrix':cm,'metrics':{'n':len(gold),'accuracy':sum(a==b for a,b in zip(gold,predicted))/len(gold) if gold else None},
        'unit':'image-linked case diagnosis','note':'All astrocytoma subtypes included; explicit GBM wins; probable/unknown source diagnoses excluded. Source diagnosis naming retained, not retrospectively WHO reclassified.',
        'cases':len({i for r in diagnosis for i in r['source_case_ids']})})
    for g in TARGETS:
        matrices.append(bin_record('diagnosis_one_vs_rest','MedPix2',g,[v==g for v in gold],[v==g for v in predicted],
            'image-linked diagnosis','No positive references means sensitivity/F1 not estimable; Unresolved outputs are not positives.',
            figures=len(gold),positive_cases=len({i for r in diagnosis if diagnosis_class(r['reference_diagnosis'])==g for i in r['source_case_ids']})))
    save(REPORT/'matrices.json',matrices)
    flat=[]
    for r in matrices:
        flat.append({'task':r['task'],'dataset':r['dataset'],'group':r['group'],'figures':r.get('figures'),
            'cases':r.get('cases'),'unit':r['unit'],**r['metrics'],'note':r['note']})
    keys=['task','dataset','group','figures','cases','unit','n','positive','negative','TP','FP','FN','TN','precision','recall','specificity','f1','accuracy','note']
    with (REPORT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(flat)
    save(REPORT/'description_text_scores.json',caption_scores)
    coverage={'planned_figures':len(records),'inference_status':dict(statuses),
        'validated_figures':len(valid),'description_parsed_figures':len(descriptions_data),
        'description_held_documents':len(rows(OUT/'description_held.jsonl')) if (OUT/'description_held.jsonl').exists() else None,
        'classification_policy':'all astrocytoma subtypes; three target classes plus Other and Unresolved',
        'source_manifest_sha256':sha(OUT/'prepared_manifest.json'),
        'reference_sha256':sha(OUT/'references.jsonl'),
        'warning':'MultiCaRe MRI target is HEAD MRI, not exact brain MRI; MultiCaRe disease strata are mentions, not verified diagnoses. MedPix description is adapter integrity. No new image model or raw image interpretation.'}
    save(REPORT/'coverage.json',coverage)
    head='<!doctype html><meta charset="utf-8"><title>GBM / Astrocytoma / CADASIL tests</title><style>body{font:16px/1.7 Arial,"Microsoft YaHei";margin:32px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}img{max-width:100%}</style>'
    body='<h1>胶质母细胞瘤／星形细胞瘤／CADASIL：分任务检测</h1><p>主评测包含明确报告的所有星形细胞瘤亚型，不将“疑似”作为确诊参考。仅使用MedPix与MultiCaRe。</p><p>病例叙述任务具体评估图片—病例归属，不声称覆盖全部临床字段抽取。逐图描述评估原文恢复，不测试看图生成。</p><p>MultiCaRe仅有head标签，因此此处测“头部MRI来源标签筛选”；不能称精确脑MRI识别。保留原有图注MRI规则，不传入来源标签作为预测输入。</p><p><a href="plots.html">混淆矩阵图</a> · <a href="metrics.csv">指标表</a> · <a href="description_text_scores.json">描述文字指标</a></p><h2>覆盖与未完成情况</h2><pre>'+escape(json.dumps(coverage,ensure_ascii=False,indent=2))+'</pre><h2>任务指标</h2><table><tr><th>任务</th><th>数据集</th><th>分层</th><th>指标</th></tr>'
    for r in flat:body+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [r['task'],r['dataset'],r['group'],json.dumps({k:r[k] for k in ['n','precision','recall','f1','accuracy'] if k in r})])+'</tr>'
    (REPORT/'report.html').write_text(head+body+'</table>',encoding='utf-8')
    print(json.dumps({'report':str(REPORT/'report.html'),'planned':len(records),'validated':len(valid),'matrices':len(matrices)},indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['scan','prepare','infer','score','run','descriptions','mri'])
    ap.add_argument('--allow-api',action='store_true');ap.add_argument('--fetch-text',action='store_true')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.command=='scan':engine.scan()
    elif a.command=='prepare':engine.prepare();mri_test();descriptions(a.fetch_text);score()
    elif a.command=='mri':mri_test()
    elif a.command=='descriptions':descriptions(a.fetch_text);score()
    elif a.command=='infer':engine.infer(a.allow_api);score()
    elif a.command=='score':score()
    else:
        if not (OUT/'prepared_manifest.json').exists():engine.prepare()
        if not (OUT/'mri_matrices.json').exists():mri_test()
        descriptions(a.fetch_text)
        engine.infer(a.allow_api);score()
    if (REPORT/'matrices.json').exists():
        from plot_requested_task_matrices import main as plot
        plot()

if __name__=='__main__':main()
