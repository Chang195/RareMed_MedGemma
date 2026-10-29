"""Source-cluster bootstrap of the existing frozen extraction evaluation.

Uses existing predictions only; no API calls and no new manual review.
"""
import json, re, csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/benchmarks/gbm_astro_cadasil_v1'
OUT=ROOT/'reports/pipeline_metrics_ci_v1'
SEED=20261027
B=5000

def rows(name):
    return [json.loads(x) for x in (DATA/name).read_text(encoding='utf-8').split('\n') if x.strip()]

def norm(s): return ' '.join(re.findall(r'\w+',s.lower()))

def diag(s):
    if not s or re.search(r'\b(?:unknown|probably|probable|suspected|possible)\b',s,re.I): return 'Unresolved'
    for k,p in [('GBM',r'\bglioblastoma\b|\bGBM\b'),('Astrocytoma',r'\bastrocytomas?\b'),('CADASIL',r'\bCADASIL\b|cerebral autosomal dominant arteriopathy')]:
        if re.search(p,s,re.I):return k
    return 'Other'

def scores(a):
    tp,fp,fn,tn=np.moveaxis(np.asarray(a,dtype=float),-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.stack([tp/(tp+fp),tp/(tp+fn),2*tp/(2*tp+fp+fn)],axis=-1)

def summarize(task,ds,groups,n,unit,source,details):
    vals=np.array(list(groups.values()),dtype=int)
    patterns,counts=np.unique(vals,axis=0,return_counts=True)
    rng=np.random.default_rng(SEED)
    # Resample independent source clusters. Combining equal confusion vectors
    # is exactly equivalent to resampling each source cluster individually.
    boot=rng.multinomial(len(vals),counts/counts.sum(),size=B) @ patterns
    est=scores(vals.sum(axis=0)); ci=np.nanpercentile(scores(boot),[2.5,97.5],axis=0)
    return dict(task=task,dataset=ds,n=n,unit=unit,clusters=len(groups),reference=source,
        details=details,new_manual_review=0,confusion=vals.sum(axis=0).tolist(),
        metrics={k:dict(value=float(est[i]),low=float(ci[0,i]),high=float(ci[1,i])) for i,k in enumerate(['Precision','Recall','F1'])})

def main():
    result=[];g=defaultdict(lambda:np.zeros(4,dtype=int));mri=rows('mri_predictions.jsonl')
    for r in mri:
        key=r['case_id'].split('_')[0];y=r['gold'];p=r['prediction']
        g[key][0 if y and p else 1 if p else 2 if y else 3]+=1
    result.append(summarize('MRI selection','MultiCaRe',g,len(mri),'image/subimage',
        'Source image_subtype=mri AND radiology_region=head','HEAD MRI, broader than brain MRI. Bootstrap by source article.'))
    records=[r for r in rows('prediction_details.jsonl') if r['inference_status']=='validated']
    for ds in ['MultiCaRe','MedPix2']:
        rr=[r for r in records if r['dataset']==ds];g=defaultdict(lambda:np.zeros(4,dtype=int))
        for r in rr:
            key=r['source_document'] if ds=='MultiCaRe' else '|'.join(sorted(r['source_case_ids']))
            gold=set(r['source_case_ids']);pred=set(r['predicted_case_ids']);universe=set(r['case_identities'].values())|pred
            g[key]+=np.array([len(gold&pred),len(pred-gold),len(gold-pred),len(universe-(gold|pred))])
        result.append(summarize('Case association',ds,g,len(rr),'figures',
            'Source image-case identifiers','Image-case edges. Validated outputs only. Bootstrap by source article (MultiCaRe) or true source case (MedPix).'))
    for ds in ['MultiCaRe','MedPix2']:
        rr=[r for r in rows('description_predictions.jsonl') if r['dataset']==ds]
        pools=defaultdict(set)
        for r in rr:pools[r['source_document']].add(norm(r['source_caption']))
        g=defaultdict(lambda:np.zeros(4,dtype=int))
        for r in rr:
            key=r['source_document'] if ds=='MultiCaRe' else '|'.join(sorted(r['source_case_ids']))
            gold={norm(r['source_caption'])};pred={norm(r['extracted_caption'])} if r.get('extracted_caption') is not None else set()
            universe=pools[r['source_document']]|pred
            g[key]+=np.array([len(gold&pred),len(pred-gold),len(gold-pred),len(universe-(gold|pred))])
        result.append(summarize('Original description',ds,g,len(rr),'figures','Source per-image caption',
            'Normalized exact image-caption association recovery. MedPix is structured import integrity only.'))
    rr=[r for r in records if r['dataset']=='MedPix2' and diag(r.get('reference_diagnosis'))!='Unresolved']
    g=defaultdict(lambda:np.zeros(4,dtype=int))
    for r in rr:
        key='|'.join(sorted(r['source_case_ids']))
        correct=diag(r['reference_diagnosis'])==diag((r.get('prediction') or {}).get('diagnosis_name'))
        g[key]+=np.array([int(correct),int(not correct),int(not correct),0])
    result.append(summarize('Diagnosis extraction','MedPix2',g,len(rr),'figures','Source Case Diagnosis field',
        'Multiclass micro P/R/F1 including Other and Unresolved. All equal accuracy. CADASIL positive n=0. Bootstrap by case.'))
    OUT.mkdir(parents=True,exist_ok=True)
    payload=dict(bootstrap_replicates=B,seed=SEED,confidence_level=.95,method='percentile nonparametric source-cluster bootstrap',
        limitations=['No new manual review. Source labels are not independent expert adjudication.',
        'Zero observed errors can yield a degenerate empirical CI; this is not proof of zero population error.',
        'Intervals condition on the existing source sample and successful outputs, not population generalization or label correctness.',
        'Diagnosis is source-text extraction, not image diagnosis. No CADASIL-positive diagnosis reference.'],rows=result)
    (OUT/'statistics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task','dataset','n','unit','clusters','metric','estimate','ci_low','ci_high','new_manual_review','reference']);w.writeheader()
        for r in result:
            for k,m in r['metrics'].items():w.writerow({**{t:r[t] for t in ['task','dataset','n','unit','clusters','new_manual_review','reference']},'metric':k,'estimate':m['value'],'ci_low':m['low'],'ci_high':m['high']})
    print(json.dumps(payload,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
