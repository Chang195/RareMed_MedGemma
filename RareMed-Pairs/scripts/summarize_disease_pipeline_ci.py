"""Disease-specific source-cluster bootstrap, retaining frozen task definitions."""
import json,csv
from collections import defaultdict
import numpy as np
from summarize_pipeline_ci import ROOT,DATA,rows,norm,diag,summarize,B,SEED

OUT=ROOT/'reports/disease_pipeline_ci_v1'

def subset(rr,ds,disease):
    return [r for r in rr if r['dataset']==ds and
        (diag(r.get('reference_diagnosis'))==disease if ds=='MedPix2' else disease in r['mention_strata'])]

def empty(task,ds,n=0,positive=0):
    return dict(task=task,dataset=ds,n=n,positive=positive,clusters=0,new_manual_review=0,metrics=None,reason='No positive source references' if n else 'No eligible source records')

def main():
    valid=[r for r in rows('prediction_details.jsonl') if r['inference_status']=='validated']
    desc=rows('description_predictions.jsonl');out=[]
    for disease in ['GBM','Astrocytoma','CADASIL']:
        result=[]
        for ds in ['MultiCaRe','MedPix2']:
            rr=subset(valid,ds,disease);g=defaultdict(lambda:np.zeros(4,dtype=int))
            for r in rr:
                key=r['source_document'] if ds=='MultiCaRe' else '|'.join(sorted(r['source_case_ids']))
                gold=set(r['source_case_ids']);pred=set(r['predicted_case_ids']);universe=set(r['case_identities'].values())|pred
                g[key]+=np.array([len(gold&pred),len(pred-gold),len(gold-pred),len(universe-(gold|pred))])
            result.append(summarize('Case association',ds,g,len(rr),'figures','Source image-case identifiers','Same frozen edge scoring; disease strata as original report.') if rr else empty('Case association',ds))
        for ds in ['MultiCaRe','MedPix2']:
            rr=subset(desc,ds,disease);g=defaultdict(lambda:np.zeros(4,dtype=int));pools=defaultdict(set)
            for r in rr:pools[r['source_document']].add(norm(r['source_caption']))
            for r in rr:
                key=r['source_document'] if ds=='MultiCaRe' else '|'.join(sorted(r['source_case_ids']))
                gold={norm(r['source_caption'])};pred={norm(r['extracted_caption'])} if r.get('extracted_caption') is not None else set()
                universe=pools[r['source_document']]|pred
                g[key]+=np.array([len(gold&pred),len(pred-gold),len(gold-pred),len(universe-(gold|pred))])
            result.append(summarize('Original description',ds,g,len(rr),'figures','Source per-image caption','Normalized exact caption association. MedPix import integrity only.') if rr else empty('Original description',ds))
        rr=[r for r in valid if r['dataset']=='MedPix2' and diag(r.get('reference_diagnosis'))!='Unresolved']
        positive=sum(diag(r['reference_diagnosis'])==disease for r in rr)
        g=defaultdict(lambda:np.zeros(4,dtype=int))
        for r in rr:
            key='|'.join(sorted(r['source_case_ids']));y=diag(r['reference_diagnosis'])==disease;p=diag((r.get('prediction') or {}).get('diagnosis_name'))==disease
            g[key][0 if y and p else 1 if p else 2 if y else 3]+=1
        if positive:
            r=summarize('Diagnosis extraction','MedPix2',g,len(rr),'figures','Source Case Diagnosis','One-vs-rest on ALL 115 diagnosis references; not a positive-only micro average.')
            r['positive']=positive;result.append(r)
        else:result.append(empty('Diagnosis extraction','MedPix2',len(rr),positive))
        out.append(dict(disease=disease,rows=result))
    OUT.mkdir(parents=True,exist_ok=True)
    payload=dict(bootstrap_replicates=B,seed=SEED,confidence_level=.95,method='percentile source-cluster bootstrap',
        diagnosis_policy='Disease-specific one-vs-rest, not the pooled multiclass micro metric used on previous slide.',
        grouping='MultiCaRe disease mentions in article; MedPix source final diagnosis. Strata are not equivalent clinical cohorts.',
        new_manual_review=0,diseases=out)
    (OUT/'statistics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['disease','task','dataset','n','clusters','positive','metric','estimate','ci_low','ci_high','status'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for d in out:
            for r in d['rows']:
                for k in ['Precision','Recall','F1']:
                    m=(r['metrics'] or {}).get(k)
                    w.writerow(dict(disease=d['disease'],task=r['task'],dataset=r['dataset'],n=r['n'],clusters=r['clusters'],positive=r.get('positive',''),metric=k,estimate=m['value'] if m else '',ci_low=m['low'] if m else '',ci_high=m['high'] if m else '',status='scored' if m else r['reason']))
    # Check every estimate against the original frozen report.
    with (ROOT/'reports/gbm_astro_cadasil_v1/metrics.csv').open(encoding='utf-8-sig') as f: old=list(csv.DictReader(f))
    taskmap={'Case association':'case_association','Original description':'original_description','Diagnosis extraction':'diagnosis_one_vs_rest'}
    for d in out:
        for r in d['rows']:
            if not r['metrics']:continue
            ref=next(x for x in old if x['task']==taskmap[r['task']] and x['dataset']==r['dataset'] and x['group']==d['disease'])
            for k,m in r['metrics'].items():assert np.isclose(m['value'],float(ref[k.lower()]))
    print(json.dumps(payload,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
