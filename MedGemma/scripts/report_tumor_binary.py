"""Image-level binary task statistics with conservative source-group bootstrap."""
from pathlib import Path
import json,importlib.util,collections,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tumor-binary-evaluation-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
p=read(OUT/'protocol.json');digest=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('binary_stats',ROOT/'MedGemma_SFT_Figures/plot_sft_results.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
summary=[];all_records=[];changes=[]
for task,reference in p['reference'].items():
 ids=[r['sample_id'] for r in reference];truth={r['sample_id']:r for r in reference}
 y=np.array([r['reference']==p['tasks'][task]['positive'] for r in reference],dtype=int)
 group=np.array([r['cluster_id'] for r in reference]);unique=np.unique(group);indices={g:np.flatnonzero(group==g) for g in unique}
 assert all(len(np.unique(y[i]))==1 for i in indices.values())
 strata=[np.array([g for g in unique if y[indices[g][0]]==c]) for c in [0,1]];rng=np.random.default_rng(42)
 draws=[np.concatenate([indices[g] for s in strata for g in rng.choice(s,len(s),replace=True)]) for _ in range(10000)]
 base_correct={}
 for m in ['Base','A','B','C']:
  assert read(OUT/'completed'/f'{m}-{task}.json')['protocol_sha256']==digest
  records=[read(f) for f in sorted((OUT/'predictions'/task/m).glob('*.json'))];assert [r['sample_id'] for r in records]==ids
  assert all(r['reference']==truth[r['sample_id']]['reference'] and r['protocol_sha256']==digest for r in records)
  all_records.extend(records);resolved=np.array([r['prediction'] in [p['tasks'][task]['positive'],p['tasks'][task]['negative']] for r in records])
  row=dict(task=task,model=m,n=len(records),positive_n=int(y.sum()),negative_n=int((1-y).sum()),
    positive_groups=p['summaries'][task]['positive_groups'],unresolved_n=int((~resolved).sum()),
    unresolved_positive_n=int(np.sum((~resolved)&(y==1))),unresolved_negative_n=int(np.sum((~resolved)&(y==0))))
  if resolved.all():
   pred=np.array([r['prediction']==p['tasks'][task]['positive'] for r in records],dtype=int);row.update(stats.rates(y,pred))
   row['all_positive_prediction_collapse']=bool(np.all(pred==1))
   boot=[stats.rates(y[i],pred[i]) for i in draws]
   for metric in ['recall','fpr','specificity','balanced_accuracy','precision','F1']:
    low,high=stats.ci([b[metric] for b in boot]);row[metric+'_low']=low;row[metric+'_high']=high
   positive_groups=len({group[i] for i in np.flatnonzero(y==1)})
   negative_groups=len({group[i] for i in np.flatnonzero(y==0)})
   # A single source group cannot estimate between-positive-group uncertainty.
   unavailable=set()
   if positive_groups<2:unavailable.update(['recall','balanced_accuracy','precision','F1'])
   if negative_groups<2:unavailable.update(['fpr','specificity','balanced_accuracy','precision','F1'])
   for metric in unavailable:row[metric+'_low']=row[metric+'_high']=np.nan
   if unavailable:row['ci_not_estimable']='Single source group in one class: '+','.join(sorted(unavailable))
  else:row['status']='Unresolved retained; main binary estimates withheld.'
  summary.append(row)
  for r in records:
   correct=r['prediction']==r['reference']
   if m=='Base':base_correct[r['sample_id']]=correct
   else:changes.append(dict(task=task,model=m,sample_id=r['sample_id'],reference=r['reference'],prediction=r['prediction'],
     change='improved' if correct and not base_correct[r['sample_id']] else 'regressed' if not correct and base_correct[r['sample_id']] else 'unchanged',
     unresolved_counts_incorrect_for_change_only=not bool(r['prediction'])))
pd.DataFrame(summary).to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(all_records).drop(columns=['input_ids','generated_token_ids']).to_csv(OUT/'per_sample_results.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(changes).to_csv(OUT/'paired_changes.csv',index=False,encoding='utf-8-sig')
(OUT/'report.json').write_text(json.dumps(dict(protocol_sha256=digest,metrics=summary,bootstrap_repeats=10000,seed=42,limitations=p['limitations']),indent=2),encoding='utf-8')
(OUT/'status.json').write_text(json.dumps(dict(status='COMPLETE',protocol_sha256=digest),indent=2),encoding='utf-8')
print(pd.DataFrame(summary).to_string(index=False))
