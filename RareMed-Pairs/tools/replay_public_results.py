"""Recompute published disease-specific P/R/F1 and cluster-bootstrap CIs offline."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def scores(v):
 tp,fp,fn,tn=np.moveaxis(np.asarray(v,dtype=float),-1,0)
 with np.errstate(divide='ignore',invalid='ignore'):
  return np.stack([tp/(tp+fp),tp/(tp+fn),2*tp/(2*tp+fp+fn)],axis=-1)
def main():
 data=json.loads((ROOT/'results/cluster_confusions.json').read_text(encoding='utf8'));checked=0;unavailable=0
 for row in data['rows']:
  if not row['metrics']:unavailable+=1;continue
  patterns=np.asarray(row['patterns_TP_FP_FN_TN'],dtype=int);counts=np.asarray(row['cluster_multiplicities'],dtype=int)
  assert counts.sum()==row['clusters']
  total=counts@patterns;assert total.tolist()==row['confusion']
  rng=np.random.default_rng(data['seed']);boot=rng.multinomial(counts.sum(),counts/counts.sum(),size=data['bootstrap_replicates'])@patterns
  est=scores(total);ci=np.nanpercentile(scores(boot),[2.5,97.5],axis=0)
  for i,name in enumerate(['Precision','Recall','F1']):
   m=row['metrics'][name]
   np.testing.assert_allclose([est[i],ci[0,i],ci[1,i]],[m['value'],m['low'],m['high']],rtol=0,atol=1e-12,equal_nan=True)
   checked+=1
 print(json.dumps({'status':'PASS','metric_estimates_and_CIs_verified':checked,'unscorable_rows_preserved':unavailable,'bootstrap_replicates':data['bootstrap_replicates'],'seed':data['seed'],'API_calls':0,'case_text_required':False},indent=2))
if __name__=='__main__':main()
