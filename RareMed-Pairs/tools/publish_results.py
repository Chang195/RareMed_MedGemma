"""Prepare anonymous aggregate sufficient statistics from frozen local predictions.
Writes only results/ and a temporary replay directory, never source datasets/reports.
No case IDs, paragraphs, captions, images, prompts or API payloads are exported.
"""
import contextlib,io,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 sys.path.insert(0,str(ROOT/'scripts'))
 import summarize_disease_pipeline_ci as source
 captured=[];original=source.summarize
 def capture(task,ds,groups,n,unit,reference,details):
  patterns,counts=np.unique(np.asarray(list(groups.values()),dtype=int),axis=0,return_counts=True)
  captured.append({'task':task,'dataset':ds,'patterns_TP_FP_FN_TN':patterns.tolist(),'cluster_multiplicities':counts.tolist()})
  return original(task,ds,groups,n,unit,reference,details)
 source.summarize=capture;source.OUT=ROOT/'.runtime_tmp/public_results_replay'
 with contextlib.redirect_stdout(io.StringIO()):source.main()
 payload=json.loads((source.OUT/'statistics.json').read_text(encoding='utf8'))
 iterator=iter(captured);ledger=[]
 for d in payload['diseases']:
  for row in d['rows']:
   if not row['metrics']:ledger.append({'disease':d['disease'],**row});continue
   item=next(iterator);assert item['task']==row['task'] and item['dataset']==row['dataset']
   ledger.append({'disease':d['disease'],**row,**item})
 assert next(iterator,None) is None
 out=ROOT/'results';out.mkdir(exist_ok=True)
 (out/'cluster_confusions.json').write_text(json.dumps({'seed':payload['seed'],'bootstrap_replicates':payload['bootstrap_replicates'],'confidence_level':.95,'vector_order':['TP','FP','FN','TN'],'privacy':'Aggregate unique confusion vectors and multiplicities only; no source case IDs or text.','rows':ledger},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 shutil.copy2(source.OUT/'statistics.csv',out/'disease_statistics.csv')
 for name in ['metrics.csv','matrices.json','coverage.json']:
  shutil.copy2(ROOT/'reports/gbm_astro_cadasil_v1'/name,out/name)
 figures=ROOT/'reports/gbm_astro_cadasil_v1/plots'
 if figures.exists():
  (out/'figures').mkdir(exist_ok=True)
  for p in figures.iterdir():
   if p.suffix in {'.png','.svg'}:shutil.copy2(p,out/'figures'/p.name)
 print(json.dumps({'public_rows':len(ledger),'scored_rows':len(captured),'clinical_source_text_exported':False,'original_predictions_changed':False}))
if __name__=='__main__':main()
