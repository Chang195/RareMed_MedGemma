"""Report official-notebook-parser accuracy plus truncation/completion diagnostics."""
from pathlib import Path
import json,hashlib,importlib.util
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'medqa-official-style-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
p=read(OUT/'protocol.json');digest=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
qref=pd.read_csv(ROOT/'mixture-evaluation-v1/medqa_reference.csv',keep_default_na=False).set_index('question_id')
records=[]
for m in ['Base','A','B','C']:
 files=sorted((OUT/'predictions/test'/m).glob('*.json'));assert len(files)==1273
 for f in files:
  r=read(f);assert r['model']==m and r['protocol_sha256']==digest and r['gold_answer']==qref.loc[r['sample_id'],'gold_answer']
  records.append(dict(model=m,question_id=r['sample_id'],question_sha256=qref.loc[r['sample_id'],'question_sha256'],
   gold_answer=r['gold_answer'],predicted_answer=r['predicted_answer'] or '',raw_output=r['raw_output'],checkpoint=r['checkpoint'],
   protocol_id=digest,dataset_revision=p['dataset_repository']['revision'],model_revision=p['revision'],truncated=r['truncated'],
   completed_correct=r['completed_correct'],generated_tokens=r['generated_tokens'],elapsed_seconds=r['elapsed_seconds']))
df=pd.DataFrame(records);df.to_csv(OUT/'medqa_item_results.csv',index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('stats',ROOT/'MedGemma_SFT_Figures/plot_sft_results.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
manifest=read(ROOT/'MedGemma_SFT_Figures/input_manifest.json');manifest.update(medqa_protocol_id=digest,medqa_dataset_revision=p['dataset_repository']['revision'],medqa_expected_n=1273,medqa_sft_overlap_audit_passed=True)
results=stats.medqa(df,manifest,10000,42,qref.reset_index())
for m,r in results.items():
 x=df[df.model==m];r.update(truncated_n=int(x.truncated.sum()),completed_correct_n=int(x.completed_correct.sum()),
   completed_conservative_accuracy=float(x.completed_correct.mean()),mean_generated_tokens=float(x.generated_tokens.mean()))
pd.DataFrame([dict(model=m,**v) for m,v in results.items()]).to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
(OUT/'paired_report.json').write_text(json.dumps(dict(protocol_sha256=digest,results=results,adaptations=p['adaptations'],limits=p['limits']),indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
