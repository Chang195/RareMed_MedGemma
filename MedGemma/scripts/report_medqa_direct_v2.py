"""Paired reporting for direct-answer v2; run with separate figure environment."""
from pathlib import Path
import json,hashlib,importlib.util
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'medqa-direct-v2'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=read(ROOT/'mixture-evaluation-v1/protocol.json');spec=read(OUT/'protocol.json');pid=sha(OUT/'protocol.json')
qref=pd.read_csv(ROOT/'mixture-evaluation-v1/medqa_reference.csv',keep_default_na=False).set_index('question_id')
records=[]
for model in ['Base','A','B','C']:
 files=sorted((OUT/'predictions/test'/model).glob('*.json'));assert len(files)==1273
 for f in files:
  r=read(f);assert r['protocol_sha256']==pid and r['model']==model
  assert qref.loc[r['sample_id'],'gold_answer']==r['reference']
  records.append(dict(model=model,question_id=r['sample_id'],question_sha256=qref.loc[r['sample_id'],'question_sha256'],
    gold_answer=r['reference'],predicted_answer=r['prediction'],raw_output=r['raw_output'],checkpoint=r['checkpoint'],
    protocol_id=pid,dataset_revision=old['medqa']['revision'],model_revision=old['revision'],
    format_valid=r['format_valid'],truncated=r['truncated']))
df=pd.DataFrame(records);df.to_csv(OUT/'medqa_item_results.csv',index=False,encoding='utf-8-sig')
module=importlib.util.spec_from_file_location('stats',ROOT/'MedGemma_SFT_Figures/plot_sft_results.py');stats=importlib.util.module_from_spec(module);module.loader.exec_module(stats)
manifest=read(ROOT/'MedGemma_SFT_Figures/input_manifest.json')
manifest.update(medqa_protocol_id=pid,medqa_dataset_revision=old['medqa']['revision'],medqa_expected_n=1273,medqa_sft_overlap_audit_passed=True)
results=stats.medqa(df,manifest,10000,42,qref.reset_index())
for m in results:
 x=df[df.model==m];results[m].update(format_valid_n=int(x.format_valid.sum()),truncated_n=int(x.truncated.sum()))
summary=pd.DataFrame([dict(model=m,**v) for m,v in results.items()]);summary.to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
report=dict(protocol_sha256=pid,results=results,validation=read(OUT/'validation-report.json'),
 method='Actual greedy generation, strict completed single-letter parser, invalid counts incorrect. Wilson accuracy CI, paired bootstrap delta CI 10000 seed42.',
 caution=spec['disclosure'],script_sha256=sha(Path(__file__)))
(OUT/'paired_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(summary[['model','medqa_correct','medqa_n','medqa_accuracy','format_valid_n','truncated_n','medqa_delta','medqa_delta_low','medqa_delta_high']].to_string(index=False))
