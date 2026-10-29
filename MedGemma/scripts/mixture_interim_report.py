"""Read-only inference audit + explicitly separate terminal-token parser correction.

Run with .figure-build/.venv/Scripts/python.exe. Does not change frozen outputs.
"""
from pathlib import Path
import argparse, collections, csv, hashlib, importlib.util, json, re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'mixture-evaluation-v1'
args=argparse.ArgumentParser()
args.add_argument('--models',nargs='+',choices=['Base','A','B','C'],default=['Base','A'])
MODELS=args.parse_args().models
assert MODELS[0]=='Base' and len(set(MODELS))==len(MODELS)
OUT=RUN/('interim-'+'-'.join(MODELS)+'-v1');OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=read(RUN/'protocol.json'); original_id=sha(RUN/'protocol.json')
assert sha(ROOT/'scripts/mixture_evaluation.py')==protocol['code_sha256']['scripts/mixture_evaluation.py']
eos=protocol['diagnosis_generation']['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
assert 106 in eos
parser_id=original_id+':terminal-eot-correction-v1'
diagnostic=[];qa=[];frozen={};inputs=[]
for m in MODELS:
    for task,total in [('diagnosis-test',68),('medqa',1273)]:
        completed=read(RUN/'completed'/f'{m}-{task}.json')
        assert completed['protocol_sha256']==original_id and completed['n']==total
        files=sorted((RUN/'predictions'/m/task).glob('*.json'));assert len(files)==total
        records=[read(f) for f in files]
        assert all(r['model']==m and r['protocol_id']==original_id for r in records)
        inputs.extend(dict(path=str(f),sha256=sha(f)) for f in files)
        if task=='medqa':qa.extend(records);continue
        frozen[m]=dict(n=total,categories=dict(collections.Counter(r['parse_category'] for r in records)),
            resolved=sum(r['y_pred'] in ['CADASIL','non-CADASIL'] for r in records))
        for r in records:
            tokens=r['generated_token_ids'];assert tokens and tokens[-1]==106 and not r['truncated']
            # Serialization correction only. Do not alter any clinical wording or negation.
            assert r['clean_output'].endswith('<end_of_turn>')
            cleaned=r['clean_output'][:-len('<end_of_turn>')].strip()
            assert cleaned in ['CADASIL','non-CADASIL'], 'Stop rather than invent an interpretation.'
            corrected={**r,'y_pred':cleaned,'parse_category':'explicit_positive' if cleaned=='CADASIL' else 'explicit_negative',
               'protocol_id':parser_id,'original_parse_category':r['parse_category'],'corrected_clean_output':cleaned}
            diagnostic.append(corrected)
spec=importlib.util.spec_from_file_location('stats',ROOT/'MedGemma_SFT_Figures/plot_sft_results.py')
stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats);stats.MODELS=MODELS
manifest=read(ROOT/'MedGemma_SFT_Figures/input_manifest.json')
manifest.update(diagnostic_protocol_id=parser_id,medqa_protocol_id=original_id,
  medqa_dataset_revision=protocol['medqa']['revision'],medqa_expected_n=1273,medqa_sft_overlap_audit_passed=True)
reference=pd.read_csv(ROOT/'MedGemma_SFT_Figures/test_reference.csv',keep_default_na=False)
d=stats.diagnostic(pd.DataFrame(diagnostic),reference,manifest,10000,42)
questions=[read(f) for f in sorted((RUN/'predictions/Base/medqa').glob('*.json'))]
qref=pd.DataFrame(questions)[['question_id','question_sha256','gold_answer']]
q=stats.medqa(pd.DataFrame(qa),manifest,10000,42,qref)
summary=pd.DataFrame([dict(model=m,**d[m],**q[m]) for m in MODELS])
summary.to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(diagnostic).drop(columns=['input_ids']).to_csv(OUT/'diagnostic_terminal_token_corrected.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(qa).drop(columns=['input_ids']).to_csv(OUT/'medqa_item_results.csv',index=False,encoding='utf-8-sig')
diag_changes=[]
base={r['sample_id']:r for r in diagnostic if r['model']=='Base'}
for r in diagnostic:
    if r['model']=='Base':continue
    b=base[r['sample_id']];cb=b['y_pred']==b['y_true'];ca=r['y_pred']==r['y_true']
    diag_changes.append(dict(model=r['model'],sample_id=r['sample_id'],cluster_id=r['cluster_id'],reference=r['y_true'],base_prediction=b['y_pred'],sft_prediction=r['y_pred'],
      change='improved' if ca and not cb else 'regressed' if cb and not ca else 'unchanged'))
pd.DataFrame(diag_changes).to_csv(OUT/'diagnostic_paired_changes.csv',index=False,encoding='utf-8-sig')
change_counts={m:dict(collections.Counter(r['change'] for r in diag_changes if r['model']==m)) for m in MODELS if m!='Base'}
report=dict(status='INTERIM: '+' / '.join(MODELS)+'; not a final four-arm conclusion',original_protocol_sha256=original_id,
  diagnostic_correction_id=parser_id,correction='Remove one exact terminal <end_of_turn> marker only, verified generated final token 106 belongs to frozen EOS IDs. All remaining outputs are exact original binary labels.',
  primary_frozen_parser_results=frozen,primary_diagnostic_metrics='Withheld: frozen parser treated every output as unresolved. Corrected metrics are a separately labeled bug-fix analysis.',
  corrected_diagnostic_metrics=d,medqa_metrics=q,diagnostic_change_counts=change_counts,
  bootstrap_repeats=10000,seed=42,script_sha256=sha(__file__),statistics_script_sha256=sha(ROOT/'MedGemma_SFT_Figures/plot_sft_results.py'),
  limitations=protocol['limits'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'input_inventory.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
print(summary[['model','TP','FN','FP','TN','recall','fpr','balanced_accuracy','precision','F1','medqa_correct','medqa_n','medqa_accuracy','medqa_delta','medqa_delta_low','medqa_delta_high','medqa_improved','medqa_regressed']].to_string(index=False))
print('Diagnostic changes:',report['diagnostic_change_counts'])
print('All-CADASIL collapse:',{m:d[m]['all_CADASIL_prediction_collapse'] for m in MODELS})
