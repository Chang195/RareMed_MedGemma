"""Re-score saved diagnosis outputs in a new, explicitly post-hoc analysis directory."""
import os
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent

def main():
    python=ROOT/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.path.insert(0,str(ROOT/'scripts'))
    from posthoc_parser import VERSION,extract
    from eval_scoring import diagnosis_metrics
    p=argparse.ArgumentParser(description='Post-hoc extraction only; no model inference or original result edits')
    p.add_argument('--run',default='diagnosis-night-01'); p.add_argument('--eval',default='evaluation-v1')
    p.add_argument('--analysis',default='posthoc-extraction-v2')
    a=p.parse_args()
    assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',x) for x in [a.run,a.eval,a.analysis])
    source=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'runs'))/a.run/a.eval
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=read(source/'protocol.json'); out=source/a.analysis
    assert not out.exists(), 'Analysis already exists; refusing overwrite'
    protected={str(f.relative_to(source)):digest(f) for f in source.glob('predictions/diagnosis-*/*/*.json')}
    protected['protocol.json']=digest(source/'protocol.json')
    protected['paired-report.json']=digest(source/'paired-report.json')
    out.mkdir(parents=True)
    report=dict(analysis='POST_HOC_SENSITIVITY_ANALYSIS_NOT_PREREGISTERED',parser_version=VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),reference_used_by_parser=False,
        model_identity_used_by_parser=False,new_inference=False,original_files_sha256=protected,
        results_provisional_pending_independent_review=True,splits={})
    review=[]; mapping={}
    for split in ['validation','test']:
        parsed={}
        for tag in ['base','sft']:
            files=sorted((source/'predictions'/('diagnosis-'+split)/tag).glob('*.json'))
            assert len(files)==(24 if split=='validation' else 33), 'Complete original predictions required'
            rows=[]
            for file in files:
                original=read(file)
                decision=extract(original['clean_output'],original['truncated'],protocol['other_label_vocabulary'])
                record=dict(original,original_v1_category=original['category'],category=decision['category'],v2_extraction=decision)
                rows.append(record)
                key=hashlib.sha256((report['original_files_sha256']['protocol.json']+split+tag+original['sample_id']).encode()).hexdigest()[:20]
                review.append(dict(review_id=key,split=split,output=original['clean_output'],truncated=original['truncated'],
                    reviewer_category='',reviewer_primary_diagnosis='',comments=''))
                mapping[key]=dict(model=tag,sample_id=original['sample_id'],split=split)
            parsed[tag]=rows
            (out/f'{split}-{tag}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        shared=[i for i,(x,y) in enumerate(zip(parsed['base'],parsed['sft'])) if x['category'] in ['positive','negative','other'] and y['category'] in ['positive','negative','other']]
        report['splits'][split]=dict(base=diagnosis_metrics(parsed['base']),sft=diagnosis_metrics(parsed['sft']),
            common_resolved=dict(n=len(shared),base=diagnosis_metrics([parsed['base'][i] for i in shared]),sft=diagnosis_metrics([parsed['sft'][i] for i in shared])))
    review.sort(key=lambda r:r['review_id'])
    with (out/'blinded-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(review[0])); writer.writeheader(); writer.writerows(review)
    (out/'review-id-map.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    report['parser_source_sha256']=digest(ROOT/'scripts/posthoc_parser.py')
    report['limitations']='Post-hoc rule design after outputs were observed; automatic extraction is not clinical adjudication. First asserted diagnosis is scored separately from explanatory quality; findings may not be definitive diagnoses. Test labels are source reports.'
    for path,value in protected.items(): assert digest(source/path)==value, 'Original file changed during analysis'
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for split,r in report['splits'].items():
        print(split)
        for tag in ['base','sft']:
            print(tag,json.dumps(dict(coverage=r[tag]['coverage'],metrics=r[tag]['resolved_only'],unresolved_positive=r[tag]['unresolved_reference_CADASIL'],unresolved_other=r[tag]['unresolved_reference_other']),ensure_ascii=False))
    print('POST-HOC provisional report:',out/'report.json')
    print('Original predictions/protocol/report unchanged. Independent review template:',out/'blinded-review.csv')

if __name__=='__main__': main()
