"""Task-specific public-reference evaluation, with unsupported tasks kept explicit.

No API calls or new predictions. Existing outputs are evaluated where comparable.
SLAKE image-only requests are exported separately for a real visual predictor.
"""
import argparse
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import re
import zipfile
from medcpt_local import ROOT, sha
from dedup_leakage import rows, write_rows, save
from dataset_leakage import normalize
from evaluate_curation import metrics

OUT=ROOT/'data/benchmarks/task_suite_v1'
SOURCE=ROOT/'data/benchmarks/four_disease/v1'
PROMPT='Does this image show a brain MRI? Reply with exactly one of: YES, NO, UNCERTAIN.'


def text_scores(prediction, reference):
    # Transparent word-level overlap, not disease sensitivity or semantic truth.
    p=re.findall(r'\w+',normalize(prediction));r=re.findall(r'\w+',normalize(reference))
    overlap=sum((Counter(p)&Counter(r)).values())
    precision=overlap/len(p) if p else 0
    recall=overlap/len(r) if r else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    previous=[0]*(len(r)+1)
    for word in p:
        current=[0]
        for j,target in enumerate(r,1):
            current.append(previous[j-1]+1 if word==target else max(previous[j],current[-1]))
        previous=current
    lcs=previous[-1]
    rouge=2*lcs/(len(p)+len(r)) if p or r else 0
    return {'normalized_exact_match':normalize(prediction)==normalize(reference),
            'word_precision':precision,'word_recall':recall,'word_f1':f1,'rouge_l_f1':rouge}


def prepare():
    slake=ROOT/'data/benchmarks/slake_audit'
    split=json.loads((slake/'test.json').read_text(encoding='utf-8'))
    grouped=defaultdict(list)
    for r in split:grouped[r['img_name']].append(r)
    folder=OUT/'slake_brain_mri';(folder/'images').mkdir(parents=True,exist_ok=True)
    requests=[];gold=[];ambiguous=[]
    with zipfile.ZipFile(slake/'imgs.zip') as archive:
        names=archive.namelist()
        for name, records in sorted(grouped.items()):
            labels={(r['modality'],r['location']) for r in records}
            if len(labels)!=1:
                ambiguous.append(name);continue
            modality,location=next(iter(labels))
            sid='slake_'+str(records[0]['img_id'])
            members=[n for n in names if n==name or n.endswith('/'+name)]
            if len(members)!=1:raise ValueError('Missing/ambiguous image archive member: '+name)
            target=folder/'images'/(sid+'.jpg')
            raw=archive.read(members[0]);target.write_bytes(raw)
            # Never include source location, modality, QA answers or case diagnoses
            # in the model-facing requests.
            requests.append({'sample_id':sid,'image':'images/'+target.name,'question':PROMPT})
            gold.append({'sample_id':sid,'positive':modality=='MRI' and location in {'Brain','Brain_Tissue'},
                         'source_modality':modality,'source_location':location,'source_image_name':name,'image_sha256':sha(target)})
    if len({r['sample_id'] for r in requests})!=len(requests):raise ValueError('Repeated SLAKE image IDs')
    write_rows(folder/'requests.jsonl',requests);write_rows(folder/'references.jsonl',gold)
    save(folder/'manifest.json',{'task':'brain_MRI_image_identification_new_component_not_caption_pipeline',
        'images':len(gold),'positive':sum(r['positive'] for r in gold),'negative':sum(not r['positive'] for r in gold),
        'requests_sha256':sha(folder/'requests.jsonl'),'reference_sha256':sha(folder/'references.jsonl'),
        'source_sha256':sha(slake/'test.json'),'archive_sha256':sha(slake/'imgs.zip'),
        'ambiguous_excluded':ambiguous,'disease_subgroups':'not_available: no case-level CADASIL/MS/NMOSD labels',
        'note':'Visual identification benchmark; cannot score existing caption-only filters using QA answers as input.'})
    (folder/'README.md').write_text('''# SLAKE brain MRI identification

This is a new visual-component test, not the existing caption extraction pipeline.
Model input: image + fixed question from requests.jsonl only.
Do not feed references.jsonl, source metadata, QA answers or filenames encoding labels.
Use a genuinely image-capable model and record its ID/revision, prompt, preprocessing and decoding settings.
Write predictions.jsonl: {"sample_id":"slake_102","prediction":"YES","model_id":"actual-model","run_id":"run-name"}
Allowed prediction values: YES, NO, UNCERTAIN. Do not infer disease subgroups from QA mentions.
Score on the originating computer with task_benchmark_suite.py score-slake --predictions PATH.
UNCERTAIN is not silently discarded; coverage and all-image positive-detection recall are reported.
''',encoding='utf-8')
    write_rows(folder/'prediction_template.jsonl',[{'sample_id':r['sample_id'],'prediction':None,'model_id':None,'run_id':None} for r in requests])
    print(json.dumps({'slake_images':len(gold),'positives':sum(r['positive'] for r in gold),'output':str(folder)},indent=2))


def matched_reference_rows():
    refs={r['sample_id']:r for r in rows(SOURCE/'silver_references.jsonl')}
    pred={r['sample_id']:r for r in rows(SOURCE/'predictions.jsonl')}
    for f in rows(SOURCE/'figures.jsonl'):
        yield f,refs[f['sample_id']],pred.get(f['sample_id'],{})


def actual_disease(name):
    name=normalize(name)
    if name in {'multiple sclerosis'}:return 'multiple_sclerosis'
    if name=='cadasil' or name.startswith('cerebral autosomal dominant arteriopathy'):return 'CADASIL'
    if name in {'nmosd','neuromyelitis optica','neuromyelitis optica spectrum disorder'}:return 'NMOSD'
    return 'other_or_not_normalized'


def aggregate(records):
    if not records:return {'status':'no_comparable_reference','n':0}
    return {'status':'source_reference_agreement_only','n':len(records),
        **{k:sum(float(r['scores'][k]) for r in records)/len(records) for k in records[0]['scores']}}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    diag=[];captions=[];med_cases={};case_link_coverage=[];mismatched=0
    for f,ref,p in matched_reference_rows():
        if ref['dataset']=='MedPix2':
            reference=ref.get('source_diagnosis') or ''
            link=p.get('link') or {};prediction=link.get('diagnosis_name') or ''
            if reference:
                diag.append({'sample_id':f['sample_id'],'reference':reference,'prediction':prediction,
                    'scores':text_scores(prediction,reference),'source_final_disease':actual_disease(reference),
                    'sampling_mentions':ref['strata'],'output_status':p.get('inference_status')})
                med_cases[f['article_id']]={'final_diagnosis':reference,'source_final_disease':actual_disease(reference),'sampling_mentions':ref['strata']}
        else:
            alternatives=ref.get('source_captions',[])
            if alternatives:
                candidates=[(text_scores(f['caption'],r),r) for r in alternatives]
                best,reference=max(candidates,key=lambda x:(x[0]['normalized_exact_match'],x[0]['word_f1']))
                captions.append({'sample_id':f['sample_id'],'reference':reference,'prediction':f['caption'],
                    'scores':best,'sampling_mentions':ref['strata'],'source_case_ids':ref.get('source_case_ids',[])})
            else:mismatched+=1
            case_link_coverage.append({'sample_id':f['sample_id'],'source_case_ids':ref.get('source_case_ids',[]),
                'predicted_case_label':(p.get('link') or {}).get('case_label'),
                'status':'not_scored_natural_language_label_vs_source_ID_no_independent_mapping'})
    write_rows(OUT/'medpix_diagnosis_details.jsonl',diag)
    write_rows(OUT/'multicare_caption_details.jsonl',captions)
    write_rows(OUT/'case_linkage_availability.jsonl',case_link_coverage)
    disease_summary={d:{'diagnosis_field_agreement':aggregate([r for r in diag if r['source_final_disease']==d]),
        'unique_medpix_cases':sum(v['source_final_disease']==d for v in med_cases.values()),
        'multicare_caption_agreement_on_mention_stratum':aggregate([r for r in captions if d in r['sampling_mentions']]),
        'note':'MultiCaRe stratum means case-text mention, not confirmed disease.'} for d in ['CADASIL','multiple_sclerosis','NMOSD']}
    panel=ROOT/'reports/curation_eval_v1/summary.json'
    legacy=json.loads(panel.read_text(encoding='utf-8')) if panel.exists() else {}
    manifest=OUT/'slake_brain_mri/manifest.json'
    status={'brain_mri_selection':{'status':'requests_ready_predictions_required' if manifest.exists() else 'run_prepare_first',
            'note':'SLAKE uses image modality/anatomy truth; not runnable with a caption-only tool.'},
        'image_text_pairing':{'multicare_whole_caption':aggregate(captions),
            'unmatched_source_figures':mismatched,'medpix':'not_scored_same_caption_supplied_as_input',
            'medicat':'not_scored_no_existing_panel_caption_prediction_component; token-F1/ROUGE-L scorer available separately'},
        'case_linkage':{'status':'not_scored_no_comparable_identity_prediction',
            'note':'Do not copy source ID or compare descriptive case labels literally. No new API or identity algorithm introduced.'},
        'author_diagnosis':{'medpix_name_field':aggregate(diag),'five_state_status':'not_scored_missing_reference',
            'multicare':'not_scored_no_structured_final_diagnosis_gold'}}
    result={'tasks':status,'three_disease_analysis':disease_summary,'medpix_unique_cases':med_cases,
        'existing_medicat_panel_rule_recall':legacy.get('medicat_positive_challenge',{}).get('recall'),
        'note':'Panel-detection recall is an auxiliary legacy component, not image-text alignment. These task results cannot be pooled into one end-to-end F1.',
        'input_sha256':{n:sha(SOURCE/n) for n in ['figures.jsonl','silver_references.jsonl','predictions.jsonl']},
        'script_sha256':sha(Path(__file__)),'new_api_calls':0}
    save(OUT/'summary.json',result)
    html='<!doctype html><meta charset="utf-8"><title>分任务公开评测</title><style>body{font-family:Arial,"Microsoft YaHei";margin:30px}pre{white-space:pre-wrap;line-height:1.7}p{line-height:1.8}</style><h1>分数据集、分任务评测 v1</h1><p>严格区分来源字段一致性、真实图像识别、任务未实现。缺失不当零分；不把复制参考答案算成功。</p><pre>'+escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>'
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'medpix_name_field':aggregate(diag),'multicare_caption':aggregate(captions),'report':str(OUT/'report.html'),'new_api_calls':0},indent=2))


def score_slake(path):
    folder=OUT/'slake_brain_mri';manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if sha(folder/'requests.jsonl')!=manifest['requests_sha256'] or sha(folder/'references.jsonl')!=manifest['reference_sha256']:raise ValueError('Frozen benchmark changed')
    gold={r['sample_id']:r['positive'] for r in rows(folder/'references.jsonl')}
    predictions=rows(path);mapping={r['sample_id']:r for r in predictions}
    if len(mapping)!=len(predictions) or mapping.keys()!=gold.keys():raise ValueError('Predictions must cover every unique test image exactly once')
    for r in predictions:
        if r.get('prediction') not in {'YES','NO','UNCERTAIN'} or not r.get('model_id') or not r.get('run_id'):raise ValueError('Real predictions, model_id and run_id required; templates cannot be scored')
    scores=metrics(list(gold.values()),[mapping[k]['prediction']=='YES' for k in gold])
    answered=[k for k in gold if mapping[k]['prediction']!='UNCERTAIN']
    result={'task':'visual_brain_MRI_identification','all_image_positive_detection':scores,
        'uncertain':len(gold)-len(answered),'coverage':len(answered)/len(gold),
        'uncertain_positive':sum(gold[k] and mapping[k]['prediction']=='UNCERTAIN' for k in gold),
        'uncertain_negative':sum(not gold[k] and mapping[k]['prediction']=='UNCERTAIN' for k in gold),
        'answered_only':metrics([gold[k] for k in answered],[mapping[k]['prediction']=='YES' for k in answered]) if answered else None,
        'note':'All-image metrics count UNCERTAIN as no positive detection; answered-only results must be read with coverage. This is not the existing caption extractor.',
        'prediction_sha256':sha(path),'source_manifest_sha256':sha(folder/'manifest.json')}
    save(folder/'scores.json',result);print(json.dumps(result,indent=2))


def score_captions(path):
    # Generic interface for actual independently generated, panel-matched outputs.
    # Explicitly not connected to current pipeline, which has no panel predictor.
    required={'sample_id','prediction','reference','prediction_origin','reference_origin'}
    data=rows(path)
    if not data:raise ValueError('No paired predictions')
    for r in data:
        if not required<=r.keys() or not r['prediction_origin'] or not r['reference_origin'] or r['prediction_origin']==r['reference_origin']:raise ValueError('Independent prediction/reference provenance required')
    result={'n':len(data),'scores':aggregate([{'scores':text_scores(r['prediction'],r['reference'])} for r in data]),
        'note':'Generic word metrics only. Not MedPMC exact reproduction or proof of image-semantic alignment; use properly paired human-reference samples.',
        'input_sha256':sha(path)}
    save(OUT/'external_caption_scores.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');sub.add_parser('run')
    p=sub.add_parser('score-slake');p.add_argument('--predictions',type=Path,required=True)
    p=sub.add_parser('score-captions');p.add_argument('--pairs',type=Path,required=True)
    args=ap.parse_args()
    {'prepare':prepare,'run':run,'score-slake':lambda:score_slake(args.predictions),'score-captions':lambda:score_captions(args.pairs)}[args.command]()
