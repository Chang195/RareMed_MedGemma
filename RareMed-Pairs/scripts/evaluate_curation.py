"""Frozen public-test evaluation of existing caption rules; no training or API inference."""
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
from inventory_dataset import MRI, PANELS, SERIAL
from preview_dataset import ANIMAL, NONPATIENT
from medcpt_local import ROOT, sha


def metrics(gold, pred):
    if len(gold) != len(pred) or not gold:
        raise ValueError('Nonempty equal-length labels required')
    tp = sum(g and p for g,p in zip(gold,pred))
    fp = sum(not g and p for g,p in zip(gold,pred))
    fn = sum(g and not p for g,p in zip(gold,pred))
    tn = sum(not g and not p for g,p in zip(gold,pred))
    def div(a,b): return a/b if b else None
    return {'n':len(gold), 'positive':tp+fn, 'negative':tn+fp,
            'TP':tp,'FP':fp,'FN':fn,'TN':tn,
            'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),
            'specificity':div(tn,tn+fp),'f1':div(2*tp,2*tp+fp+fn),
            'accuracy':(tp+tn)/len(gold)}


def predict(caption):
    nonmedical = bool(ANIMAL.search(caption) or NONPATIENT.search(caption))
    panels = len(PANELS.findall(caption)) >= 2 or bool(re.search(r'\bmulti.panel\b', caption, re.I))
    return {'medical_text_proxy':not nonmedical,
            'actual_mri_caption_gate':bool(MRI.search(caption)) and not nonmedical,
            'multipanel_caption_flag':panels,
            'composite_or_serial_flag':panels or bool(SERIAL.search(caption))}


def main():
    out = ROOT/'reports/curation_eval_v1'
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT/'data/benchmarks/medpmc'
    screen = pq.read_table(base/'screening_test.parquet', columns=['image_id','caption','reference_text','is_medical']).to_pylist()
    panel = pq.read_table(base/'panel_test.parquet', columns=['image_id','caption','is_multipanel','source']).to_pylist()
    logs, results = [], {}
    for name, data, label, methods in [
        ('screening',screen,'is_medical',['medical_text_proxy','actual_mri_caption_gate']),
        ('panels',panel,'is_multipanel',['multipanel_caption_flag','composite_or_serial_flag'])]:
        for method in methods:
            truth, predictions = [], []
            for r in data:
                predicted = predict(r['caption'])[method]
                truth.append(r[label]); predictions.append(predicted)
                logs.append({'dataset':name,'method':method,'image_id':r['image_id'],
                    'gold':r[label],'prediction':predicted,'correct':predicted==r[label],
                    'source':r.get('source','MedPMC screening test'), 'caption':r['caption']})
            results[method] = metrics(truth,predictions)
    strata = {}
    for origin in sorted({r['source'] for r in panel}):
        subset = [r for r in panel if r['source']==origin]
        strata[origin] = metrics([r['is_multipanel'] for r in subset], [predict(r['caption'])['multipanel_caption_flag'] for r in subset])
    # Independent MedICaT stress test. The annotation set is compound-selected,
    # so no specificity/negative-population inference is permitted.
    med = ROOT/'data/benchmarks/medicat'
    annotations = [json.loads(s) for s in (med/'subcaptions_public.jsonl').read_text(encoding='utf-8').split('\n') if s.strip()]
    split = json.loads((med/'subcaption_split_keys.json').read_text(encoding='utf-8'))
    keys = {(x['pdf_hash'],x['fig_key']) for x in split['test']}
    grouped = {}
    for r in annotations:
        key=(r['pdf_hash'],r['fig_key'])
        if key in keys: grouped.setdefault(key,[]).append(r)
    challenge=[]; excluded=Counter()
    for key, versions in grouped.items():
        counts={len(r.get('subfigures',[])) for r in versions}
        if len(counts)!=1:
            excluded['conflicting_annotation_panel_counts']+=1;continue
        r=versions[0]
        if len(r.get('subfigures',[]))<2:
            excluded['not_confirmed_compound_by_box_count']+=1;continue
        challenge.append(r)
    stress = metrics([True]*len(challenge),[predict(r['text'])['multipanel_caption_flag'] for r in challenge])
    stress['interpretation']='Compound-selected positive-only challenge: report recall; F1/precision do not establish general classification performance.'
    stress['excluded']=dict(excluded)
    for r in challenge:
        p=predict(r['text'])['multipanel_caption_flag']
        logs.append({'dataset':'medicat_compound_test','method':'multipanel_caption_flag','image_id':r['pdf_hash']+'_'+r['fig_key'],
                     'gold':True,'prediction':p,'correct':p,'source':'MedICaT official test','caption':r['text']})
    with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in logs),encoding='utf-8')
    availability={
        'initial_screening':'Measured existing exclusion-rule proxy and actual MRI gate; neither is MedPMC PubMedBERT. Proxy ignores inline references; MRI gate has a narrower target.',
        'multi_panel_detection':'Measured existing caption heuristic against both positive and negative released labels.',
        'figure_separation':'Skipped by user.',
        'caption_separation_alignment':'Not measured: no panel-specific prediction implementation. Whole-caption copying is not a valid substitute.',
        'medical_figure_classification':'Not measured as an independent visual component: no visual classifier or matched classification benchmark acquired. Screening proxy is NOT an independent stage-5 result.'}
    identities={str(p.relative_to(ROOT)):sha(p) for p in [base/'screening_test.parquet',base/'panel_test.parquet',med/'subcaptions_public.jsonl',med/'subcaption_split_keys.json',Path(__file__),ROOT/'scripts/inventory_dataset.py',ROOT/'scripts/preview_dataset.py']}
    summary={'results':results,'panel_source_strata':strata,'medicat_positive_challenge':stress,
        'availability':availability,'input_sha256':identities,'new_api_calls':0,
        'screening_dataset_revision':'5cfe878f39c60665faeca0caecf44c8355117026',
        'panel_dataset_revision':'5497657b90323198fab4c2117d74d84ddeae6caa',
        'test_policy':'Existing rules frozen before evaluation; no tuning on this test. Future edits need development data.',
        'sources':['https://huggingface.co/datasets/Yale-BIDS-Chen/medpmc-screening-dataset',
                   'https://huggingface.co/datasets/Yale-BIDS-Chen/medpmc-multi-fig-detection-dataset',
                   'https://github.com/allenai/medicat'],
        'limitations':['Public component performance is not CADASIL label/association accuracy.',
                       'Released panel test labels include 3456 ImageCLEF rows and 500 rows tagged PMC_synthetic; do not describe all labels as independently human-adjudicated.',
                       'No visual-model inference, diagnostic model evaluation, clinical review or image-level truth audit performed.',
                       'Published MedPMC numbers are reference only; verify released split identity and metric implementation before claiming exact head-to-head comparison.',
                       'MedICaT challenge uses subcaption-task test keys, not a representative binary panel test.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    names={'medical_text_proxy':'医学相关性：既有排除规则代理', 'actual_mri_caption_gate':'实际 MRI 图注门槛（与通用医学标签不同任务）',
           'multipanel_caption_flag':'多面板图注规则','composite_or_serial_flag':'多面板或连续随访图注规则'}
    def pct(v):return 'N/A' if v is None else f'{v*100:.2f}%'
    columns=['n','positive','negative','TP','FP','FN','TN','precision','recall','specificity','f1','accuracy']
    table='| 方法 | '+' | '.join(columns)+' |\n|---|'+'---|'*len(columns)+'\n'
    for name,m in results.items():
        table+='| '+names[name]+' | '+' | '.join(pct(m[c]) if c in ['precision','recall','specificity','f1','accuracy'] else str(m[c]) for c in columns)+' |\n'
    md='# 数据处理组件评测 v1\n\n'+table+'\n## 解释范围\n\n'
    md+='初筛医学相关性是从现有动物/图表排除规则构造的透明代理，不是原先的疾病全文筛选，也不是新训练分类器。实际 MRI 门槛另外报告，不能用其分数与通用医学相关性模型排名。\n\n'
    md+=f"MedICaT 官方测试中的可判定拼图挑战：{stress['n']} 条，检出 {stress['TP']}，漏检 {stress['FN']}，召回率 {pct(stress['recall'])}。缺少阴性，不能用于推断特异度或总体 F1。\n\n"
    md+='## 拼图标签来源分层\n\n| 来源字段 | N | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|\n'
    for origin,m in strata.items():
        md+=f"| {origin} | {m['n']} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} |\n"
    md+='## 项目状态\n\n'+ '\n'.join('- '+k+': '+v for k,v in availability.items())
    md+='\n\n## 限制\n\n'+'\n'.join('- '+s for s in summary['limitations'])+'\n\n## 来源\n\n'+'\n'.join('- '+u for u in summary['sources'])
    (out/'report.md').write_text(md,encoding='utf-8')
    cards=''
    for name in results:
        m=results[name]
        cards+=f'<h2>{escape(names[name])}</h2><p>N={m["n"]}；阳性 {m["positive"]}，阴性 {m["negative"]}；F1 {pct(m["f1"])}，召回 {pct(m["recall"])}</p><div id="{name}" style="height:300px"></div>'
    errors=''
    for r in [r for r in logs if not r['correct'] and r['method'] in ['medical_text_proxy','multipanel_caption_flag']][:60]:
        errors+=f'<details><summary>{escape(r["dataset"]+" / "+r["image_id"])} · 真值 {r["gold"]} / 预测 {r["prediction"]}</summary><pre>{escape(r["caption"])}</pre></details>'
    plotly=(ROOT/'assets/plotly-2.35.2.min.js').read_text(encoding='utf-8')
    payload=json.dumps(results).replace('<','\\u003c')
    html='<!doctype html><meta charset="utf-8"><title>数据处理评测</title><style>body{font-family:Arial,"Microsoft YaHei";margin:35px;max-width:1200px}pre{white-space:pre-wrap}p{line-height:1.8}details{padding:10px;border-bottom:1px solid #ddd}</style><h1>公开测试集：现有文字规则性能</h1><p>这是处理工具的外部基线，不是 CADASIL 数据集质量或 MedGemma 诊断成绩。没有使用付费 API，没有针对测试标签调规则。</p>'+cards+'<h2>评测状态与限制</h2><pre>'+escape(md)+'</pre><h2>错误样例（前60条；完整结果见 predictions.csv）</h2>'+errors+'<script>'+plotly+'</script><script>const R='+payload+';for(const [k,m] of Object.entries(R)){Plotly.newPlot(k,[{type:"heatmap",x:["预测阴性","预测阳性"],y:["实际阴性","实际阳性"],z:[[m.TN,m.FP],[m.FN,m.TP]],text:[[m.TN,m.FP],[m.FN,m.TP]],texttemplate:"%{text}",colorscale:"Blues"}],{margin:{t:20,b:50},xaxis:{title:"预测"},yaxis:{autorange:"reversed"}},{responsive:true,displaylogo:false})}</script>'
    (out/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'results':results,'medicat_positive_recall':stress['recall'],'report':str(out/'report.html')},indent=2))


if __name__=='__main__':main()
