"""Build a local human-review form, then import its exported decisions.

Human review concerns source attribution, not independent clinical diagnosis.
No automatic approvals or training-set promotion.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
from collect import ROOT, digest, dump
from link_cases import validate

VERSION = 'source-link-review-v0.1'


def load_context(article, figures_path):
    if not re.fullmatch(r'PMC\d+', article):
        raise ValueError('Invalid article ID')
    folder = ROOT/'data/model_inputs'/article
    inp, resp = folder/'input.json', folder/'response.json'
    request = json.loads(inp.read_text(encoding='utf-8'))
    response = json.loads(resp.read_text(encoding='utf-8-sig'))
    errors, _ = validate(response, request)
    if errors:
        raise ValueError('Model response must pass quote/structure validation: '+str(errors))
    records = [json.loads(line) for line in figures_path.read_text(encoding='utf-8').split('\n') if line.strip()]
    records = {r['figure_id']: r for r in records if r['article_id'] == article}
    expected = {r['figure_id'] for r in response['figures']}
    if set(records) != expected:
        raise ValueError('Figure manifest and response do not cover the same figures')
    assets = {}
    for row in records.values():
        for asset in row['assets']:
            path = (ROOT/asset['path']).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != asset['sha256']:
                raise ValueError('Image location/hash mismatch')
            assets[asset['path']] = asset['sha256']
    binding = {'input_sha256':digest(inp), 'response_sha256':digest(resp),
               'figure_manifest_sha256':digest(figures_path), 'image_hashes':assets}
    return request, response, records, binding


def validate_review(review, article, expected_ids, binding):
    errors = []
    if not isinstance(review, dict):
        return ['Review must be a JSON object']
    if review.get('review_version') != VERSION or review.get('article_id') != article:
        errors.append('Wrong review version or article')
    if review.get('binding') != binding:
        errors.append('Inputs have changed; regenerate the form and review the current files')
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        errors.append('Reviewer name or alias is required')
    if not isinstance(review.get('reviewed_at'), str) or not review['reviewed_at'].strip():
        errors.append('Review timestamp is required')
    rows = review.get('figures')
    if not isinstance(rows, list):
        return errors+['Missing review figures']
    seen = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append('Each figure review must be an object')
            continue
        fid = row.get('figure_id')
        if not isinstance(fid, str):
            errors.append('Missing figure ID')
            continue
        seen.append(fid)
        decision = row.get('decision')
        if decision not in ('approve_source_link', 'hold', 'reject'):
            errors.append(fid+': choose an explicit decision')
        checks = row.get('checks', {})
        if not isinstance(checks, dict):
            checks = {}
        if decision == 'approve_source_link' and not all(checks.get(k) is True for k in
                ('brain_mri_caption_match', 'same_case', 'source_diagnosis_supported')):
            errors.append(fid+': approval requires all three source checks')
        for field in ('arrows', 'multi_panel', 'visible_identifiers'):
            if row.get(field) not in ('yes', 'no', 'uncertain'):
                errors.append(fid+': missing '+field+' observation')
        if not isinstance(row.get('notes'), str) or not row['notes'].strip():
            errors.append(fid+': a short review note is required')
    if set(seen) != set(expected_ids) or len(seen) != len(set(seen)):
        errors.append('Review must cover every figure exactly once')
    return errors


def build(article, request, response, records, binding):
    passages = {p['passage_id']:p for p in request['passages']}
    cards = []
    for row in response['figures']:
        fid = row['figure_id']
        record = records[fid]
        images = []
        for asset in record['assets']:
            kind = {'JPEG':'jpeg','PNG':'png','WEBP':'webp','GIF':'gif'}.get(asset['format'])
            if kind:
                data = base64.b64encode((ROOT/asset['path']).read_bytes()).decode('ascii')
                images.append(f'<img src="data:image/{kind};base64,{data}" alt="{escape(fid)}">')
        evidence = []
        for role, label in [('link_evidence','病例与图号'),('diagnosis_evidence','作者报告诊断'),('basis_evidence','诊断依据')]:
            evidence.append('<h4>'+label+'</h4>')
            for item in row[role]:
                p = passages[item['passage_id']]
                quote = escape(item['quote'])
                text = escape(p['text']).replace(quote,'<mark>'+quote+'</mark>',1)
                evidence.append('<details open><summary>'+escape(item['passage_id'])+' · 原文段落</summary><p>'+text+'</p></details>')
        uncertainty = '<ul>'+''.join('<li>'+escape(v)+'</li>' for v in row['uncertainties'])+'</ul>'
        checks = ''.join('<label class="check"><input type="checkbox" data-check="'+key+'">'+label+'</label>'
            for key,label in [('brain_mri_caption_match','图片为脑MRI，且与所示图注对应'),
                              ('same_case','图号、正文与诊断指向同一病例'),
                              ('source_diagnosis_supported','作者报告的诊断状态有对应原文支持')])
        observations = ''.join('<label>'+label+' <select data-observation="'+key+'"><option value="">请选择</option>'
            '<option value="yes">有</option><option value="no">未发现</option><option value="uncertain">不确定</option></select></label>'
            for key,label in [('arrows','箭头/作者标注'),('multi_panel','多面板'),('visible_identifiers','疑似患者标识')])
        cards.append('<section class="figure" data-id="'+escape(fid)+'"><h2>'+escape(fid)+'</h2><div class="pair"><div>'
            +''.join(images)+'<h3>原文图注</h3><p>'+escape(record['caption'])+'</p></div><div><h3>待审核的模型结论</h3><p>'
            +escape('病例：'+str(row['case_label'])+'；疾病：'+str(row['diagnosis_name'])+'；状态：'+row['diagnosis_status']
                    +'；图文关联：'+row['link_status'])+'</p><p>'+escape(row['diagnosis_basis'] or '诊断依据未确定')
            +'</p>'+uncertainty+'</div></div><details><summary>展开原文证据（高亮为引用）</summary>'
            +''.join(evidence)+'</details><h3>你的审核</h3>'+checks+'<div class="observations">'+observations+'</div>'
            '<label>结论 <select data-decision><option value="">请选择</option><option value="approve_source_link">通过来源关联审核</option>'
            '<option value="hold">暂存，待进一步核查</option><option value="reject">拒绝当前关联</option></select></label>'
            '<label class="check">审核说明（必填）<textarea data-notes rows="3" placeholder="写明支持依据或未解决的问题"></textarea></label></section>')
    payload = json.dumps({'review_version':VERSION,'article_id':article,'binding':binding}, ensure_ascii=False).replace('<','\\u003c')
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>病例与图号关联审核</title>
<style>body{font:17px/1.7 "Microsoft YaHei",sans-serif;max-width:1200px;margin:35px auto;padding:0 20px;color:#173044}h1,h3{color:#087f83}.pair{display:grid;grid-template-columns:1fr 1fr;gap:25px}img{width:100%;height:auto}section{border-top:2px solid #ccc;padding:20px 0}details{background:#f3f6f8;padding:12px;margin:10px 0}mark{background:#fff0a8}.check{display:block;margin:12px 0}textarea{display:block;width:95%;font:inherit}select,input,button{font:inherit;padding:6px}.observations{display:flex;gap:20px;flex-wrap:wrap;margin:18px 0}button{background:#087f83;color:white;border:0;padding:12px 22px;cursor:pointer}#status{color:#a03020}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>病例—诊断—图号：人工来源审核</h1>
<p>这是来源与关联审核，不是独立临床确诊。所有选项初始为空；图片未改动。通过仅确认当前模型结论有来源依据，不能确认单图具有疾病特异性。</p>
<p>图片权利、全面脱敏、重复病例及训练集划分另行处理，审核后仍不会自动进入训练集。疑似标识仅是目视观察，不是脱敏认证。</p>
<p><a href="https://pmc.ncbi.nlm.nih.gov/articles/ARTICLE/" target="_blank" rel="noopener">打开原文章</a></p>
<label>审核者姓名或代号 <input id="reviewer" placeholder="填写你的姓名或代号"></label>
<p>填写后请导出记录。刷新或关闭本页不会自动保存；本页不发送网络请求，原文章链接由你选择打开。</p>
CARDS
<button id="export">导出审核JSON</button><p id="status"></p>
<script id="binding" type="application/json">PAYLOAD</script>
<script>
document.getElementById('export').addEventListener('click',()=>{
 const out=JSON.parse(document.getElementById('binding').textContent);
 out.reviewer=document.getElementById('reviewer').value.trim();out.reviewed_at=new Date().toISOString();out.figures=[];
 const errors=[];if(!out.reviewer)errors.push('请填写审核者姓名或代号');
 document.querySelectorAll('.figure').forEach(section=>{
  const row={figure_id:section.dataset.id,decision:section.querySelector('[data-decision]').value,checks:{},notes:section.querySelector('[data-notes]').value.trim()};
  section.querySelectorAll('[data-check]').forEach(el=>row.checks[el.dataset.check]=el.checked);
  section.querySelectorAll('[data-observation]').forEach(el=>row[el.dataset.observation]=el.value);
  if(!row.decision||!row.notes||!row.arrows||!row.multi_panel||!row.visible_identifiers)errors.push(row.figure_id+'：请填写结论、观察和说明');
  if(row.decision==='approve_source_link'&&!Object.values(row.checks).every(Boolean))errors.push(row.figure_id+'：通过需勾选三项来源检查；无法确认请选择暂存');
  out.figures.push(row);
 });
 const status=document.getElementById('status');if(errors.length){status.textContent=errors.join('；');return;}
 const url=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download=out.article_id+'_human_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 status.textContent='审核文件已交给浏览器下载；请将下载文件放入项目的 data/reviews/ 文件夹。';
});
</script></html>'''
    html = html.replace('ARTICLE',article).replace('CARDS',''.join(cards)).replace('PAYLOAD',payload)
    target = ROOT/'reports'/(article+'_human_review.html')
    target.write_text(html,encoding='utf-8')
    (ROOT/'data/reviews').mkdir(parents=True,exist_ok=True)
    dump(ROOT/'reports'/(article+'_review_form_receipt.json'),
         {'review_version':VERSION,'binding':binding,'form_sha256':digest(target),
          'figures':len(records),'human_decisions_recorded':0})
    print('Review form: '+str(target))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['build','import'])
    ap.add_argument('--article',default='PMC12675991')
    ap.add_argument('--figures',type=Path,default=ROOT/'data/processed/fulltext_pilot_3_figures.jsonl')
    ap.add_argument('--review',type=Path)
    args=ap.parse_args()
    request,response,records,binding=load_context(args.article,args.figures)
    if args.command=='build':
        build(args.article,request,response,records,binding)
        return
    if not args.review:
        ap.error('--review is required for import')
    review=json.loads(args.review.read_text(encoding='utf-8-sig'))
    errors=validate_review(review,args.article,records,binding)
    if errors:
        ap.exit(1,'\n'.join(errors)+'\n')
    decisions={r['figure_id']:r for r in review['figures']}
    output=[]
    for prediction in response['figures']:
        row={'article_id':args.article,'figure_id':prediction['figure_id'],
             'proposed_linkage':prediction,'source_review':decisions[prediction['figure_id']],
             'reviewer':review['reviewer'],'reviewed_at':review['reviewed_at'],
             'review_scope':'source_linkage_only_not_independent_clinical_diagnosis',
             'training_ready':False,'remaining_reviews':['figure_rights','privacy','duplicate_cases','split_assignment'],
             'binding':binding}
        output.append(row)
    target=ROOT/'data/processed'/(args.article+'_reviewed_links.jsonl')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    receipt={'article_id':args.article,'imported_at':datetime.now(timezone.utc).isoformat(),
             'review_version':VERSION,'review_file_sha256':digest(args.review),
             'decisions':dict(Counter(r['decision'] for r in review['figures'])),
             'training_ready':0,'output_sha256':digest(target)}
    dump(ROOT/'reports'/(args.article+'_review_import_receipt.json'),receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
