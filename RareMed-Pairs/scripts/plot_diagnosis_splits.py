"""Render saved diagnosis split results as a local interactive report; no API calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
from medcpt_local import ROOT, sha
from dedup_leakage import rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT/'data/datasets/diagnosis_split_v1')
    ap.add_argument('--output', type=Path, default=ROOT/'reports/diagnosis_split_v1.html')
    args = ap.parse_args()
    source = args.source.resolve()
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    dedup = json.loads((source/'dedup/summary.json').read_text(encoding='utf-8'))
    if sha(source/'dedup/records.jsonl') != dedup['records_sha256']:
        raise ValueError('Dedup records changed')
    dataset = []
    for split in ['train', 'validation', 'test']:
        records = rows(source/(split+'.jsonl'))
        if len(records) != summary['split_counts'][split]:
            raise ValueError('Split count mismatch')
        if dict(Counter(r['diagnosis_family'] for r in records)) != summary['split_diagnosis_families'][split]:
            raise ValueError('Diagnosis counts changed')
        for r in records:
            link = r['proposed_linkage']
            dataset.append({'id': r['sample_id'], 'split': split, 'family': r['diagnosis_family'],
                'diagnosis': link['diagnosis_name'], 'caption': r['caption'], 'article': r['article_id'],
                'group': r['split_group_id'], 'image': (ROOT/r['assets'][0]['path']).resolve().as_uri(),
                'source_text': r.get('source_text', {})})
    if len(dataset) != summary['canonical_candidate_records']:
        raise ValueError('Canonical count mismatch')
    payload = json.dumps({'summary': summary, 'dedup': dedup, 'records': dataset,
                         'review': (source/'dedup/review.html').as_uri()}, ensure_ascii=False).replace('<', '\\u003c')
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>诊断数据集划分</title>
<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:32px;background:#f5f7fa;color:#193044}h1{font-size:28px}.cards{display:flex;gap:15px;flex-wrap:wrap}.card,section{background:white;padding:20px;border-radius:12px;margin-bottom:20px}.card b{display:block;font-size:32px;color:#127c87}.chart{height:390px}.note{line-height:1.8;color:#5c4a20}select,input,button{padding:9px;margin:5px}article{border-top:1px solid #ddd;padding:20px 0;overflow:auto}img{max-width:350px;max-height:330px;float:left;margin:10px 25px 15px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#087982}</style>
<h1>诊断数据集：去重与划分结果</h1><p>单位为 Figure 记录，不能理解为独立患者数。当前为划分草案。</p>
<div class="cards" id="cards"></div>
<section><div id="flow" class="chart"></div><p>867 是全库查重范围；302 是诊断候选范围。二者没有混入同一个训练集合。</p></section>
<section><div id="classes" class="chart"></div><div id="groupchart" class="chart"></div></section>
<section class="note" id="notes"></section>
<section><h2>逐图查看</h2><select id="split"><option value="">全部集合</option><option value="train">训练</option><option value="validation">验证</option><option value="test">测试</option></select><select id="family"><option value="">全部诊断类别</option></select><input id="query" placeholder="搜索图号、诊断或图注"><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span><div id="gallery"></div></section>
<script>__PLOTLY__</script><script>const data=__DATA__;
const S=data.summary,D=data.dedup,R=data.records;
const names={train:'训练',validation:'验证',test:'测试',CADASIL_author_reported:'作者报告 CADASIL',sporadic_CSVD_author_reported:'作者报告散发性 CSVD',needs_label_review:'诊断名称需复核',other_reported_diagnosis_needs_control_selection:'其他诊断：对照适用性待定'};
function el(tag,text){let n=document.createElement(tag);n.textContent=text;return n}
for(const [n,v] of [['全库查重图片',S.images_searched],['入选诊断候选',S.diagnosis_candidates],['去重后候选',S.canonical_candidate_records],['疑似重复图片对',D.candidate_pairs]]){let c=el('div',n);c.className='card';c.append(el('b',v));document.querySelector('#cards').append(c)}
const cfg={responsive:true,displaylogo:false};
let splitNames=['train','validation','test'];
Plotly.newPlot('flow',[{type:'sankey',node:{label:['诊断候选 '+S.diagnosis_candidates,'保留代表记录 '+S.canonical_candidate_records,'完全重复冗余 '+(S.diagnosis_candidates-S.canonical_candidate_records),...splitNames.map(k=>names[k]+' '+S.split_counts[k])],color:['#128b94','#128b94','#b9bdc6','#128b94','#e2aa45','#537ac0']},link:{source:[0,0,1,1,1],target:[1,2,3,4,5],value:[S.canonical_candidate_records,S.diagnosis_candidates-S.canonical_candidate_records,...splitNames.map(k=>S.split_counts[k])]}}],{title:'302 条诊断候选的去重与划分流向',font:{size:15},margin:{t:65,l:20,r:20,b:30}},cfg);
let families=[...new Set(R.map(r=>r.family))];
Plotly.newPlot('classes',families.map((f,i)=>({type:'bar',name:names[f]||f,x:splitNames.map(s=>names[s]),y:splitNames.map(s=>R.filter(r=>r.split===s&&r.family===f).length),marker:{color:['#128b94','#9daabd','#e0aa45','#8b6cc0'][i%4]}})),{title:'各集合的诊断类别：按 Figure 计数',barmode:'stack',legend:{orientation:'h',y:-.2},margin:{b:100},yaxis:{title:'Figure 数'}},cfg);
Plotly.newPlot('groupchart',[{type:'bar',name:'来源文章数',x:splitNames.map(s=>names[s]),y:splitNames.map(s=>new Set(R.filter(r=>r.split===s).map(r=>r.article)).size)},{type:'bar',name:'保守划分组数',x:splitNames.map(s=>names[s]),y:splitNames.map(s=>new Set(R.filter(r=>r.split===s).map(r=>r.group)).size)}],{title:'来源文章与划分组数量：均不等于患者数',barmode:'group',legend:{orientation:'h'}},cfg);
let notes=document.querySelector('#notes');notes.append(el('h2','如何解释结果'));
for(const t of [`完全重复组 ${D.exact_or_reviewed_duplicate_groups}；疑似重复对 ${D.candidate_pairs}，相似图片暂时同组，不代表确认重复。`,`全库最大保守组 ${D.largest_group_records} 条记录，可能过度合并，应检查相似图片。`,`已检测关联的跨集合泄露失败 ${S.audit_failures} 项；这不是全部患者关系均已核验的证明。`,'这里只检查数据划分关系，尚未检查最终模型输入中的答案泄露。','其他疾病尚未确认为合适阴性对照，诊断名称需复核的记录也尚未完成标签审查。',`测试集 CADASIL 候选 ${R.filter(r=>r.split==='test'&&r.family==='CADASIL_author_reported').length} 张：样本很少，目前不适合据此得出稳定的诊断提升结论。`])notes.append(el('p',t));
let a=el('a','打开疑似重复图片对照');a.href=data.review;notes.append(a);
for(const f of families){let o=el('option',names[f]||f);o.value=f;document.querySelector('#family').append(o)}
let page=0;function render(){let s=document.querySelector('#split').value,f=document.querySelector('#family').value,q=document.querySelector('#query').value.toLowerCase();let found=R.filter(r=>(!s||r.split===s)&&(!f||r.family===f)&&(!q||(r.id+' '+r.diagnosis+' '+r.caption).toLowerCase().includes(q)));let pages=Math.max(1,Math.ceil(found.length/12));page=Math.min(page,pages-1);document.querySelector('#count').textContent=`${found.length} 条 · 第 ${page+1}/${pages} 页`;let g=document.querySelector('#gallery');g.replaceChildren();for(const r of found.slice(page*12,(page+1)*12)){let art=el('article','');art.append(el('h3',r.id+' · '+names[r.split]));let im=document.createElement('img');im.src=r.image;im.loading='lazy';art.append(im,el('p','作者诊断：'+r.diagnosis),el('p','类别：'+(names[r.family]||r.family)),el('p','分组：'+r.group),el('pre',r.caption));let det=document.createElement('details');det.append(el('summary','对应原文与诊断依据（不是直接模型输入）'),el('pre',JSON.stringify(r.source_text,null,2)));art.append(det);g.append(art)}}
for(const id of ['split','family','query'])document.querySelector('#'+id).addEventListener('input',()=>{page=0;render()});document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);render()};document.querySelector('#next').onclick=()=>{page++;render()};render();</script></html>'''
    plotly = (ROOT/'assets/plotly-2.35.2.min.js').read_text(encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace('__PLOTLY__', plotly).replace('__DATA__', payload), encoding='utf-8')
    print('Report saved: '+str(args.output))


if __name__ == '__main__':
    main()
