"""Show article flow and figure-record flow separately; never mix their units."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from medcpt_local import ROOT, sha


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def flow(paths):
    edges = Counter()
    for path in paths:
        if len(path) != len(set(path)):
            raise ValueError('Repeated node in a record route')
        edges.update(zip(path, path[1:]))
    incoming, outgoing = Counter(), Counter()
    for (a,b), count in edges.items():
        outgoing[a] += count; incoming[b] += count
    for node in incoming.keys() & outgoing.keys():
        if incoming[node] != outgoing[node]: raise ValueError('Flow not conserved')
    return edges, incoming, outgoing


def trace(nodes, edges, unit):
    totals = Counter()
    outgoing = Counter()
    for (a,b), count in edges.items():
        totals[b] += count; outgoing[a] += count
    for key in outgoing:
        if key not in totals: totals[key] = outgoing[key]
    active = {n for edge in edges for n in edge}
    keys = [n for n in nodes if n in active]
    colors = ['#d86468' if 'excluded' in key or key == 'no_asset' else '#dfa740' if
              'held' in key or 'failed' in key else '#1a8d99' if key != 'draft_test' else '#586dc6' for key in keys]
    return {'type':'sankey', 'arrangement':'snap', 'valueformat':',d', 'valuesuffix':' '+unit,
        'node': {'label':[f'{nodes[k]} · {totals[k]:,}{unit}' for k in keys], 'color':colors,
                 'pad':25, 'thickness':22, 'align':'left', 'line':{'width':0}},
        'link': {'source':[keys.index(a) for a,b in edges], 'target':[keys.index(b) for a,b in edges],
                 'value':list(edges.values()), 'color':['rgba(26,141,153,.23)' for _ in edges]}}


def record_paths(figures, inventory, dedup):
    inv = {r['sample_id']:r for r in inventory}
    final = {r['sample_id']:r for r in dedup}
    original = {r['sample_id']:r for r in figures}
    if len(inv) != len(inventory) or len(final) != len(dedup) or len(original) != len(figures):
        raise ValueError('Duplicate sample IDs')
    expected = {key for key,row in original.items() if row['assets']}
    if inv.keys() != expected or final.keys() != expected:
        raise ValueError('Figure/inventory/dedup coverage mismatch')
    paths, per_record = [], []
    for sid, raw in original.items():
        path = ['figures_raw']
        if not raw['assets']:
            path.append('no_asset')
            queue, split = 'no_asset', ''
        else:
            path.append('records_with_images')
            queue = inv[sid]['inventory_queue']
            split = final[sid]['proposed_split']
            if queue.startswith('candidate_'):
                if queue not in {'candidate_single_figure_caption_task','candidate_composite_or_serial_caption_task'} or split not in {'train','validation','test'}:
                    raise ValueError('Unexpected candidate queue/split')
                path.extend(['caption_candidates', queue, 'draft_'+split])
            else:
                if queue not in {'held_case_linkage_uncertain','held_nonpatient_figure_signal','held_animal_signal','held_modality_uncertain','held_file_or_caption_problem'}:
                    raise ValueError('Unknown inventory queue: '+queue)
                if split != 'hold': raise ValueError('Held record unexpectedly assigned a draft split')
                path.extend(['held_records', queue])
        paths.append(path)
        per_record.append({'sample_id':sid,'article_id':raw['article_id'],'figure_id':raw['figure_id'],
                           'image_assets':len(raw['assets']),'queue':queue,'proposed_split':split})
    return paths, per_record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'reports/dataset_flow_v1.html')
    args=ap.parse_args()
    article_graph_path=ROOT/'reports/collection_v3_sankey.json'
    article_graph=json.loads(article_graph_path.read_text(encoding='utf-8'))
    for name,expected in article_graph['input_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=expected: raise ValueError('Article graph source changed')
    figure_path=ROOT/'data/processed/collection_v3_figures.jsonl'
    inventory_path=ROOT/'data/datasets/formal_candidates_v1/inventory.jsonl'
    dedup_path=ROOT/'data/datasets/dedup_v1_1/records.jsonl'
    inventory_summary=json.loads((inventory_path.parent/'summary.json').read_text(encoding='utf-8'))
    dedup_summary=json.loads((dedup_path.parent/'summary.json').read_text(encoding='utf-8'))
    if sha(inventory_path)!=inventory_summary['inventory_sha256'] or sha(dedup_path)!=dedup_summary['records_sha256']:
        raise ValueError('Inventory or dedup dataset changed')
    if dedup_summary['source_sha256']!=sha(inventory_path): raise ValueError('Dedup refers to another inventory')
    figures,inventory,dedup=map(rows,[figure_path,inventory_path,dedup_path])
    paths,per_record=record_paths(figures,inventory,dedup)
    edges,inc,out=flow(paths)
    if out['figures_raw']!=len(figures): raise ValueError('Root count mismatch')
    record_nodes={
        'figures_raw':'提取的 Figure 条目', 'records_with_images':'有图片的图文记录', 'no_asset':'无图片资产，未进入图文索引',
        'caption_candidates':'影像描述任务候选', 'held_records':'暂存，尚未纳入任务候选',
        'candidate_single_figure_caption_task':'单幅图候选（文字规则判断）',
        'candidate_composite_or_serial_caption_task':'多面板／纵向比较候选',
        'held_case_linkage_uncertain':'病例关联不明', 'held_nonpatient_figure_signal':'统计图／示意图等文字信号',
        'held_animal_signal':'动物模型文字信号', 'held_modality_uncertain':'MRI模态线索不明确',
        'held_file_or_caption_problem':'文件或图注问题',
        'draft_train':'建议训练候选', 'draft_validation':'建议验证候选', 'draft_test':'建议测试候选'}
    article_nodes={r['id']:r['label'] for r in article_graph['nodes']}
    article_edges=Counter({(r['source'],r['target']):r['articles'] for r in article_graph['links']})
    traces=[trace(article_nodes,article_edges,'篇'),trace(record_nodes,edges,'条')]
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    data={'article_graph':article_graph,'record_nodes':record_nodes,
          'record_links':[{'source':a,'target':b,'records':n} for (a,b),n in edges.items()],
          'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [article_graph_path,figure_path,inventory_path,dedup_path]},
          'articles_with_candidates':len({r['article_id'] for r in inventory}),
          'figure_entries':len(figures),'image_caption_records':len(inventory),
          'image_assets':sum(len(r['assets']) for r in inventory),'flow_conserved':True}
    output.with_suffix('.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with output.with_name(output.stem+'_record_paths.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_record[0]));writer.writeheader();writer.writerows(per_record)
    plotly=(ROOT/'assets/plotly-2.35.2.min.js').read_text(encoding='utf-8')
    def js(value): return json.dumps(value,ensure_ascii=False).replace('<','\\u003c')
    charts=[]
    for i,t in enumerate(traces):
        layout={'height':850 if i==0 else 1150,'font':{'family':'Microsoft YaHei, Arial, sans-serif','size':15},
                'margin':{'l':25,'r':25,'t':35,'b':25},'paper_bgcolor':'white'}
        config={'responsive':True,'displaylogo':False,'toImageButtonOptions':{'format':'svg','filename':'RareMed_flow_'+str(i+1),'width':1700,'height':layout['height']}}
        charts.append(f"Plotly.newPlot('chart{i}',[{js(t)}],{js(layout)},{js(config)});")
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>867条记录从哪来</title>
<style>body{font:16px/1.8 'Microsoft YaHei',Arial;background:#f3f6fa;color:#203044;margin:0}header,section{background:white;padding:28px;margin:20px auto;max-width:1700px;border-radius:12px}header{border-top:6px solid #198d99}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{padding:16px 24px;background:#eaf5f6;border-radius:8px}.card b{display:block;font-size:30px}.chart{min-width:1250px}.scroll{overflow-x:auto}.bridge{background:#eaf5f6;border:1px solid #b4d9df;padding:24px;margin:20px auto;max-width:1700px;text-align:center;font-size:21px}.note{background:#fff4db;padding:18px;border-left:4px solid #dca541}code{background:#edf1f5;padding:3px 6px}h1,h2{line-height:1.4}</style>
<header><h1>867条记录，究竟是什么？</h1><p><b>一条记录＝一个论文 Figure 条目＋它的图片文件、图注、来源和处理状态。</b><br>
不是一篇文章，不是一位患者，也不一定是一张独立脑片：一幅Figure可能含多个面板、多个时间点或多个患者。</p>
<div class="cards"><div class="card"><b>370篇</b>产生候选图的文章</div><div class="card"><b>869条</b>原始Figure条目</div><div class="card"><b>867条</b>有图片的图文记录</div><div class="card"><b>315条</b>描述任务候选，尚非最终训练集</div></div>
<p>例：<code>PMC12675991_FIG2</code>是一条；同文献的<code>FIG3</code>是另一条。两条可以属于同一个患者。两张都用于了smoke训练，所以同组不能进入未接触的评测集。</p></header>
<section><h2>① 先按文章看：995篇如何流动？</h2><p>下图宽度单位始终是“篇”。每篇只有一个最终去向。</p><div class="scroll"><div id="chart0" class="chart"></div></div></section>
<div class="bridge">在这370篇文章中，提取出869个Figure条目<br><small>这是“一篇文章可产生多条记录”的展开关系，不把篇数与记录数放进同一条流量中。</small></div>
<section><h2>② 再按图文记录看：867条去了哪里？</h2><p>下图宽度单位始终是“条”。先区分有无图片，再按任务候选和暂存原因分流，最后展示候选的建议划分。</p><div class="scroll"><div id="chart1" class="chart"></div></div>
<p class="note"><b>右侧253／24／38只是建议划分，尚未冻结。</b>单幅图与多面板类型来自图注规则；病例归属、目标描述和评测资格仍需核验。</p></section>
<section><h2>③ 去重为什么没有让867减少？</h2><p>当前精确重复为0组，另找到65对相似候选；<b>65是“对数”，不是要删除65条</b>。没有删除原图，也没有把疑似相似直接认定为同一病例。</p>
<p>去重与防泄漏目前主要改变的是<b>分组关系</b>：同来源、同图和未解决的相似关系保守放同组。得到334个保守组，不等于334名患者；最大组含216条记录，提示可能存在过度合并，仍需处理。</p>
<p>315条候选仍未全部确定目标答案，也没有得到315条训练就绪样本。2条caption-pilot仅是这批数据的一个流程验证子集。</p><p>图表离线可打开；右上角相机按钮可导出SVG。</p></section>'''
    html+='<script>'+plotly+'</script><script>'+''.join(charts)+'</script></html>'
    # Keep displayed count cards honest if an upstream version changes.
    expected=(len(figures),len(inventory),len({r['article_id'] for r in inventory}),out['caption_candidates'],
              inc['draft_train'],inc['draft_validation'],inc['draft_test'])
    if expected!=(869,867,370,315,253,24,38):
        raise ValueError('Data changed: update explanatory cards before rendering this version')
    if dedup_summary['candidate_pairs']!=65 or dedup_summary['conservative_split_groups']!=334 or dedup_summary['largest_group_records']!=216:
        raise ValueError('Dedup version changed: update explanatory text')
    output.write_text(html,encoding='utf-8')
    print(json.dumps({'html':str(output),'articles':995,'figure_entries':len(figures),'records':len(inventory),'flow_conserved':True},indent=2))


if __name__=='__main__':main()
