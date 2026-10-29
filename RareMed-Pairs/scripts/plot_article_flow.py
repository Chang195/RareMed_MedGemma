"""Render the completed v3 article flow as a standalone, offline Plotly Sankey."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULLTEXT = {
    'fulltext_passed': '全文线索通过',
    'held_fulltext_unavailable': '全文不可获得',
    'held_fulltext_error': '全文获取或解析错误',
    'held_fulltext_incomplete': '全文不完整',
    'excluded_fulltext_signals_insufficient': '疾病 / MRI 全文线索不足',
}
ACQUISITION = {
    'candidate': '提取到候选图的文章',
    'held_no_mri_caption_signal': '未找到明确 MRI 图注线索',
    'failed': '图文采集失败',
    'excluded_license_by_nc_nd': '全文许可识别为 CC BY-NC-ND',
}


def indexed(rows):
    result = {}
    for row in rows:
        key = row['article_id']
        if key in result:
            raise ValueError(f'Duplicate article ID: {key}')
        result[key] = row
    return result


def read_jsonl(path):
    # U+2028/U+2029 inside JSON strings must not become record separators.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def build_flow(candidates, screened, results):
    initial, full, acquired = map(indexed, (candidates, screened, results))
    if initial.keys() != full.keys():
        raise ValueError('Initial and full-text manifests have different article IDs')
    expected = {key for key, row in full.items() if row['queue'] == 'fulltext_passed'}
    if acquired.keys() != expected:
        raise ValueError('Acquisition is incomplete or contains unexpected article IDs; finish collection first')
    counts = Counter()
    paths = []
    for key, row in initial.items():
        queue = full[key]['queue']
        if row['queue'] == 'excluded_license_by_nc_nd':
            if queue != 'excluded_license_by_nc_nd':
                raise ValueError(f'Inconsistent license exclusion: {key}')
            route = ['pool', 'license_excluded']
            status = ''
        else:
            if queue not in FULLTEXT:
                raise ValueError(f'Unknown or unfinished full-text status: {queue}')
            route = ['pool', 'retained', queue]
            status = ''
            if queue == 'fulltext_passed':
                status = acquired[key]['status']
                if status not in ACQUISITION:
                    raise ValueError(f'Unknown acquisition status: {status}')
                route.append('acq_' + status)
        counts.update(zip(route, route[1:]))
        paths.append({'article_id': key, 'title': row.get('title', ''),
                      'license_stage': route[1], 'fulltext_status': queue,
                      'acquisition_status': status, 'terminal_stage': route[-1]})
    incoming, outgoing = Counter(), Counter()
    for (source, target), value in counts.items():
        outgoing[source] += value
        incoming[target] += value
    for node in incoming.keys() & outgoing.keys():
        if incoming[node] != outgoing[node]:
            raise ValueError(f'Flow is not conserved at {node}')
    terminal = sum(value for node, value in incoming.items() if node not in outgoing)
    if outgoing['pool'] != len(initial) or terminal != len(initial):
        raise ValueError('Article totals are not conserved')
    return counts, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/collection_v3_sankey.html')
    args = parser.parse_args()
    sources = [ROOT / 'data/manifests/brain_mri_all_candidates_v3.jsonl',
               ROOT / 'data/manifests/brain_mri_fulltext_v3.jsonl',
               ROOT / 'reports/collection_v3_receipt.json']
    initial, screened = (read_jsonl(p) for p in sources[:2])
    receipt = json.loads(sources[2].read_text(encoding='utf-8'))
    actual_hash = hashlib.sha256(sources[1].read_bytes()).hexdigest()
    if receipt.get('manifest_sha256') != actual_hash:
        raise ValueError('Acquisition receipt does not match the current full-text manifest')
    counts, paths = build_flow(initial, screened, receipt['results'])
    labels = {'pool': '初始检索文章', 'retained': '许可阶段保留',
              'license_excluded': '排除 CC BY-NC-ND', **FULLTEXT,
              **{'acq_' + key: value for key, value in ACQUISITION.items()}}
    order = ['pool', 'retained', 'license_excluded', *FULLTEXT,
             *('acq_' + key for key in ACQUISITION)]
    active = {node for edge in counts for node in edge}
    nodes = [node for node in order if node in active]
    totals = Counter()
    for (source, target), value in counts.items():
        totals[target] += value
    totals['pool'] = len(paths)
    colors = ['#dc6966' if 'excluded' in node else '#d5a14c' if 'held_' in node or node.endswith('failed')
              else '#198e99' for node in nodes]
    x = [0.01 if n == 'pool' else 0.28 if n in ('retained', 'license_excluded')
         else 0.61 if n in FULLTEXT else 0.99 for n in nodes]
    ys = {'pool': .35, 'retained': .29, 'license_excluded': .88,
          'fulltext_passed': .25, 'held_fulltext_unavailable': .67,
          'held_fulltext_error': .76, 'held_fulltext_incomplete': .85,
          'excluded_fulltext_signals_insufficient': .94,
          'acq_candidate': .16, 'acq_held_no_mri_caption_signal': .47,
          'acq_failed': .65, 'acq_excluded_license_by_nc_nd': .92}
    edges = [{'source': s, 'target': t, 'articles': v} for (s, t), v in counts.items()]
    trace = {'type': 'sankey', 'arrangement': 'snap', 'valueformat': ',d', 'valuesuffix': ' 篇',
             'node': {'label': [f'{labels[n]} · {totals[n]:,} 篇' for n in nodes],
                      'color': colors, 'pad': 30, 'thickness': 22, 'x': x,
                      'y': [ys[n] for n in nodes], 'line': {'width': 0}},
             'link': {'source': [nodes.index(e['source']) for e in edges],
                      'target': [nodes.index(e['target']) for e in edges],
                      'value': [e['articles'] for e in edges],
                      'color': ['rgba(25,142,153,0.22)' for e in edges]}}
    figure_path = ROOT / 'data/processed/collection_v3_figures.jsonl'
    figures = read_jsonl(figure_path)
    figure_articles = {row['article_id'] for row in figures}
    candidate_articles = {row['article_id'] for row in paths if row['acquisition_status'] == 'candidate'}
    if figure_articles != candidate_articles:
        raise ValueError('Figure manifest article coverage does not match acquisition candidates')
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows, fields in [('_article_paths.csv', paths, list(paths[0])),
                                  ('_links.csv', edges, ['source', 'target', 'articles'])]:
        with output.with_name(output.stem + suffix).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [*sources, figure_path]}
    payload = {'nodes': [{'id': n, 'label': labels[n], 'articles': totals[n]} for n in nodes],
               'links': edges, 'input_sha256': provenance, 'article_total': len(paths),
               'figure_records': len(figures), 'conservation_checked': True}
    output.with_suffix('.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    plotly = (ROOT / 'assets/plotly-2.35.2.min.js').read_text(encoding='utf-8')
    config = {'displaylogo': False, 'responsive': True,
              'toImageButtonOptions': {'format': 'svg', 'filename': output.stem, 'width': 1600, 'height': 1000}}
    layout = {'title': {'text': 'RareMed-Pairs · 文章采集流向（collection v3）', 'x': .03},
              'font': {'family': 'Microsoft YaHei, Arial, sans-serif', 'size': 15},
              'height': 900, 'margin': {'l': 35, 'r': 35, 't': 90, 'b': 40}, 'paper_bgcolor': 'white'}
    def js(value):
        return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>RareMed-Pairs 文章流向</title><style>body{{font-family:Microsoft YaHei,Arial;margin:20px;background:#f5f8fa}}main{{background:white;padding:20px;min-width:1150px}}p{{color:#465563;line-height:1.7}}</style>
<main><div id="chart"></div><p>单位为文章，每篇文章只有一个最终去向；各分支合计 {len(paths):,} 篇。综述未按文章类型排除。<br>
候选图记录：{len(figures):,} 条（单独统计，不与文章数相加）。候选图尚未等同于确诊 CADASIL 影像，也未标记为训练就绪。<br>
悬停查看数量；右上角相机按钮导出 SVG。许可保留仅表示进入采集，不代表已获再发布许可。</p></main>
<script>{plotly}</script><script>Plotly.newPlot('chart', [{js(trace)}], {js(layout)}, {js(config)});</script></html>'''
    output.write_text(html, encoding='utf-8')
    print(json.dumps({'html': str(output), 'articles': len(paths), 'figure_records': len(figures),
                      'conservation_checked': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
