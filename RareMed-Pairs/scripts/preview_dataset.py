"""Build a local, self-contained per-figure preview; never exports training-ready SFT."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
from html import escape
import io
import json
from pathlib import Path
import re
from medcpt_local import ROOT, sha
from evidence_rules import DIAGNOSTIC_LEAK

ANIMAL = re.compile(r'\b(?:mice|mouse|murine|5xFAD|transgenic|animal\s+model)\b', re.I)
NONPATIENT = re.compile(r'\b(?:flowchart|flow\s+chart|scatter\s*plots?|box\s*plots?|violin\s*plots?|Kaplan|heatmap|schematic|correlation(?:s)?|forest\s+plot|study\s+design)\b', re.I)
CLINICAL_FIELDS = ('presentation', 'medical_history', 'family_history', 'demographics')


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def local_path(value):
    path = (ROOT/value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Source path escapes project')
    return path


def check_span(span, passages):
    p = passages.get(span['passage_id'])
    a, b = span['start_in_passage'], span['end_in_passage']
    if p is None or not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(p['text']):
        raise ValueError('Invalid source span')
    if p['text'][a:b] != span['text']:
        raise ValueError('Preview text is not an exact source slice')


def classify_record(row):
    link = row['prior_linkage']
    text = row['caption'] + ' ' + (link.get('case_label') or '') + ' ' + (link.get('diagnosis_name') or '')
    flags = []
    if ANIMAL.search(text):
        flags.append('animal_model_text_signal')
    if NONPATIENT.search(row['caption']):
        flags.append('nonpatient_figure_text_signal')
    if link['link_status'] == 'multiple_cases':
        flags.append('multiple_cases_or_cohorts')
    elif link['link_status'] not in {'explicit', 'contextual'} or not link.get('case_label'):
        flags.append('single_case_not_established')
    if flags:
        disposition = 'hold_nonpatient_or_unresolved'
    elif link['diagnosis_status'] != 'author_reported_confirmed':
        disposition = 'single_case_uncertain_diagnosis'
    else:
        disposition = 'single_case_author_confirmed_candidate'
    return disposition, flags


def prepare_preview(row, pool):
    disposition, flags = classify_record(row)
    link = row['prior_linkage']
    caption_flags = []
    if DIAGNOSTIC_LEAK.search(row['caption']):
        caption_flags.append('caption_contains_diagnostic_terms_not_pure_visual_description')
    if 'ambiguous_or_non_mri_reference' in {s.get('routing_status') for s in row['figure_reference_spans']}:
        caption_flags.append('body_reference_needs_figure_or_modality_attribution')
    clinical = []
    seen = set()
    for field in CLINICAL_FIELDS:
        for span in pool['candidate_pools'].get(field, []):
            key = (span['passage_id'], span['start_in_passage'], span['end_in_passage'])
            if key in seen:
                continue
            seen.add(key)
            clinical.append({**span, 'retrieval_field': field, 'assignment_status': 'article_candidate_not_assigned_to_image'})
    confirmed = disposition == 'single_case_author_confirmed_candidate' and bool(row['author_diagnosis']['name'])
    return {**row, 'preview_disposition': disposition, 'text_screening_flags': flags,
            'clinical_context_assigned': None, 'clinical_context_candidates': clinical,
            'task_drafts': {
                'image_description': {
                    'question': 'Describe the abnormalities visible in this MRI figure.',
                    'input_clinical_text': None,
                    'target_candidate': row['caption'],
                    'target_source': 'verbatim_original_caption',
                    'status': 'blocked_nonpatient_or_unresolved' if flags else 'candidate_caption_requires_task_fit_check',
                    'flags': caption_flags,
                    'note': 'The original caption is displayed unchanged; it may include nonvisual facts, panel labels or diagnosis. It is not an approved answer.'},
                'image_and_context_diagnosis': {
                    'question': 'What is the most likely diagnosis given this MRI figure and the clinical context?',
                    'input_clinical_text': None,
                    'target_candidate': row['author_diagnosis']['name'] if confirmed else None,
                    'target_source': 'prior_unreviewed_author_reported_diagnosis',
                    'status': 'candidate_requires_context_assignment' if confirmed else 'blocked_no_confirmed_single_human_case_candidate',
                    'note': 'Context is deliberately unassigned. Source diagnosis alone does not prove the image supports this task.'}},
            'training_ready': False, 'sft_exported': False}


def image_uri(asset):
    from PIL import Image
    path = local_path(asset['path'])
    if sha(path) != asset['sha256']:
        raise ValueError('Image checksum mismatch')
    with Image.open(path) as image:
        width, height = image.size
        image.convert('RGB').save(buf := io.BytesIO(), format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(), (width, height)


def render_span(span):
    return ('<blockquote><small>' + escape(str(span.get('passage_id', '')) +
        f" [{span.get('start_in_passage', '?')}:{span.get('end_in_passage', '?')}] " +
        str(span.get('retrieval_field', span.get('routing_status', '')))) + '</small><pre>' + escape(span['text']) + '</pre></blockquote>')


def render_html(records, output):
    names = {'hold_nonpatient_or_unresolved': '暂存：非患者图／动物模型／关联不明',
             'single_case_uncertain_diagnosis': '单病例候选：诊断未确诊',
             'single_case_author_confirmed_candidate': '单病例候选：作者报告确诊（未审核）'}
    cards = []
    for row in records:
        images = ''
        for asset in row['images']:
            uri, size = image_uri(asset)
            images += f'<figure><img loading="lazy" src="{uri}"><figcaption>{size[0]} × {size[1]} · {escape(asset["path"])}</figcaption></figure>'
        title = escape(row['sample_id'])
        scope = escape(row['scope'])
        flags = escape(', '.join(row['text_screening_flags']) or '未发现上述文字信号，尚未检查像素')
        description = row['task_drafts']['image_description']
        diagnosis = row['task_drafts']['image_and_context_diagnosis']
        quote_sections = ''
        for label, spans in [('图号关联原文（可能涉及多图或其他模态）', row['figure_reference_spans']),
                              ('作者诊断原文依据', row['author_diagnosis']['evidence']),
                              ('作者诊断依据原文', row['diagnostic_basis']['evidence'])]:
            quote_sections += '<details><summary>' + label + f' · {len(spans)}</summary>' + ''.join(render_span(s) for s in spans) + '</details>'
        cards.append(f'''<article data-group="{row['preview_disposition']}"><h2>{title}</h2>
<p class="badge">{names[row['preview_disposition']]}</p><p>范围：{scope} ｜ 图像、患者归属和语义均未独立审核</p>
<p>文字筛查：{flags}</p><div class="grid"><div>{images}<h3>原始图注（完整保留）</h3><pre>{escape(row['caption'])}</pre></div>
<div><h3>任务一：影像描述草稿</h3><p>输入：这幅图＋固定问题；不包含图注。</p>
<pre>{escape(description['question'])}</pre><p>目标候选：左侧原始图注，尚未认定为合格视觉描述。</p>
<p>{escape(', '.join(description['flags']) or '仍需确认图注与任务相符')}</p>
<h3>任务二：图像＋临床上下文的诊断草稿</h3><pre>{escape(diagnosis['question'])}</pre>
<p>临床输入：尚未分配。下面是文章级候选，不能默认属于当前图片。</p>
<p>目标候选：<b>{escape(diagnosis['target_candidate'] or '无：暂不建立此任务')}</b></p>
<p>作者原始诊断：{escape(str(row['author_diagnosis']['name']))} ｜ {escape(row['author_diagnosis']['status'])}</p>
<details><summary>临床原文候选 · {len(row['clinical_context_candidates'])}</summary>
{''.join(render_span(s) for s in row['clinical_context_candidates']) or '<p>没有候选。</p>'}</details>
{quote_sections}<p>来源：{escape(row['source_url'])}<br>许可：{escape(row['license'])}</p></div></div></article>''')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareMed-Pairs 逐图样本预览</title>
<style>body{font:16px/1.65 'Microsoft YaHei',Arial;margin:0;background:#f3f6fa;color:#213044}header{padding:24px;background:#fff;position:sticky;top:0;z-index:2;border-bottom:1px solid #ccd5df}main{max-width:1500px;margin:auto}article{margin:24px 12px;background:white;border-radius:12px;padding:24px;box-shadow:0 2px 8px #ddd}.grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}img{max-width:100%;max-height:650px;object-fit:contain}figure{margin:0}figcaption,small{color:#637186;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 Consolas,monospace;background:#f4f7fa;padding:12px}blockquote{margin:12px 0;border-left:3px solid #168b97;padding-left:12px}.badge{display:inline-block;background:#e5f3f5;padding:4px 10px;border-radius:5px}summary{cursor:pointer;padding:8px;background:#f0f3f7;margin-top:8px}select{font:inherit;padding:5px}h2{overflow-wrap:anywhere}@media(max-width:900px){.grid{grid-template-columns:1fr}header{position:static}}</style>
<header><h1>RareMed-Pairs · 逐图样本预览</h1><p>这是可追溯的候选样本草稿，不是训练集或临床评测标准答案。原文没有生成改写，未导出 SFT。</p>
<label>筛选 <select id="filter"><option value="all">全部记录</option><option value="single_case_author_confirmed_candidate">作者报告确诊的单病例候选</option><option value="single_case_uncertain_diagnosis">诊断未确诊的单病例候选</option><option value="hold_nonpatient_or_unresolved">非患者图／动物模型／关联不明</option></select></label> <span id="count"></span></header><main>'''
    html += ''.join(cards) + '''</main><script>
const filter=document.getElementById('filter');function update(){let n=0;document.querySelectorAll('article').forEach(a=>{let show=filter.value==='all'||a.dataset.group===filter.value;a.hidden=!show;if(show)n++;});document.getElementById('count').textContent=n+' 条记录';}filter.addEventListener('change',update);update();</script></html>'''
    output.write_text(html, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT/'data/processed/medcpt_pilot10_v2_1/evidence_records.jsonl')
    ap.add_argument('--output', type=Path, default=ROOT/'reports/sample_preview_v1')
    args = ap.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    source_dir = args.source.parent
    summary_path = source_dir/'summary.json'
    if json.loads(summary_path.read_text(encoding='utf-8'))['records_sha256'] != sha(args.source):
        raise ValueError('Evidence manifest checksum mismatch')
    plan = json.loads((source_dir/'plan.json').read_text(encoding='utf-8'))
    records, pools, originals, image_files = [], {}, {}, {}
    seen = set()
    for row in read_rows(args.source):
        if row['sample_id'] in seen:
            raise ValueError('Duplicate sample ID')
        seen.add(row['sample_id'])
        article = row['article_id']
        if article not in originals:
            path = ROOT/'data/model_inputs'/article/'input.json'
            if sha(path) != plan['article_input_sha256'][article]:
                raise ValueError('Article input changed')
            originals[article] = {p['passage_id']: p for p in json.loads(path.read_text(encoding='utf-8'))['passages']}
        poolpath = local_path(row['article_candidate_pool_path'])
        if sha(poolpath) != row['article_candidate_pool_sha256']:
            raise ValueError('Candidate pool changed')
        if article not in pools:
            pools[article] = json.loads(poolpath.read_text(encoding='utf-8'))
            for candidates in pools[article]['candidate_pools'].values():
                for span in candidates:
                    check_span(span, originals[article])
        for span in [*row['figure_reference_spans'], *row['author_diagnosis']['evidence'], *row['diagnostic_basis']['evidence']]:
            check_span(span, originals[article])
        for asset in row['images']:
            path = local_path(asset['path'])
            if sha(path) != asset['sha256']:
                raise ValueError('Image checksum mismatch')
            image_files[asset['path']] = asset['sha256']
        records.append(prepare_preview(row, pools[article]))
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'samples.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    with (args.output/'samples.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['sample_id', 'article_id', 'figure_id', 'preview_disposition', 'flags', 'caption', 'clinical_candidate_count', 'author_diagnosis', 'diagnosis_status', 'training_ready']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({**{k: r[k] for k in ['sample_id', 'article_id', 'figure_id', 'preview_disposition', 'caption', 'training_ready']},
                             'flags': '; '.join(r['text_screening_flags']), 'clinical_candidate_count': len(r['clinical_context_candidates']),
                             'author_diagnosis': r['author_diagnosis']['name'], 'diagnosis_status': r['author_diagnosis']['status']})
    render_html(records, args.output/'preview.html')
    manifest = {'records': len(records), 'articles': len(originals),
                'dispositions': dict(Counter(r['preview_disposition'] for r in records)),
                'training_ready': 0, 'new_api_calls': 0, 'sft_exported': False,
                'source_sha256': sha(args.source), 'source_plan_sha256': sha(source_dir/'plan.json'),
                'sample_manifest_sha256': sha(target), 'html_sha256': sha(args.output/'preview.html'),
                'script_sha256': sha(Path(__file__)), 'image_files_sha256': image_files,
                'candidate_pool_sha256': {str(local_path(r['article_candidate_pool_path']).relative_to(ROOT)): r['article_candidate_pool_sha256'] for r in records},
                'note': 'Text signals are screening heuristics. No pixel verification, context assignment, clinical review, release approval or train/test split.'}
    (args.output/'summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in {'image_files_sha256', 'candidate_pool_sha256'}}, indent=2))
    print('Open: ' + str(args.output/'preview.html'))


if __name__ == '__main__':
    main()
