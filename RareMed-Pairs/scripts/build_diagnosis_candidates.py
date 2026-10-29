"""Label every collected figure and export source-reported diagnosis candidates. No API calls."""
import csv
import json
from collections import Counter
from pathlib import Path
from html import escape
from inventory_dataset import read_rows, screen
from medcpt_local import ROOT, sha


def classify(row, link):
    if not link:
        return 'held_no_validated_extraction'
    queue, _ = screen({**row, 'proposed_linkage': link}, row.get('assets', []))
    if not queue.startswith('candidate_'):
        return queue
    if link.get('diagnosis_status') != 'author_reported_confirmed':
        return 'held_diagnosis_not_confirmed'
    if not link.get('diagnosis_name'):
        return 'held_missing_diagnosis_name'
    return 'single_case_author_confirmed_mri_candidate'


def exact_evidence(items, passages):
    result = []
    for item in items:
        p = passages.get(item['passage_id'])
        quote = item['quote']
        if not p or not quote or quote not in p['text']:
            raise ValueError('Evidence does not match source text')
        start = p['text'].index(quote)
        result.append({'passage_id': item['passage_id'], 'text': quote,
                       'start_in_passage': start, 'end_in_passage': start + len(quote),
                       'section': p['section'], 'passage_source_offset': p['source_offset']})
    return result


def main():
    source = ROOT/'data/processed/collection_v3_figures.jsonl'
    links_path = ROOT/'data/processed/deepseek_collection_v3_links.jsonl'
    output = ROOT/'data/datasets/diagnosis_candidates_v1'
    output.mkdir(parents=True, exist_ok=True)
    links = {(r['article_id'], r['figure_id']): r for r in read_rows(links_path)}
    records, selected = [], []
    for raw in read_rows(source):
        proposal = links.get((raw['article_id'], raw['figure_id']))
        link = proposal['proposed_linkage'] if proposal else None
        status = classify(raw, link)
        row = {**raw, 'proposed_linkage': link, 'selection_status': status,
               'label_origin': 'model_extracted_author_report_not_independently_verified',
               'split': 'unassigned', 'training_ready': False,
               'deduplication_status': 'not_applied_in_this_export',
               'leakage_check_status': 'not_applied_in_this_export'}
        if proposal:
            path = ROOT/'data/model_inputs'/raw['article_id']/'input.json'
            if sha(path) != proposal['input_sha256']:
                raise ValueError('Extraction source hash changed')
            payload = json.loads(path.read_text(encoding='utf-8'))
            passages = {p['passage_id']: p for p in payload['passages']}
            row['source_input_sha256'] = proposal['input_sha256']
            row['extraction_response_sha256'] = proposal['response_sha256']
            row['source_text'] = {
                'image_description': {'text': raw['caption'], 'source': raw['caption_evidence']},
                'figure_case_link_evidence': exact_evidence(link.get('link_evidence', []), passages),
                'author_diagnosis_evidence': exact_evidence(link.get('diagnosis_evidence', []), passages),
                'diagnostic_basis_evidence': exact_evidence(link.get('basis_evidence', []), passages),
                'clinical_context_assigned': None,
                'article_case_section_candidates': [p for p in payload['passages'] if p['section'] == 'CASE'],
                'context_status': 'article_level_source_only_not_assigned_to_image_or_ready_as_model_input'}
        records.append(row)
        if status == 'single_case_author_confirmed_mri_candidate':
            for asset in row['assets']:
                if sha(ROOT/asset['path']) != asset['sha256']:
                    raise ValueError('Image checksum mismatch')
            selected.append(row)
    for name, rows in [('all_figures.jsonl', records), ('confirmed_mri_candidates.jsonl', selected)]:
        (output/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    columns = ['sample_id', 'article_id', 'figure_id', 'selection_status', 'diagnosis_name', 'diagnosis_status', 'link_status', 'image_paths', 'caption']
    with (output/'all_figures.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            link = row['proposed_linkage'] or {}
            writer.writerow({**{k: row[k] for k in columns[:4]},
                **{k: link.get(k) for k in columns[4:7]},
                'image_paths': '; '.join(a['path'] for a in row['assets']), 'caption': row['caption']})
    summary = {'figures': len(records), 'selection_counts': dict(Counter(r['selection_status'] for r in records)),
               'selected': len(selected), 'selected_articles': len({r['article_id'] for r in selected}),
               'selected_diagnosis_names': dict(Counter(r['proposed_linkage']['diagnosis_name'] for r in selected)),
               'new_api_calls': 0, 'training_ready': 0, 'source_sha256': sha(source),
               'links_sha256': sha(links_path), 'script_sha256': sha(Path(__file__)),
               'records_sha256': sha(output/'all_figures.jsonl'),
               'note': 'Missing/failed extractions remain held, never inferred negative. MRI/nonpatient screening is text heuristic. No deduplication or split performed.'}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    cards = []
    for r in selected:
        images = ''.join('<img loading="lazy" style="max-width:420px;max-height:400px" src="'+escape((ROOT/a['path']).as_uri())+'">' for a in r['assets'])
        cards.append('<article><h2>'+escape(r['sample_id'])+'</h2><b>'+escape(r['proposed_linkage']['diagnosis_name'])+'</b>'+images+'<pre>'+escape(json.dumps(r['source_text'], ensure_ascii=False, indent=2))+'</pre></article>')
    (output/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>全量确诊 MRI 候选</title><style>body{font-family:Arial;margin:30px}article{border-bottom:2px solid #999;padding:20px}pre{white-space:pre-wrap}</style><h1>作者报告确诊：脑 MRI 候选</h1><p>已有抽取结果的逐图整理；未独立审核。临床段落仍为文章级候选，诊断证据不可直接当作诊断任务输入。</p>'+''.join(cards), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'selected_diagnosis_names'}, indent=2))


if __name__ == '__main__':
    main()
