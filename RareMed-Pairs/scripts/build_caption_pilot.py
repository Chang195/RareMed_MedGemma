"""Export a source-extractive MedGemma-format smoke-test dataset; not a benchmark."""
import argparse
import base64
from collections import Counter
from html import escape
import json
from pathlib import Path
import shutil
from medcpt_local import ROOT, sha

SOURCE = ROOT/'reports/sample_preview_v1/samples.jsonl'
CONFIG = ROOT/'configs/caption_pilot_v1.json'
OUTPUT = ROOT/'data/datasets/caption-pilot-v0.1'


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def within(root, value):
    path = (root/value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path outside intended dataset/project')
    return path


def target_span(selection, row, article):
    target = selection['target_text']
    if not isinstance(target, str) or not target.strip() or target not in row['caption']:
        raise ValueError('Target must be copied verbatim from this figure caption')
    passage = next(p for p in article['passages'] if p['passage_id'] == selection['target_passage_id'])
    if target not in passage['text']:
        raise ValueError('Target does not occur in the cited source passage')
    if row['prior_linkage']['figure_id'] != row['figure_id']:
        raise ValueError('Figure identity mismatch')
    a = passage['text'].index(target)
    return {'passage_id': passage['passage_id'], 'text': target,
            'start_in_passage': a, 'end_in_passage': a+len(target),
            'passage_source_offset': passage['source_offset'], 'caption_start': row['caption'].index(target)}


def make_messages(question, target):
    if not question.strip() or not target.strip():
        raise ValueError('Question and target required')
    return [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': target}]}]


def load_examples(folder):
    """Yield rows in the official example's image + messages layout, with PIL images."""
    from PIL import Image
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if sha(folder/'sft_pilot.jsonl') != manifest['dataset_sha256']:
        raise ValueError('Dataset manifest changed')
    for row in read_rows(folder/'sft_pilot.jsonl'):
        path = within(folder, row['image'])
        if sha(path) != row['metadata']['image_sha256']:
            raise ValueError('Image changed')
        messages = row['messages']
        if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
            raise ValueError('Unexpected message roles')
        if messages[0]['content'] != [{'type': 'image'}, {'type': 'text', 'text': row['metadata']['question']}]:
            raise ValueError('Unexpected input; possible target or diagnosis leakage')
        if messages[1]['content'] != [{'type': 'text', 'text': row['metadata']['target_evidence']['text']}]:
            raise ValueError('Target changed')
        with Image.open(path) as image:
            rgb = image.convert('RGB')
        yield {**row, 'image': rgb}


def build(source, config_path, output):
    from PIL import Image
    config = json.loads(config_path.read_text(encoding='utf-8'))
    source_summary = json.loads((source.parent/'summary.json').read_text(encoding='utf-8'))
    if source_summary['sample_manifest_sha256'] != sha(source):
        raise ValueError('Preview source changed')
    rows = read_rows(source)
    mapping = {r['sample_id']: r for r in rows}
    if len(mapping) != len(rows):
        raise ValueError('Duplicate source sample ID')
    selection_ids = [r['sample_id'] for r in config['include'] + config['hold']]
    expected = {r['sample_id'] for r in rows if r['preview_disposition'] == 'single_case_author_confirmed_candidate'}
    if len(selection_ids) != len(set(selection_ids)) or set(selection_ids) != expected:
        raise ValueError('Selection must account for every confirmed single-case preview candidate exactly once')
    identity = {'source_sha256': sha(source), 'config_sha256': sha(config_path), 'script_sha256': sha(Path(__file__))}
    if (output/'manifest.json').exists():
        old = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if any(old[k] != value for k, value in identity.items()):
            raise ValueError('Existing pilot differs; choose a new --output directory')
        check(output)
        return
    (output/'images').mkdir(parents=True, exist_ok=True)
    records, articles, cards, article_hashes = [], {}, [], {}
    for selection in config['include']:
        row = mapping[selection['sample_id']]
        if row['preview_disposition'] != 'single_case_author_confirmed_candidate' or len(row['images']) != 1:
            raise ValueError('Selection is not a single-case candidate with one source image')
        article_id = row['article_id']
        if article_id not in articles:
            path = ROOT/'data/model_inputs'/article_id/'input.json'
            expected_hash = json.loads((ROOT/'data/processed/medcpt_pilot10_v2_1/plan.json').read_text(encoding='utf-8'))['article_input_sha256'][article_id]
            if sha(path) != expected_hash:
                raise ValueError('Article source changed')
            articles[article_id] = json.loads(path.read_text(encoding='utf-8'))
            article_hashes[article_id] = sha(path)
        evidence = target_span(selection, row, articles[article_id])
        asset = row['images'][0]
        raw = within(ROOT, asset['path'])
        if sha(raw) != asset['sha256']:
            raise ValueError('Original image changed')
        with Image.open(raw) as image:
            image.verify()
        with Image.open(raw) as image:
            extension = '.jpg' if image.format == 'JPEG' else '.png' if image.format == 'PNG' else None
            if extension is None:
                raise ValueError('Unsupported original image format')
            size = list(image.size)
        filename = row['sample_id'] + extension
        if Path(filename).name != filename:
            raise ValueError('Invalid sample filename')
        destination = output/'images'/filename
        shutil.copyfile(raw, destination)  # Preserve original bytes, labels and arrows.
        record = {'sample_id': row['sample_id'], 'image': 'images/' + filename,
                  'messages': make_messages(config['question'], selection['target_text']),
                  'metadata': {'article_id': article_id, 'figure_id': row['figure_id'],
                      'source_url': row['source_url'], 'license': row['license'],
                      'original_caption': row['caption'], 'question': config['question'],
                      'target_evidence': evidence, 'target_method': 'verbatim_caption_substring',
                      'selection_note': selection['selection_note'], 'image_sha256': asset['sha256'], 'image_size': size,
                      'article_input_sha256': article_hashes[article_id], 'source_image_path': asset['path'],
                      'split_group_id': article_id, 'split': 'unassigned',
                      'group_note': 'Conservative same-article grouping; these two images belong to one source-reported case.',
                      'annotation_status': 'original_arrows_and_orientation_labels_retained',
                      'clinical_review': 'not_performed', 'deidentification_review': 'not_completed',
                      'figure_level_visual_inspection': 'assistant_inspected_for_layout_not_clinical_correctness',
                      'training_ready': False, 'purpose': config['purpose']}}
        records.append(record)
        uri = 'data:image/' + ('jpeg' if extension == '.jpg' else 'png') + ';base64,' + base64.b64encode(raw.read_bytes()).decode()
        cards.append('<article><h2>' + escape(row['sample_id']) + '</h2><div class="grid"><img src="' + uri + '"><div><h3>用户输入</h3><p>图片＋固定问题；不含图注、病例诊断或基因结果。</p><pre>' +
                     escape(config['question']) + '</pre><h3>assistant 目标文本</h3><pre>' + escape(selection['target_text']) +
                     '</pre><h3>原始完整图注</h3><pre>' + escape(row['caption']) + '</pre><p>原文：' + escape(evidence['passage_id']) +
                     f" [{evidence['start_in_passage']}:{evidence['end_in_passage']}]</p><p>保留原箭头；未做临床审核。两张图同组，不划分测试集。</p></div></div></article>")
    target = output/'sft_pilot.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    (output/'held.json').write_text(json.dumps(config['hold'], ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'preview.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Caption pilot</title>
<style>body{font:16px/1.7 'Microsoft YaHei',Arial;background:#f2f5f9;margin:25px;color:#203044}article{background:white;padding:24px;margin:20px auto;max-width:1300px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{max-width:100%;max-height:700px;object-fit:contain}pre{white-space:pre-wrap;background:#f0f4f8;padding:14px;font:14px/1.7 Consolas,monospace}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>MedGemma 图像描述 · 格式验证样本</h1><p>2张图片，来自同一病例。仅供加载和训练流程冒烟验证，不能用于效果评估。</p>''' + ''.join(cards) + '</html>', encoding='utf-8')
    manifest = {**identity, 'version': config['version'], 'purpose': config['purpose'],
                'examples': len(records), 'source_article_groups': len(articles),
                'source_reported_case_groups': len({(mapping[s['sample_id']]['article_id'], mapping[s['sample_id']]['prior_linkage'].get('case_label')) for s in config['include']}), 'held_candidates': len(config['hold']),
                'split': 'unassigned', 'training_ready': 0, 'new_api_calls': 0,
                'dataset_sha256': sha(target), 'article_input_sha256': article_hashes,
                'limitations': ['Not a clinical benchmark or generalization evaluation.',
                                'Verbatim source captions are weak supervision, not independently verified findings.',
                                'Annotated images retain arrows; this pilot does not test unannotated MRI interpretation.',
                                'No diagnosis training task, train/test split or model-tokenizer forward pass has been performed.']}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output/'README.md').write_text('# MedGemma caption-format pilot\n\n'
        'Two annotated MRI figures from one source-reported case. For format/image-loading smoke tests only.\n'
        'No independent clinical labels, evaluation split or diagnosis-training claims.\n'
        'See manifest.json and per-record source/rights metadata. Original image bytes are retained.\n'
        'image is a dataset-relative path; load_examples() converts it to a PIL RGB image for the official-style image + messages layout.\n', encoding='utf-8')
    check(output)


def check(output):
    examples = list(load_examples(output))
    report = {'examples_loaded': len(examples), 'image_shapes': [list(e['image'].size) for e in examples],
              'groups': dict(Counter(e['metadata']['split_group_id'] for e in examples)),
              'roles_validated': True, 'image_bytes_checked': True, 'target_source_spans_recorded': True,
              'tokenizer_or_model_forward_tested': False, 'training_started': False,
              'output': str(output)}
    (output/'loading_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--config', type=Path, default=CONFIG)
    ap.add_argument('--output', type=Path, default=OUTPUT)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    if args.check_only:
        check(args.output.resolve())
    else:
        build(args.source.resolve(), args.config.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
