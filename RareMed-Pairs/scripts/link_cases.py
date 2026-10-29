"""Prepare text-only model inputs or validate a manually supplied model response.

No network calls, API credentials, or model predictions are made here.
The validator checks structure and source quotes, NOT clinical truth or entailment.
"""
import argparse
import json
from pathlib import Path
import re
from collect import ROOT, dump, digest

VERSION = 'case-linkage-v0.1'
FIGURE_FIELDS = {'figure_id', 'case_label', 'link_status', 'diagnosis_name', 'diagnosis_status',
                'diagnosis_basis', 'link_evidence', 'diagnosis_evidence', 'basis_evidence', 'uncertainties'}
LINK_STATES = {'explicit', 'contextual', 'unresolved', 'multiple_cases'}
DIAG_STATES = {'author_reported_confirmed', 'suspected', 'differential', 'excluded', 'unclear'}


def prepare(figures_path):
    rows = [json.loads(line) for line in figures_path.read_text(encoding='utf-8').split('\n') if line.strip()]
    grouped = {}
    for row in rows:
        grouped.setdefault(row['article_id'], []).append(row)
    if not grouped:
        raise ValueError('No figure candidates to prepare')
    schema_path = ROOT / 'configs/linkage_schema.json'
    prompt_path = ROOT / 'configs/linkage_prompt.txt'
    template = prompt_path.read_text(encoding='utf-8')
    summary = []
    for pmcid, figures in grouped.items():
        if not re.fullmatch(r'PMC\d+', pmcid):
            raise ValueError('Invalid article ID')
        article_path = ROOT / 'data/processed/articles' / (pmcid+'.json')
        article = json.loads(article_path.read_text(encoding='utf-8'))
        if article['article_id'] != pmcid:
            raise ValueError('Article identity mismatch')
        passages = []
        for idx, p in enumerate(article['passages']):
            # Exclude bibliography and publication metadata; keep exact body text.
            if p['metadata'].get('section_type') == 'REF':
                continue
            passages.append({'passage_id': f'p{idx:04d}', 'section': p['metadata'].get('section_type'),
                             'source_offset': p['offset'], 'text': p['text']})
        request = {'article_id': pmcid, 'passages': passages,
                   'candidate_figures': [{'figure_id': r['figure_id'], 'caption': r['caption'],
                                         'caption_source_offset': r['caption_evidence']['offset']}
                                        for r in figures]}
        folder = ROOT / 'data/model_inputs' / pmcid
        folder.mkdir(parents=True, exist_ok=True)
        dump(folder / 'input.json', request)
        prompt = template.replace('{{SCHEMA}}', schema_path.read_text(encoding='utf-8')).replace(
            '{{ARTICLE_INPUT}}', json.dumps(request, ensure_ascii=False, indent=2))
        (folder / 'prompt.txt').write_text(prompt, encoding='utf-8')
        # Deliberately unresolved template, not a model or manual prediction.
        response_template = {'article_id': pmcid, 'figures': [
            {'figure_id': r['figure_id'], 'case_label': None, 'link_status': 'unresolved',
             'diagnosis_name': None, 'diagnosis_status': 'unclear', 'diagnosis_basis': None,
             'link_evidence': [], 'diagnosis_evidence': [], 'basis_evidence': [],
             'uncertainties': ['UNFILLED TEMPLATE: no model inference has been run.']} for r in figures]}
        dump(folder / 'response_template.json', response_template)
        receipt = {'pipeline_version': VERSION, 'article_id': pmcid, 'figure_candidates': len(figures),
                   'passages': len(passages), 'prompt_characters': len(prompt),
                   'input_sha256': digest(folder/'input.json'), 'article_sha256': digest(article_path),
                   'figure_manifest_sha256': digest(figures_path), 'prompt_sha256': digest(folder/'prompt.txt'),
                   'schema_sha256': digest(schema_path), 'model_called': False,
                   'privacy_status': 'published_case_text_not_fully_deidentified; local_preparation_only',
                   'note': 'Source text can contain patient attributes. Review before any external API transfer.'}
        dump(folder / 'preparation_receipt.json', receipt)
        summary.append(receipt)
    dump(ROOT/'reports/linkage_preparation_receipt.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate(response, request):
    errors, anchored = [], []
    if not isinstance(response, dict) or set(response) != {'article_id', 'figures'}:
        return ['Root must contain exactly article_id and figures'], []
    if response['article_id'] != request['article_id']:
        errors.append('Article ID mismatch')
    if not isinstance(response['figures'], list):
        return errors + ['figures must be an array'], []
    expected = {f['figure_id'] for f in request['candidate_figures']}
    passages = {p['passage_id']: p for p in request['passages']}
    seen = []
    for idx, row in enumerate(response['figures']):
        prefix = f'figures[{idx}]'
        if not isinstance(row, dict) or set(row) != FIGURE_FIELDS:
            errors.append(prefix+': invalid or missing fields')
            continue
        fid = row['figure_id']
        if not isinstance(fid, str):
            errors.append(prefix+': figure_id must be a string')
            continue
        seen.append(fid)
        if fid not in expected:
            errors.append(prefix+': unknown figure ID')
        if not isinstance(row['link_status'], str) or not isinstance(row['diagnosis_status'], str):
            errors.append(prefix+': states must be strings')
            continue
        if row['link_status'] not in LINK_STATES or row['diagnosis_status'] not in DIAG_STATES:
            errors.append(prefix+': invalid state')
        for field in ('case_label', 'diagnosis_name', 'diagnosis_basis'):
            if row[field] is not None and (not isinstance(row[field], str) or not row[field].strip()):
                errors.append(prefix+': '+field+' must be nonempty text or null')
        if not isinstance(row['uncertainties'], list) or not all(isinstance(v, str) for v in row['uncertainties']):
            errors.append(prefix+': uncertainties must be an array of strings')
        for field in ('link_evidence', 'diagnosis_evidence', 'basis_evidence'):
            evidence = row[field]
            if not isinstance(evidence, list):
                errors.append(prefix+': '+field+' must be an array')
                continue
            for item in evidence:
                if not isinstance(item, dict) or set(item) != {'passage_id', 'quote'}:
                    errors.append(prefix+': malformed evidence')
                    continue
                pid, quote = item['passage_id'], item['quote']
                if not isinstance(pid, str) or pid not in passages or not isinstance(quote, str) or not quote.strip():
                    errors.append(prefix+': invalid passage or empty quote')
                    continue
                passage = passages[pid]
                start = passage['text'].find(quote)
                if start < 0:
                    errors.append(prefix+': quote not found verbatim in '+pid)
                else:
                    anchored.append({'figure_id': fid, 'evidence_role': field, 'passage_id': pid,
                        'quote': quote, 'passage_source_offset': passage['source_offset'],
                        'quote_start_in_passage': start, 'quote_end_in_passage': start+len(quote)})
        if row['link_status'] in {'explicit', 'contextual'} and (not row['link_evidence'] or not row['case_label']):
            errors.append(prefix+': supported linkage requires case label and evidence')
        if row['diagnosis_name'] is not None and not row['diagnosis_evidence']:
            errors.append(prefix+': diagnosis requires evidence')
        if row['diagnosis_status'] != 'unclear' and row['diagnosis_name'] is None:
            errors.append(prefix+': diagnosis state requires a named diagnosis')
        if row['diagnosis_basis'] is not None and not row['basis_evidence']:
            errors.append(prefix+': diagnostic basis requires evidence')
        if row['link_status'] in {'unresolved', 'multiple_cases'} and any(
                row[k] is not None for k in ('case_label', 'diagnosis_name', 'diagnosis_basis')):
            errors.append(prefix+': unresolved/multiple-case image cannot receive one case diagnosis')
        if row['link_status'] in {'contextual', 'unresolved', 'multiple_cases'} and not row['uncertainties']:
            errors.append(prefix+': uncertain linkage requires explanation')
    if len(seen) != len(set(seen)):
        errors.append('Duplicate figure ID')
    if set(seen) != expected:
        errors.append('Figure coverage mismatch')
    return errors, anchored


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--figures', type=Path, default=ROOT/'data/processed/fulltext_pilot_3_figures.jsonl')
    check = sub.add_parser('validate')
    check.add_argument('--article', required=True)
    check.add_argument('--response', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare(args.figures)
        return
    if not re.fullmatch(r'PMC\d+', args.article):
        raise ValueError('Invalid article ID')
    request_path = ROOT/'data/model_inputs'/args.article/'input.json'
    request = json.loads(request_path.read_text(encoding='utf-8'))
    try:
        response_text = args.response.read_text(encoding='utf-8-sig')
    except FileNotFoundError:
        ap.exit(1, 'Response file not found. Check the path and save the file in VS Code.\n')
    if not response_text.strip():
        ap.exit(1, 'Response file is empty. Paste the JSON into response.json and press Ctrl+S before validating.\n')
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        ap.exit(1, f'Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}. Save only the JSON object, without Markdown fences.\n')
    errors, anchors = validate(response, request)
    result = {'article_id': args.article, 'structural_and_quote_checks_passed': not errors,
              'errors': errors, 'anchored_evidence': anchors, 'training_ready': False,
              'semantic_validation': 'not_performed', 'review_status': 'human_review_required',
              'input_sha256': digest(request_path), 'response_sha256': digest(args.response)}
    dump(ROOT/'reports'/(args.article+'_linkage_validation.json'), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
