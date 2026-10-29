"""Build local article-scoped evidence candidates; no automatic case attribution or SFT."""
import argparse
from collections import Counter
import csv
import gc
import json
import os
from pathlib import Path
import re
from medcpt_local import ROOT, MODELS, sha, corpus, encode, load

QUERIES = {
    'clinical_context': 'patient presentation symptoms clinical history family history disease onset',
    'imaging_description': 'brain MRI findings white matter lesions location signal abnormalities',
    'diagnostic_basis': 'patient diagnosis confirmation genetic testing NOTCH3 pathogenic variant biopsy results',
}
VERSION = 'medcpt-evidence-v1'


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def substantive(row):
    text = row['text'].strip()
    # Conservative title filter, not a relevance classifier.
    if re.fullmatch(r'(?:case\s+(?:presentation|report|\d+)|discussion|conclusions?|introduction|clinical\s+(?:history|presentation)|results|methods)', text, re.I):
        return False
    return len(re.findall(r'\b\w+\b', text)) >= 8


def figure_mentions(row, passages):
    # Only infer common numeric figure IDs; do not guess opaque publisher IDs.
    match = re.match(r'^(?:fig|f)(\d+)(?:$|[-_.])', row['figure_id'], re.I)
    if not match:
        return []
    number = match[1]
    pattern = re.compile(r'\b(?:fig(?:ure)?s?\.?)[ \t]*' + number + r'(?:[A-Za-z])?\b', re.I)
    return [p for p in passages if p.get('section') != 'FIG' and pattern.search(p['text'])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--source', type=Path, default=ROOT/'data/releases/candidate-v0.4/candidates.jsonl')
    ap.add_argument('--output', type=Path, default=ROOT/'data/processed/medcpt_pilot10')
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.limit <= 370 or not 1 <= args.top_k <= 10:
        ap.error('limit must be 1..370; top-k 1..10')
    source = rows(args.source)
    grouped = {}
    for row in source:
        if row.get('proposed_linkage'):
            grouped.setdefault(row['article_id'], []).append(row)
    ids = list(grouped)[:args.limit]
    if not ids:
        ap.error('No articles with prior linkage proposals')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'version': VERSION, 'articles': ids, 'source_sha256': sha(args.source),
                'model_manifest_sha256': sha(MODELS/'manifest.json'), 'queries': QUERIES,
                'top_k': args.top_k, 'script_sha256': sha(Path(__file__)),
                'retrieval_script_sha256': sha(ROOT/'scripts/medcpt_local.py'),
                'article_input_sha256': {a: sha(ROOT/'data/model_inputs'/a/'input.json') for a in ids}}
    planpath = args.output/'plan.json'
    if planpath.exists() and json.loads(planpath.read_text(encoding='utf-8')) != identity:
        raise ValueError('Output belongs to a different plan; choose a new --output folder')
    planpath.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    print(json.dumps({'selected_articles': len(ids), 'images_with_prior_proposals': sum(len(grouped[a]) for a in ids),
                      'new_api_calls': 0, 'dry_run': args.dry_run, 'output': str(args.output)}, indent=2), flush=True)
    if args.dry_run:
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pending = [a for a in ids if not (args.output/(a + '.json')).exists()]
    if pending:
        article_chunks = {}
        for article in pending:
            all_chunks = corpus(article)
            case_chunks = [p for p in all_chunks if p.get('section') == 'CASE' and substantive(p)]
            # No CASE section: retain body evidence candidates, never certify patient scope.
            candidates = case_chunks or [p for p in all_chunks if substantive(p) and
                p.get('section') not in {'FIG', 'TITLE', 'ABSTRACT', 'INTRO', 'DISCUSS', 'CONCL', 'AUTH_CONT', 'COMP_INT', 'REF'}]
            article_chunks[article] = candidates
        print('Encoding queries and source paragraphs...', flush=True)
        qvec = encode(list(QUERIES.values()), 'query', device)
        flat = [p for a in pending for p in article_chunks[a]]
        dvec = encode([['', p['text']] for p in flat], 'article', device) if flat else None
        tokenizer, model = load('cross', device)
        offset = 0
        for index, article in enumerate(pending, 1):
            passages = article_chunks[article]
            vectors = dvec[offset:offset+len(passages)] if passages else None
            offset += len(passages)
            pools = {}
            for qi, (field, query) in enumerate(QUERIES.items()):
                scores = (qvec[qi] @ vectors.T) if passages else []
                selected = torch.argsort(scores, descending=True)[:20].tolist() if passages else []
                hits = []
                for start in range(0, len(selected), 4):
                    batch_ids = selected[start:start+4]
                    encoded = tokenizer([[query, passages[j]['text']] for j in batch_ids], padding=True,
                                        truncation='only_second', max_length=512, return_tensors='pt').to(device)
                    with torch.inference_mode():
                        logits = model(**encoded).logits.squeeze(-1).cpu().tolist()
                    hits.extend({**passages[j], 'retrieval_inner_product': float(scores[j]),
                                 'reranker_logit': score, 'status': 'retrieved_candidate_not_verified'}
                                for j, score in zip(batch_ids, logits))
                pools[field] = sorted(hits, key=lambda p: p['reranker_logit'], reverse=True)[:args.top_k]
            result = {'article_id': article, 'scope': 'CASE_section' if passages and all(p.get('section') == 'CASE' for p in passages) else 'body_fallback',
                      'case_scope_verified': False, 'candidate_pools': pools,
                      'input_sha256': identity['article_input_sha256'][article],
                      'status': 'retrieved' if passages else 'no_eligible_body_text',
                      'note': 'Top-k candidates are not accepted evidence; no calibrated threshold or no-answer classifier.'}
            (args.output/(article + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'[{index}/{len(pending)}] {article}: {len(passages)} chunks', flush=True)
        del model
        gc.collect()
    output = []
    for article in ids:
        result = json.loads((args.output/(article + '.json')).read_text(encoding='utf-8'))
        original = json.loads((ROOT/'data/model_inputs'/article/'input.json').read_text(encoding='utf-8'))
        for row in grouped[article]:
            link = row['proposed_linkage']
            output.append({'sample_id': row['sample_id'], 'article_id': article, 'figure_id': row['figure_id'],
                'images': row['assets'], 'caption': row['caption'], 'caption_evidence': row['caption_evidence'],
                'prior_linkage': link, 'author_diagnosis': {'name': link['diagnosis_name'], 'status': link['diagnosis_status'],
                    'evidence': link['diagnosis_evidence']},
                'diagnostic_basis': {'value': link['diagnosis_basis'], 'evidence': link['basis_evidence']},
                'figure_reference_passages': figure_mentions(row, original['passages']),
                'article_candidate_pool_path': str((args.output/(article + '.json')).relative_to(ROOT)),
                'article_candidate_pool_sha256': sha(args.output/(article + '.json')),
                'scope': result['scope'], 'case_scope_verified': False, 'training_ready': False,
                'human_review': 'skipped_by_user', 'semantic_validation': 'not_performed',
                'source_url': row['source_url'], 'license': row['license']})
    (args.output/'evidence_records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output), encoding='utf-8')
    with (args.output/'overview.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['sample_id', 'article_id', 'figure_id', 'caption', 'diagnosis', 'diagnosis_status', 'link_status', 'scope', 'figure_reference_passages']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output:
            writer.writerow({**{f: row[f] for f in ['sample_id', 'article_id', 'figure_id', 'caption', 'scope']},
                'diagnosis': row['author_diagnosis']['name'], 'diagnosis_status': row['author_diagnosis']['status'],
                'link_status': row['prior_linkage']['link_status'], 'figure_reference_passages': len(row['figure_reference_passages'])})
    summary = {'articles': len(ids), 'image_records': len(output), 'scope_counts': dict(Counter(r['scope'] for r in output)),
               'new_api_calls': 0, 'training_ready': 0, 'records_sha256': sha(args.output/'evidence_records.jsonl')}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
