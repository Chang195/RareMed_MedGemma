"""Rule-routed, multi-query local evidence retrieval; preserves v1 and exact source spans."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
from medcpt_local import ROOT, MODELS, sha, corpus, encode, load
from medcpt_batch import rows, substantive
from evidence_rules import sentence_spans, route, attach_figure_sentences, ground_existing

QUERIES = {
    'presentation': 'What symptoms and complaints did the patient present with?',
    'medical_history': 'What was the patient past medical history before presentation?',
    'family_history': 'What diseases and ages of onset were reported in the patient family history?',
    'demographics': 'What was the age and sex of the patient at presentation?',
    'imaging_description': 'What abnormalities and anatomical locations were described on brain MRI?',
    'diagnostic_results': 'What diagnostic test results were reported for this patient?',
    'genetic_results': 'What genetic testing or biopsy results were reported for this patient?',
}


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def build_article_spans(chunks):
    case = [c for c in chunks if c.get('section') == 'CASE' and substantive(c)]
    chosen = case or [c for c in chunks if substantive(c) and c.get('section') not in
        {'FIG', 'TITLE', 'ABSTRACT', 'INTRO', 'DISCUSS', 'CONCL', 'AUTH_CONT', 'COMP_INT', 'REF'}]
    spans = {}
    for chunk in chosen:
        for span in sentence_spans(chunk):
            if len(span['text'].split()) >= 4:
                key = (span['passage_id'], span['start_in_passage'], span['end_in_passage'])
                spans[key] = span
    return list(spans.values()), 'CASE_section' if case else 'body_fallback'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--source', type=Path, default=ROOT/'data/releases/candidate-v0.4/candidates.jsonl')
    ap.add_argument('--output', type=Path, default=ROOT/'data/processed/medcpt_pilot10_v2_1')
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    if not 1 <= args.limit <= 370 or not 1 <= args.top_k <= 10:
        ap.error('Invalid limit or top-k')
    grouped = {}
    for row in rows(args.source):
        if row.get('proposed_linkage'):
            grouped.setdefault(row['article_id'], []).append(row)
    ids = list(grouped)[:args.limit]
    if not ids:
        raise ValueError('No linked image candidates')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {'version': 'medcpt-evidence-v2.1', 'articles': ids, 'queries': QUERIES, 'top_k': args.top_k,
            'source_sha256': sha(args.source), 'model_manifest_sha256': sha(MODELS/'manifest.json'),
            'script_hashes': {name: sha(ROOT/'scripts'/name) for name in
                              ['medcpt_batch_v2.py', 'evidence_rules.py', 'medcpt_batch.py', 'medcpt_local.py']},
            'article_input_sha256': {a: sha(ROOT/'data/model_inputs'/a/'input.json') for a in ids}}
    if (args.output/'plan.json').exists() and json.loads((args.output/'plan.json').read_text(encoding='utf-8')) != plan:
        raise ValueError('Plan changed; use a new --output directory')
    save(args.output/'plan.json', plan)
    print(json.dumps({'articles': len(ids), 'images': sum(len(grouped[a]) for a in ids), 'dry_run': args.dry_run,
                      'new_api_calls': 0, 'output': str(args.output)}, indent=2), flush=True)
    if args.dry_run:
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pending = [a for a in ids if not (args.output/(a+'.json')).exists()]
    article_spans, scopes = {}, {}
    if pending:
        for article in pending:
            article_spans[article], scopes[article] = build_article_spans(corpus(article))
        # Encode only sentences eligible for at least one field, retaining exact offsets.
        flat, offsets = [], {}
        for article in pending:
            spans = [s for s in article_spans[article] if any(route(s['text'], f)[0] for f in QUERIES)]
            offsets[article] = (len(flat), len(spans))
            article_spans[article] = spans
            flat.extend(spans)
        print(f'Encoding {len(flat)} rule-eligible sentences; {len(QUERIES)} queries...', flush=True)
        qvec = encode(list(QUERIES.values()), 'query', device)
        dvec = encode([['', s['text']] for s in flat], 'article', device) if flat else None
        tokenizer, model = load('cross', device)
        for index, article in enumerate(pending, 1):
            spans = article_spans[article]
            start, count = offsets[article]
            vectors = dvec[start:start+count] if count else None
            pools, audit = {}, {}
            for qi, (field, query) in enumerate(QUERIES.items()):
                eligible = [j for j, s in enumerate(spans) if route(s['text'], field)[0]]
                audit[field] = {'eligible_sentences': len(eligible), 'rejected_sentences': count-len(eligible),
                    'rejection_reasons': dict(Counter(reason for s in spans if not route(s['text'], field)[0]
                                                     for reason in route(s['text'], field)[1]))}
                scores = qvec[qi] @ vectors.T if count else None
                selected = sorted(eligible, key=lambda j: float(scores[j]), reverse=True)[:20]
                hits = []
                for batch_start in range(0, len(selected), 4):
                    batch_ids = selected[batch_start:batch_start+4]
                    batch = tokenizer([[query, spans[j]['text']] for j in batch_ids], padding=True,
                                      truncation='only_second', max_length=512, return_tensors='pt').to(device)
                    with torch.inference_mode():
                        logits = model(**batch).logits.squeeze(-1).cpu().tolist()
                    hits.extend({**spans[j], 'retrieval_inner_product': float(scores[j]), 'reranker_logit': score,
                                 'matched_rules': route(spans[j]['text'], field)[1],
                                 'status': 'rule_eligible_retrieved_candidate_not_verified'}
                                for j, score in zip(batch_ids, logits))
                pools[field] = sorted(hits, key=lambda h: h['reranker_logit'], reverse=True)[:args.top_k]
            result = {'article_id': article, 'scope': scopes[article], 'case_scope_verified': False,
                      'candidate_pools': pools, 'routing_audit': audit,
                      'input_sha256': plan['article_input_sha256'][article],
                      'status': 'retrieved' if count else 'no_rule_eligible_text',
                      'note': 'Lexical gates are conservative heuristics, not evidence validation or calibrated abstention. Empty means no rule-eligible result, not proof the fact is absent.'}
            save(args.output/(article+'.json'), result)
            print(f'[{index}/{len(pending)}] {article}: {count} eligible sentences', flush=True)
        del model
    output, comparison = [], []
    for article in ids:
        result = json.loads((args.output/(article+'.json')).read_text(encoding='utf-8'))
        if result['input_sha256'] != plan['article_input_sha256'][article]:
            raise ValueError('Cached input mismatch')
        original = json.loads((ROOT/'data/model_inputs'/article/'input.json').read_text(encoding='utf-8'))
        mapping = {p['passage_id']: p for p in original['passages']}
        for candidates in result['candidate_pools'].values():
            for span in candidates:
                if mapping[span['passage_id']]['text'][span['start_in_passage']:span['end_in_passage']] != span['text']:
                    raise ValueError('Retrieved span differs from source')
        previous = ROOT/'data/processed/medcpt_pilot10'/(article+'.json')
        old = json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else None
        for field, hits in result['candidate_pools'].items():
            oldfield = 'clinical_context' if field in {'presentation', 'medical_history', 'family_history', 'demographics'} else 'diagnostic_basis' if field in {'diagnostic_results', 'genetic_results'} else field
            oldhits = old['candidate_pools'].get(oldfield, []) if old else []
            comparison.append({'article_id': article, 'field': field, 'v1_top_text': oldhits[0]['text'] if oldhits else '',
                               'v2_top_text': hits[0]['text'] if hits else '', 'v2_candidates': len(hits)})
        for row in grouped[article]:
            link = row['proposed_linkage']
            output.append({'sample_id': row['sample_id'], 'article_id': article, 'figure_id': row['figure_id'],
                'images': row['assets'], 'caption': row['caption'], 'caption_evidence': row['caption_evidence'],
                'prior_linkage': link,
                'author_diagnosis': {'name': link['diagnosis_name'], 'status': link['diagnosis_status'],
                                    'evidence': ground_existing(link['diagnosis_evidence'], original['passages'])},
                'diagnostic_basis': {'value': link['diagnosis_basis'],
                                    'evidence': ground_existing(link['basis_evidence'], original['passages'])},
                'figure_reference_spans': attach_figure_sentences(row, original['passages']),
                'article_candidate_pool_path': str((args.output/(article+'.json')).relative_to(ROOT)),
                'article_candidate_pool_sha256': sha(args.output/(article+'.json')),
                'scope': result['scope'], 'case_scope_verified': False, 'training_ready': False,
                'human_review': 'skipped_by_user', 'semantic_validation': 'not_performed',
                'source_url': row['source_url'], 'license': row['license']})
    target = args.output/'evidence_records.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output), encoding='utf-8')
    with (args.output/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    summary = {'articles': len(ids), 'image_records': len(output), 'new_api_calls': 0, 'training_ready': 0,
               'empty_field_pools': sum(r['v2_candidates'] == 0 for r in comparison),
               'reused_basis_quotes': sum(len(r['diagnostic_basis']['evidence']) for r in output),
               'figure_reference_spans': sum(len(r['figure_reference_spans']) for r in output),
               'records_sha256': sha(target), 'note': 'Counts are coverage, not accuracy. No human or independent model review.'}
    save(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
