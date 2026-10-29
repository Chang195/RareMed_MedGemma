"""Offline, evidence-logged metadata triage; not final clinical eligibility."""
import argparse
from collections import Counter
import csv
import hashlib
from html import escape, unescape
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from collect import ROOT, dump, digest

RULE_VERSION = 'metadata-review-exclusion-v0.2'
PATTERNS = {
    'cadasil': r'\bCADASIL\b|cerebral autosomal dominant arteriopathy',
    'sporadic_csvd': r'\bsporadic\b.{0,100}(?:small[ -]vessel|CSVD|SVD)|(?:small[ -]vessel|CSVD|SVD).{0,100}\bsporadic\b',
    'mri': r'\bMRI\b|magnetic resonance',
    'case': r'\bcase report\b|\bcase series\b|\breport (?:a|the) case\b',
    'review': r'\breview\b|meta[ -]analysis'
}


def plain(text):
    return re.sub(r'\s+', ' ', unescape(re.sub('<[^>]+>', ' ', text or ''))).strip()


def triage(item):
    fields = {'title': plain(item.get('title')), 'abstract': plain(item.get('abstractText')),
              'publication_type': '; '.join(item.get('pubTypeList', {}).get('pubType', []))}
    evidence = {}
    for name, pattern in PATTERNS.items():
        hits = []
        for field, text in fields.items():
            match = re.search(pattern, text, re.I)
            if match:
                hits.append({'field': field, 'matched': match.group(),
                             'context': text[max(0, match.start()-60):match.end()+100]})
        evidence[name] = hits
    # Publication metadata can mark Case Reports even when title/abstract do not.
    if re.search(r'case[ -]reports?', fields['publication_type'], re.I) and not evidence['case']:
        evidence['case'] = [{'field': 'publication_type', 'matched': fields['publication_type'],
                             'context': fields['publication_type']}]
    if evidence['review']:
        queue, reason = 'excluded_review', 'User policy: review/meta-analysis match in title, abstract or publication type.'
    else:
        queue, reason = 'fulltext_required', 'Check disease and MRI signals in full text; missing title/abstract signals do not exclude.'
    return {'article_id': item['pmcid'], 'title': fields['title'], 'year': item.get('pubYear'),
            'doi': item.get('doi'), 'publication_types': fields['publication_type'],
            'queue': queue, 'reason': reason, 'evidence': evidence,
            'topic_mentions': [k for k in ('cadasil', 'sporadic_csvd') if evidence[k]],
            'topic_note': 'Mention signals only; not patient diagnoses or rare/common cohort labels.',
            'article_license_hint': item.get('license'), 'figure_eligibility': 'not_checked',
            'final_eligibility': 'excluded_review_policy' if queue=='excluded_review' else 'pending_fulltext_and_figure_review',
            'training_ready': False, 'rule_version': RULE_VERSION}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/discovery_200.json')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    run_id = cfg['run_id']
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    manifest = ROOT / 'data/manifests' / (run_id + '_sources.json')
    ids = {r['article_id'] for r in json.loads(manifest.read_text(encoding='utf-8'))}
    found, inputs = {}, {}
    for query in cfg['queries']:
        key = hashlib.sha256(query.encode()).hexdigest()[:16]
        path = ROOT / 'data/raw/search' / (key + '.json')
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode({
            'query': query, 'format': 'json', 'pageSize': cfg['page_size'], 'resultType': 'core'})
        meta=path.with_name(path.name+'.download.json')
        if not meta.exists() or json.loads(meta.read_text(encoding='utf-8')).get('url') != url:
            path=ROOT/'data/raw/search'/(hashlib.sha256(url.encode()).hexdigest()[:16]+'.json')
        receipt=json.loads(path.with_name(path.name+'.download.json').read_text(encoding='utf-8'))
        if receipt.get('url')!=url or receipt.get('sha256')!=digest(path):
            raise ValueError('Search source URL or hash mismatch')
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        for item in data['resultList']['result']:
            if item.get('pmcid') in ids:
                found[item['pmcid']] = item
    if set(found) != ids:
        raise ValueError('Search metadata does not cover every manifest article')
    rows = [triage(item) for item in found.values()]
    counts = dict(Counter(row['queue'] for row in rows))
    target = ROOT / 'data/manifests' / (run_id + '_screened_v2.jsonl')
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    csv_path = target.with_suffix('.csv')
    fields = ['article_id', 'title', 'year', 'queue', 'reason', 'publication_types', 'training_ready']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    labels = {'excluded_review': '命中review/meta-analysis，排除',
              'fulltext_required': '进入全文检查，不以标题摘要线索不足排除'}
    cards = []
    for queue in labels:
        cards.append('<h2>'+labels[queue]+f'（{counts.get(queue, 0)}篇）</h2>')
        for row in rows:
            if row['queue'] != queue:
                continue
            snippets = ''.join('<li>'+escape(name+': '+h['field']+' — '+h['context'])+'</li>'
                               for name, hits in row['evidence'].items() for h in hits)
            cards.append('<section><a href="https://pmc.ncbi.nlm.nih.gov/articles/'+row['article_id']+'/">'
                         +escape(row['article_id']+' | '+row['title'])+'</a><p>'+escape(row['reason'])
                         +'</p><details><summary>查看规则命中的原文证据</summary><ul>'+snippets+'</ul></details></section>')
    (ROOT / 'reports' / (run_id+'_screening_v2.html')).write_text(
        '<!doctype html><meta charset="utf-8"><title>文献初筛</title>'
        '<style>body{font:17px/1.7 sans-serif;max-width:1100px;margin:40px auto}section{padding:15px;border-bottom:1px solid #ddd}a{color:#087f83}</style>'
        f'<h1>{len(rows)}篇候选文献：新规则</h1><p>标题、摘要、文章类型命中review/meta-analysis即排除；其余全文检查疾病与MRI。图片、病例归属和许可仍需后续检查。</p>'
        +''.join(cards), encoding='utf-8')
    receipt = {'rule_version': RULE_VERSION, 'input_articles': len(rows), 'queues': counts,
               'finally_included': 0, 'finally_excluded': counts.get('excluded_review',0), 'training_ready': 0,
               'input_hashes': inputs, 'manifest_sha256': digest(manifest), 'output_sha256': digest(target)}
    dump(ROOT / 'reports' / (run_id+'_screening_v2_receipt.json'), receipt)
    print(json.dumps({k:v for k,v in receipt.items() if not k.endswith('sha256') and k != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
