"""Bounded public-literature acquisition. No model-generated labels.

Requires Python 3.10+, Pillow and curl. Outputs are research candidates only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlencode, urljoin, urlparse
import xml.etree.ElementTree as ET
from PIL import Image
from license_rules import classify, collection_decision

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'acquisition-v0.7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class Downloader:
    def __init__(self, offline=False):
        self.offline = offline
        self.events = []

    def get(self, url, path):
        host = urlparse(url).hostname or ''
        if urlparse(url).scheme != 'https' or host not in {
            'www.ebi.ac.uk', 'www.ncbi.nlm.nih.gov', 'pmc.ncbi.nlm.nih.gov', 'cdn.ncbi.nlm.nih.gov'
        }:
            raise ValueError('Unapproved source URL: ' + url)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = path.with_name(path.name + '.download.json')
        cached = path.exists() and metadata.exists()
        if cached:
            prior = json.loads(metadata.read_text(encoding='utf-8'))
            if prior['url'] != url or prior['sha256'] != digest(path):
                raise ValueError('Cache provenance/hash mismatch: ' + str(path))
        else:
            if self.offline:
                raise ValueError('No verified cache: ' + str(path))
            temp = path.with_name(path.name + '.part')
            result = subprocess.run([
                'curl.exe', '--fail', '--location', '--silent', '--show-error',
                '--proto', '=https', '--proto-redir', '=https',
                '--retry', '2', '--connect-timeout', '15', '--max-time', '60',
                '--user-agent', 'RareMed-Pairs/0.2 research-pilot',
                '--output', str(temp), url
            ], capture_output=True, text=True)
            if result.returncode:
                temp.unlink(missing_ok=True)
                raise RuntimeError(result.stderr.strip())
            if not temp.stat().st_size:
                raise ValueError('Empty download')
            temp.replace(path)
            dump(metadata, {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                            'sha256': digest(path), 'bytes': path.stat().st_size})
            time.sleep(0.4)
        self.events.append({'path': path.relative_to(ROOT).as_posix(), 'url': url,
                            'cached': cached, 'sha256': digest(path)})
        return path


class Figures(HTMLParser):
    def __init__(self):
        super().__init__()
        self.figures = {}
        self.current = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'figure':
            self.current = a.get('id')
            if self.current:
                self.figures[self.current] = []
        elif tag == 'img' and self.current and a.get('src'):
            self.figures[self.current].append(a['src'])

    def handle_endtag(self, tag):
        if tag == 'figure':
            self.current = None


def parse_bioc(path, pmcid):
    doc = ET.parse(path).find('.//document')
    if doc is None:
        raise ValueError('BioC document identity mismatch')
    doc_id = doc.findtext('id') or ''
    normalized_id = 'PMC'+doc_id if doc_id.isdigit() else doc_id
    passage_ids = [i.text for i in doc.findall(".//passage/infon[@key='article-id_pmc']")]
    normalized_passage_ids = [('PMC'+v if v and v.isdigit() else v) for v in passage_ids]
    if normalized_id != pmcid and not (doc_id == 'unknown' and pmcid in normalized_passage_ids):
        raise ValueError('BioC document identity mismatch')
    passages = [{'metadata': {i.attrib['key']: i.text or '' for i in p.findall('infon')},
                 'offset': int(p.findtext('offset')), 'text': p.findtext('text') or ''}
                for p in doc.findall('passage')]
    return passages


def license_allowed(text):
    # Require an explicit versioned BY license; NC/ND/SA are not silently accepted.
    restricted = r'CC[ -]?BY[ -](?:NC|ND|SA)|/licenses/by-(?:nc|nd|sa)|non[ -]?commercial|no[ -]?derivatives|share[ -]?alike'
    allowed = r'CC[ -]?BY\s*4\.0|creativecommons\.org/licenses/by/4\.0|Creative Commons Attribution(?:\s+International)?(?:\s+License)?\s+4\.0'
    return bool(re.search(allowed, text, re.I)) and not bool(re.search(restricted, text, re.I))


def caption_signals(text):
    patterns = {
        'mri': r'\bMRI\b|magnetic resonance|\b(?:T1|T2)[ -]?(?:W|weighted)?\b|\bFLAIR\b|\bDWI\b|\bADC\b|\bSWI\b',
        'other_modality': r'\bCT\b|computed tomography|histolog|\bPET\b|\bultrasound\b',
        'panel': r'\([A-Za-z]\)|\bpanel[s]?\b|\b[A-D][–-][A-D]\b',
        'annotation': r'\barrows?\b|arrowheads?|asterisks?'
    }
    return {key: [m.group() for m in re.finditer(pattern, text, re.I)]
            for key, pattern in patterns.items()}


def collect_article(pmcid, dl, mri_only=False, license_policy='by4_only', license_hint=''):
    if license_policy=='exclude_by_nc_nd':
        hint_decision=collection_decision('',license_hint)
        if not hint_decision['collect']:
            return [],{'article_id':pmcid,'status':'excluded_license_by_nc_nd',
                       'license_metadata_hint':license_hint,'license_policy':license_policy}
    raw = ROOT / 'data/raw' / pmcid
    xml = dl.get('https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/'
                 + pmcid + '/unicode', raw / 'article.bioc.xml')
    if '[Error] : No result can be found.' in xml.read_text(encoding='utf-8'):
        return [], {'article_id': pmcid, 'status': 'held_bioc_unavailable',
                    'reason': 'BioC returned no result; not a finding that the article has no figures.',
                    'response_sha256': digest(xml)}
    passages = parse_bioc(xml, pmcid)
    front = next(p for p in passages if p['metadata'].get('section_type') == 'TITLE')
    license_text = front['metadata'].get('license', '')
    declared=classify(license_text)
    if license_policy=='by4_only':
        if not license_allowed(license_text):
            return [], {'article_id': pmcid, 'status': 'held_license', 'license_evidence': license_text}
        license_info={'family':'CC BY','version':'4.0','status':'article_declaration_only'}
    elif license_policy=='exclude_by_nc_nd':
        license_info=collection_decision(license_text,license_hint)
        if not license_info['collect']:
            return [],{'article_id':pmcid,'status':'excluded_license_by_nc_nd',
                       'license_evidence':license_text,'license_metadata_hint':license_hint,'license_policy':license_policy}
    else:
        raise ValueError('Unknown license policy')
    html = dl.get(f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/', raw / 'article.html')
    parser = Figures()
    parser.feed(html.read_text(encoding='utf-8'))
    if not parser.figures:
        raise ValueError('No HTML figures: unsupported page or access challenge')
    # Preserve every source passage, including figure/body offsets, for later evidence review.
    dump(ROOT / 'data/processed/articles' / (pmcid + '.json'),
         {'article_id': pmcid, 'passages': passages, 'source_sha256': digest(xml)})
    records, audit = [], []
    grouped_captions = {}
    for p in passages:
        meta = p['metadata']
        if meta.get('section_type') == 'FIG' and meta.get('type') in {'fig_title_caption', 'fig_caption'}:
            grouped_captions.setdefault(meta.get('id'), []).append(p)
    for caption_parts in grouped_captions.values():
        p = next((part for part in caption_parts if part['metadata'].get('type') == 'fig_title_caption'), caption_parts[0])
        m = p['metadata']
        signals = caption_signals(' '.join(part['text'] for part in caption_parts))
        audit.append({'figure_id': m.get('id'), 'caption': p['text'], 'offset': p['offset'],
                      'signals': signals, 'decision': 'candidate' if signals['mri'] or not mri_only
                      else 'held_no_explicit_mri_caption_signal'})
        if mri_only and not signals['mri']:
            continue
        fid = m.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', fid):
            raise ValueError('Unsafe or missing figure ID')
        urls = list(dict.fromkeys(parser.figures.get(fid, [])))
        rec = {'sample_id': pmcid + '_' + fid, 'article_id': pmcid,
               'article_title': front['text'], 'doi': front['metadata'].get('article-id_doi'),
               'source_url': f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/',
            'license': license_info['family']+(' '+license_info['version'] if license_info['version'] else ''),
            'license_evidence': license_text,'license_metadata_hint':license_hint,
            'license_policy':license_policy,'license_review_status':license_info['status'],
               'figure_id': fid, 'caption': p['text'],
               'additional_caption_evidence': [{'text':part['text'], 'offset':part['offset']}
                                               for part in caption_parts if part is not p],
               'caption_evidence': {'path': xml.relative_to(ROOT).as_posix(), 'offset': p['offset'],
                                    'offset_unit': 'BioC source text offset'},
               'caption_signals': signals,
               'assets': [], 'quality_flags': ['figure_rights_review_pending',
                   'image_privacy_review_pending', 'modality_and_panel_review_pending'],
               'case_group_id': None, 'split_group_id': pmcid,
               'group_note': 'Conservative article grouping; patient identity and cross-article overlap unresolved.',
               'diagnosis': None, 'findings': [], 'training_ready': False,
               'review_status': 'candidate_pending_review', 'synthetic': False, 'pipeline_version': VERSION}
        for signal, flag in [('other_modality', 'possible_mixed_modalities'),
                             ('panel', 'possible_multi_panel'), ('annotation', 'caption_mentions_annotations')]:
            if signals[signal]:
                rec['quality_flags'].append(flag)
        if len(urls) != 1:
            rec['quality_flags'].append('unresolved_figure_asset_mapping')
        else:
            url = urljoin(rec['source_url'], urls[0])
            image_path = dl.get(url, raw / (fid + '.image'))
            with Image.open(image_path) as im:
                im.verify()
            with Image.open(image_path) as im:
                dims, fmt = list(im.size), im.format
            rec['assets'].append({'url': url, 'path': image_path.relative_to(ROOT).as_posix(),
                                  'sha256': digest(image_path), 'size': dims, 'format': fmt})
        records.append(rec)
    dump(ROOT / 'data/processed/articles' / (pmcid + '_figure_audit.json'), audit)
    if not records and not mri_only:
        raise ValueError('No supported caption records found')
    return records, {'article_id': pmcid, 'status': 'candidate' if records else 'held_no_mri_caption_signal',
                     'figures': len(records), 'total_caption_records': len(audit)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/acquisition.json')
    ap.add_argument('--offline', action='store_true', help='Only use hash-verified cached downloads')
    ap.add_argument('--search-only', action='store_true', help='Discover candidates without acquiring articles')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    run_id = cfg['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run ID')
    dl = Downloader(args.offline)
    candidates, records, results = {}, [], []
    for query in cfg['queries']:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode({
            'query': query, 'format': 'json', 'pageSize': cfg['page_size'], 'resultType': 'core'})
        # Include pagination/size in new cache identity. Retain compatible old caches.
        old_key = hashlib.sha256(query.encode()).hexdigest()[:16]
        old_meta = ROOT / 'data/raw/search' / (old_key+'.json.download.json')
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        if old_meta.exists() and json.loads(old_meta.read_text(encoding='utf-8')).get('url') == url:
            key = old_key
        path = dl.get(url, ROOT / 'data/raw/search' / (key + '.json'))
        response = json.loads(path.read_text(encoding='utf-8'))
        for item in response.get('resultList', {}).get('result', []):
            pmcid = item.get('pmcid')
            if pmcid and re.fullmatch(r'PMC\d+', pmcid):
                candidate = candidates.setdefault(pmcid, {'article_id': pmcid, 'title': item.get('title'),
                    'year': item.get('pubYear'), 'queries': [], 'status': 'discovered_not_reviewed'})
                candidate['queries'].append(query)
    selected = [] if args.search_only else list(candidates)[:cfg['max_articles']]
    for pmcid in selected:
        try:
            rows, status = collect_article(pmcid, dl)
            records.extend(rows)
            results.append(status)
        except Exception as exc:
            results.append({'article_id': pmcid, 'status': 'failed', 'error': str(exc)})
    dump(ROOT / 'data/manifests' / (run_id + '_sources.json'), list(candidates.values()))
    output = ROOT / 'data/processed' / (run_id + '_figures.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    receipt = {'pipeline_version': VERSION, 'config': cfg, 'offline': args.offline,
               'discovered_articles': len(candidates), 'attempted_articles': len(selected),
               'results': results, 'figure_candidates': len(records),
               'decoded_images': sum(len(r['assets']) for r in records),
               'training_ready': 0, 'downloads': dl.events, 'output_sha256': digest(output)}
    dump(ROOT / 'reports' / (run_id + '_receipt.json'), receipt)
    cards = ''.join('<section><h2>' + escape(r['sample_id']) + '</h2>' +
        ''.join('<img width="480" src="../' + escape(a['path']) + '">' for a in r['assets']) +
        '<p>' + escape(r['caption']) + '</p><p>待审核：' + escape(', '.join(r['quality_flags'])) + '</p></section>'
        for r in records)
    (ROOT / 'reports' / (run_id + '_review.html')).write_text(
        '<!doctype html><meta charset="utf-8"><title>自动采集结果</title>'
        '<style>body{font:18px/1.6 sans-serif;max-width:1000px;margin:40px auto}section{border-top:1px solid #aaa}</style>'
        f'<h1>代码流水线采集结果</h1><p>{len(candidates)}篇检索候选，{len(records)}个图注记录；训练就绪0。</p>' + cards,
        encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('downloads', 'config')}, ensure_ascii=False, indent=2))
    if (not records and not args.search_only) or any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
