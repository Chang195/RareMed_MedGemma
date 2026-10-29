"""Two narrow changes: cursor pagination and verified BioC PMCID aliases."""
import hashlib
import json
from pathlib import Path


def patched_bytes(relative, raw):
    if relative not in {'scripts/collect.py', 'scripts/audit_licenses.py'}:
        return raw
    text = raw.decode('utf8').replace('\r\n', '\n')
    if relative == 'scripts/collect.py':
        anchor = '    passages = [{\'metadata\':'
        start = text.index("    passage_ids = [i.text")
        end = text.index(anchor, start)
        text = text[:start] + '''    keys = ('article-id_pmc', 'article-id_pmcid', 'article-id_pmcaid')
    passage_ids = [i.text for key in keys
                   for i in doc.findall(".//passage/infon[@key='" + key + "']")]
    normalized_passage_ids = {('PMC'+v if v.isdigit() else v)
                              for raw in passage_ids if (v := (raw or '').strip())}
    if normalized_id != pmcid and not (
            doc_id == 'unknown' and normalized_passage_ids == {pmcid}):
        raise ValueError('BioC document identity mismatch')
''' + text[end:]
        start = text.index('    for query in cfg[\'queries\']:', text.index('def main():'))
        end = text.index('    selected = []', start)
        text = text[:start] + '''    for query in cfg['queries']:
        for path, response in iter_search_pages(dl, query, cfg['page_size']):
            for item in response.get('resultList', {}).get('result', []):
                pmcid = item.get('pmcid')
                if pmcid and re.fullmatch(r'PMC\\d+', pmcid):
                    candidate = candidates.setdefault(pmcid, {'article_id': pmcid, 'title': item.get('title'),
                        'year': item.get('pubYear'), 'queries': [], 'status': 'discovered_not_reviewed'})
                    if query not in candidate['queries']:
                        candidate['queries'].append(query)
''' + text[end:]
        helper = '''
def iter_search_pages(dl, query, page_size):
    # First request is byte-for-byte the original URL. Subsequent requests only add cursorMark.
    cursor = None
    seen = set()
    while True:
        params = {'query': query, 'format': 'json', 'pageSize': page_size, 'resultType': 'core'}
        if cursor is not None:
            params['cursorMark'] = cursor
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode(params)
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        if cursor is None:
            old_key = hashlib.sha256(query.encode()).hexdigest()[:16]
            old_meta = ROOT / 'data/raw/search' / (old_key+'.json.download.json')
            if old_meta.exists() and json.loads(old_meta.read_text(encoding='utf-8')).get('url') == url:
                key = old_key
        path = dl.get(url, ROOT / 'data/raw/search' / (key + '.json'))
        response = json.loads(path.read_text(encoding='utf-8'))
        if 'resultList' not in response:
            raise ValueError('Search response missing resultList; do not freeze partial discovery')
        rows = response['resultList'].get('result', [])
        yield path, response
        nxt = response.get('nextCursorMark')
        if not rows or not nxt or nxt == cursor or nxt in seen:
            break
        seen.add(nxt)
        cursor = nxt


'''
        text = text.replace('def main():', helper + 'def main():', 1)
    elif relative == 'scripts/audit_licenses.py':
        # The original metadata reader must read metadata for page 2 as well as page 1.
        start = text.index('def read_metadata(cfg):')
        end = text.index('\n\ndef main():', start)
        text = text[:start] + '''def read_metadata(cfg):
    from collect import iter_search_pages, Downloader
    records, inputs = {}, {}
    for query in cfg['queries']:
        for path, data in iter_search_pages(Downloader(offline=True), query, cfg['page_size']):
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            for item in data['resultList']['result']:
                if item.get('pmcid'):
                    records[item['pmcid']] = item
    return records, inputs
''' + text[end:]
    result = text.encode('utf8')
    if relative == 'scripts/collect.py':
        from pipeline_runtime_repair import apply_runtime_patch
        result = apply_runtime_patch(result)
    return result


def prepare_patched(source, run, files, exclusions):
    import difflib
    records = {}
    for src in files:
        name = src.relative_to(source).as_posix()
        raw = src.read_bytes()
        data = patched_bytes(name, raw)
        dest = run / name
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError('Run file differs from approved patch: ' + name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        records[name] = {'original_sha256': hashlib.sha256(raw).hexdigest(),
                         'run_sha256': hashlib.sha256(data).hexdigest(), 'changed': raw != data}
        if raw != data:
            diff = ''.join(difflib.unified_diff(raw.decode().splitlines(True), data.decode().splitlines(True),
                           fromfile='original/'+name, tofile='approved-patch/'+name))
            p = run / 'local_run' / (Path(name).name+'.diff')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(diff, encoding='utf8')
    local = run / 'local_run'
    local.mkdir(exist_ok=True)
    helper = Path(__file__).with_name('pipeline_runtime_repair.py')
    target = run / 'scripts/pipeline_runtime_repair.py'
    if target.exists() and target.read_bytes() != helper.read_bytes():
        raise ValueError('Runtime helper changed; preserve run before upgrade')
    target.write_bytes(helper.read_bytes())
    records['scripts/pipeline_runtime_repair.py'] = {
        'original_sha256': None, 'run_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
        'changed': True, 'purpose': 'Approved checkpoint I/O recovery and access-challenge hold; no medical rule changes'}
    f = local / 'historical_exclusions.json'
    if f.exists() and json.loads(f.read_text(encoding='utf8')) != exclusions:
        raise ValueError('Exclusions changed. Use a new run version.')
    f.write_text(json.dumps(exclusions, ensure_ascii=False, indent=2), encoding='utf8')
    (local / 'file_verification.json').write_text(json.dumps(records, indent=2), encoding='utf8')
    return records
