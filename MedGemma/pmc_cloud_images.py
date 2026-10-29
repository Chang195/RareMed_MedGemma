"""PMC official anonymous cloud media adapter; preserves original BioC captions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import unicodedata
from urllib.parse import urlencode, urlparse, parse_qs, unquote
import xml.etree.ElementTree as ET
from pipeline_runtime_repair import atomic_dump

BASE = 'https://pmc-oa-opendata.s3.amazonaws.com/'


class CloudHold(ValueError):
    def __init__(self, status, reason):
        self.status = status
        super().__init__(reason)


def https_url(url):
    p = urlparse(url)
    if p.scheme == 's3' and p.netloc == 'pmc-oa-opendata':
        return BASE + p.path.lstrip('/') + ('?'+p.query if p.query else '')
    if p.scheme != 'https' or p.netloc != 'pmc-oa-opendata.s3.amazonaws.com':
        raise ValueError('Non-official cloud URL')
    return url


class CloudClient:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.events = []

    def get(self, url):
        url = https_url(url)
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache / (key+'.bin')
        side = self.cache / (key+'.json')
        self.cache.mkdir(parents=True, exist_ok=True)
        expected = parse_qs(urlparse(url).query).get('md5', [None])[0]
        if path.exists() and side.exists():
            meta = json.loads(side.read_text(encoding='utf8'))
            if meta['url'] != url or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError('Cloud cache provenance/hash mismatch')
        else:
            part = path.with_suffix('.part')
            result = subprocess.run(['curl.exe','--fail','--location','--silent','--show-error',
                '--proto','=https','--proto-redir','=https','--connect-timeout','15','--max-time','90',
                '--max-filesize','104857600','--output',str(part),url],capture_output=True,text=True)
            if result.returncode:
                raise CloudHold('held_cloud_fetch_error', result.stderr.strip())
            data = part.read_bytes()
            if expected and hashlib.md5(data).hexdigest().lower() != expected.lower():
                raise ValueError('Official MD5 mismatch; download not accepted')
            if not data:
                raise ValueError('Empty cloud file')
            part.replace(path)
            atomic_dump(side, {'url':url,'sha256':hashlib.sha256(data).hexdigest(),
                'md5':hashlib.md5(data).hexdigest(),'expected_md5':expected,'bytes':len(data),
                'retrieved_at':datetime.now(timezone.utc).isoformat()})
            time.sleep(.4)
        if expected and hashlib.md5(path.read_bytes()).hexdigest().lower() != expected.lower():
            raise ValueError('Cached official MD5 mismatch')
        self.events.append(json.loads(side.read_text(encoding='utf8')))
        return path

    def materialize(self, url, target):
        source = self.get(url)
        target = Path(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Local target has different bytes; refusing overwrite')
        if not target.exists():
            shutil.copy2(source,target)
        meta = dict(self.events[-1])
        atomic_dump(target.with_name(target.name+'.download.json'),meta)
        return target

    def mapping(self, aid, raw):
        bio = ET.parse(raw/'article.bioc.xml')
        versions = {n.text for n in bio.findall(".//infon[@key='article-id_pmcid-ver']") if n.text}
        listing = self.get(BASE+'?'+urlencode({'list-type':2,'prefix':aid+'.','delimiter':'/'}))
        tree = ET.parse(listing)
        if tree.findtext('.//{*}IsTruncated') == 'true':
            raise CloudHold('held_cloud_version_ambiguous','Version listing truncated')
        available = [n.text.rstrip('/') for n in tree.findall('.//{*}CommonPrefixes/{*}Prefix')]
        selected = [v for v in available if v in versions]
        fallback = not versions and len(available) == 1
        if fallback:
            selected = available
        # Never assume that a higher cloud version is equivalent to this BioC article version.
        if len(selected) != 1:
            raise CloudHold('held_cloud_version_unmatched',f'BioC versions={sorted(versions)}, cloud versions={available}')
        version = selected[0]
        meta_path = self.materialize(BASE+version+'/'+version+'.json', raw/'article.cloud.json')
        meta = json.loads(meta_path.read_text(encoding='utf8'))
        if meta.get('pmcid') != aid or version != aid+'.'+str(meta.get('version')):
            raise ValueError('Cloud article identity mismatch')
        if fallback:
            dois = {n.text.lower().strip() for n in bio.findall(".//infon[@key='article-id_doi']") if n.text}
            if not meta.get('doi') or dois != {meta['doi'].lower().strip()}:
                raise CloudHold('held_cloud_version_unmatched','Unversioned BioC requires unique cloud version and identical DOI')
        xml_url = https_url(meta['xml_url'])
        if not unquote(urlparse(xml_url).path).startswith('/'+version+'/'):
            raise ValueError('JATS URL outside selected article version')
        xml_path = self.materialize(xml_url,raw/'article.cloud.jats.xml')
        jats = ET.parse(xml_path)
        mapping = media_mapping(jats.getroot(), meta.get('media_urls',[]), version)
        # Require exact normalized caption identity before attaching cloud media to a BioC figure.
        normalized = lambda s: re.sub(r'\s+', '', unicodedata.normalize('NFKC',s))
        jats_captions = {}
        for fig in jats.findall('.//{*}fig'):
            label = fig.find('{*}label')
            caption = fig.find('{*}caption')
            if caption is not None:
                cap = ''.join(caption.itertext())
                lab = ''.join(label.itertext()) if label is not None else ''
                jats_captions[fig.get('id')] = {normalized(cap),normalized(lab+cap)}
        matching = set()
        for p in bio.findall('.//passage'):
            inf = {n.get('key'):n.text for n in p.findall('infon')}
            if inf.get('section_type') == 'FIG' and inf.get('type') in {'fig_title_caption','fig_caption'}:
                if normalized(p.findtext('text') or '') in jats_captions.get(inf.get('id'),set()):
                    matching.add(inf.get('id'))
        mismatched = [fid for fid in mapping if fid not in matching]
        mapping = {fid:urls for fid,urls in mapping.items() if fid in matching}
        atomic_dump(raw/'cloud_mapping_receipt.json',{'article_id':aid,'selected_version':version,
            'metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            'jats_sha256':hashlib.sha256(xml_path.read_bytes()).hexdigest(),
            'license_code':meta.get('license_code'),'is_retracted':meta.get('is_retracted'),
            'is_manuscript':meta.get('is_manuscript'),'figure_urls':mapping,
            'caption_source':'unchanged original BioC passages and offsets',
            'caption_mismatch_held':mismatched,
            'version_policy':'exact BioC version; if absent, unique cloud version plus exact DOI and figure caption identity. Never highest-version guessing',
            'used_unversioned_fallback':fallback})
        return mapping


def media_mapping(jats, urls, version):
    media = {}
    for url in urls:
        url = https_url(url)
        p = urlparse(url)
        if not unquote(p.path).startswith('/'+version+'/'):
            raise ValueError('Media URL outside selected article version')
        media.setdefault(PurePosixPath(unquote(p.path)).name,[]).append(url)
    result = {}
    for fig in jats.findall('.//{*}fig'):
        fid = fig.get('id','')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',fid):
            continue
        found = []
        for graphic in fig.findall('.//{*}graphic'):
            href = graphic.get('{http://www.w3.org/1999/xlink}href','')
            # Exact graphic filename only, not guessed extensions or panel matching.
            name = PurePosixPath(href).name
            found.extend(media.get(name,[]))
        result[fid] = list(dict.fromkeys(found))
    return result
