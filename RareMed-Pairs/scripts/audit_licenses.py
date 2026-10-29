"""Count article-license declarations across a discovery manifest.

Uses verified local BioC when available; falls back to Europe PMC metadata hints.
No network, image reuse decisions, or privacy certification.
"""
import argparse
from collections import Counter
import csv
from html import escape, unescape
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from collect import ROOT, dump, digest, parse_bioc, license_allowed

VERSION='license-audit-v0.2'


from license_rules import classify


def read_metadata(cfg):
    records={}
    inputs={}
    for query in cfg['queries']:
        url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode({
            'query':query,'format':'json','pageSize':cfg['page_size'],'resultType':'core'})
        old_key=hashlib.sha256(query.encode()).hexdigest()[:16]
        key=hashlib.sha256(url.encode()).hexdigest()[:16]
        old=ROOT/'data/raw/search'/(old_key+'.json')
        old_meta=old.with_name(old.name+'.download.json')
        if old_meta.exists() and json.loads(old_meta.read_text(encoding='utf-8')).get('url')==url:
            path=old
        else:
            path=ROOT/'data/raw/search'/(key+'.json')
        meta=json.loads(path.with_name(path.name+'.download.json').read_text(encoding='utf-8'))
        if meta['url']!=url or meta['sha256']!=digest(path):
            raise ValueError('Search metadata provenance mismatch')
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
        data=json.loads(path.read_text(encoding='utf-8'))
        for item in data['resultList']['result']:
            if item.get('pmcid'):
                records[item['pmcid']]=item
    return records,inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/discovery_1000.json')
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    run_id=cfg['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise ValueError('Invalid run ID')
    manifest=ROOT/'data/manifests'/(run_id+'_sources.json')
    candidates=json.loads(manifest.read_text(encoding='utf-8'))
    meta,inputs=read_metadata(cfg)
    screening_path=ROOT/'data/manifests'/(run_id+'_screened_v2.jsonl')
    screened={r['article_id']:r for r in (json.loads(s) for s in screening_path.read_text(encoding='utf-8').split('\n') if s.strip())} if screening_path.exists() else {}
    rows=[]
    for item in candidates:
        aid=item['article_id']
        hint=meta[aid].get('license') or ''
        metadata_license=classify(hint)
        raw=ROOT/'data/raw'/aid/'article.bioc.xml'
        sidecar=raw.with_name(raw.name+'.download.json')
        evidence=''
        raw_hash=None
        raw_status='not_downloaded'
        if raw.exists() and sidecar.exists():
            info=json.loads(sidecar.read_text(encoding='utf-8'))
            raw_hash=digest(raw)
            if info['sha256']!=raw_hash:
                raise ValueError('Raw cache hash mismatch: '+aid)
            if not info['url'].endswith('/'+aid+'/unicode'):
                raise ValueError('Raw URL article mismatch: '+aid)
            try:
                passages=parse_bioc(raw,aid)
                evidence=next((p['metadata']['license'] for p in passages if p['metadata'].get('license')),'')
                raw_status='article_identity_verified' if evidence else 'no_license_in_parsed_text'
            except Exception as exc:
                raw_status='unusable_fulltext:'+type(exc).__name__
        full=classify(evidence)
        if evidence and full['family'] not in ('unknown','unrecognized_declaration','conflicting_declarations'):
            chosen=full
            basis='fulltext_declaration'
        elif evidence:
            chosen=full
            basis='unresolved_fulltext_declaration'
        elif hint:
            chosen=metadata_license
            basis='metadata_hint_only'
        else:
            chosen={'family':'unknown','version':None}
            basis='no_license_evidence'
        mismatch=bool(evidence and full['family'] not in ('unknown','unrecognized_declaration','CC_unspecified')
                      and metadata_license['family'] not in ('unknown','unrecognized_declaration')
                      and full['family']!=metadata_license['family'])
        rows.append({'article_id':aid,'title':item['title'],'family':chosen['family'],
            'license_version':chosen['version'],'evidence_basis':basis,
            'fulltext_license_text':evidence,'metadata_license_hint':hint,
            'separate_data_cc0_waiver_mentioned':bool('publicdomain/zero' in evidence and 'licenses/by' in evidence),
            'metadata_family':metadata_license['family'],'fulltext_metadata_family_disagreement':mismatch,
            'raw_status':raw_status,'raw_sha256':raw_hash,
            'review_policy_queue':screened.get(aid,{}).get('queue'),
            'passes_current_narrow_license_parser':license_allowed(evidence) if evidence else None,
            'figure_rights':'not_determined_by_article_license_audit','privacy_status':'not_assessed'})
    report_name=run_id+'_licenses'
    target=ROOT/'data/manifests'/(report_name+'.csv')
    with target.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    counts=Counter(r['family'] for r in rows)
    table=[]
    for family,count in counts.most_common():
        subset=[r for r in rows if r['family']==family]
        table.append({'family':family,'articles':count,
            'fulltext_declaration':sum(r['evidence_basis']=='fulltext_declaration' for r in subset),
            'metadata_hint_only':sum(r['evidence_basis']=='metadata_hint_only' for r in subset),
            'versions':dict(Counter(r['license_version'] or 'unspecified' for r in subset))})
    report={'audit_version':VERSION,'articles':len(rows),'by_license_family':table,
        'evidence_bases':dict(Counter(r['evidence_basis'] for r in rows)),
        'metadata_fulltext_disagreements':sum(r['fulltext_metadata_family_disagreement'] for r in rows),
        'review_excluded_articles':sum(r['review_policy_queue']=='excluded_review' for r in rows),
        'includes_review_excluded_articles':True,'source_manifest_sha256':digest(manifest),
        'search_input_hashes':inputs,'csv_sha256':digest(target),
        'limitations':['Article declarations only, not figure-level rights or privacy review.',
                       'Metadata hints do not establish exact license version or permissions.',
                       'Counts cover 995 actual candidates, not 1000 independent papers.']}
    acquisition=ROOT/'reports/fulltext_policy_v2_figures_receipt.json'
    if acquisition.exists():
        previous=json.loads(acquisition.read_text(encoding='utf-8'))
        held={r['article_id'] for r in previous['results'] if r['status']=='held_license'}
        subset=[r for r in rows if r['article_id'] in held]
        report['previous_license_hold_breakdown']={'articles':len(subset),
            'families':dict(Counter(r['family'] for r in subset)),
            'CC_BY_versions':dict(Counter(r['license_version'] or 'unspecified' for r in subset if r['family']=='CC BY')),
            'acquisition_receipt_sha256':digest(acquisition)}
    dump(ROOT/'reports'/(report_name+'.json'),report)
    dump(ROOT/'data/manifests'/(report_name+'.json'),rows)
    labels={'fulltext_declaration':'全文许可声明','metadata_hint_only':'仅检索元数据提示',
            'unresolved_fulltext_declaration':'全文声明识别未决','no_license_evidence':'无可用许可信息'}
    header='<tr><th>许可类型</th><th>篇数</th><th>全文声明</th><th>仅元数据提示</th><th>版本分布</th></tr>'
    body=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in
              (r['family'],r['articles'],r['fulltext_declaration'],r['metadata_hint_only'],r['versions']))+'</tr>' for r in table)
    details=''.join('<details><summary>'+escape(r['article_id']+' | '+r['family']+' | '+labels[r['evidence_basis']])+'</summary><p>'
        +escape(r['title'])+'</p><p>全文声明：'+escape(r['fulltext_license_text'] or '未取得')+'</p><p>元数据提示：'
        +escape(r['metadata_license_hint'] or '无')+'</p></details>' for r in rows)
    (ROOT/'reports'/(report_name+'.html')).write_text(
        '<!doctype html><meta charset="utf-8"><title>候选文献许可统计</title><style>body{font:17px/1.7 sans-serif;max-width:1200px;margin:40px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}details{border-bottom:1px solid #ddd;padding:8px}</style>'
        f'<h1>{len(rows)}篇候选文献许可统计</h1><p>包括已按review规则排除的文献。区分全文声明与元数据提示；不代表图级版权、AI用途许可或脱敏已审核。</p>'
        '<table>'+header+body+'</table><h2>逐篇声明</h2>'+details,encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
