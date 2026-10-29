"""Download Orphadata and build a provenance-preserving disease knowledge subgraph.

Standard library only. Does not generate patient facts, call an LLM or use images.
"""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/orphadata_v1.json'
BASE = ROOT/'data/knowledge/orphadata'

def now(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def get(e,path):
    return (e.findtext(path) or '').strip() if e is not None else ''
def xml_object(e):
    if e is None:return None
    return dict(tag=e.tag,attributes=dict(e.attrib),text=(e.text or '').strip(),children=[xml_object(c) for c in e])

def inspect_xml(path):
    """Reject HTML/errors, truncation, and unexpected exports before caching."""
    root=None; count=0; license_data=None
    for event,e in ET.iterparse(path,events=('start','end')):
        if root is None:
            if e.tag!='JDBOR':raise ValueError('Expected Orphadata JDBOR XML, not HTML or an API error')
            root=dict(e.attrib)
        if event=='end' and e.tag=='Availability':license_data=xml_object(e)
        if event=='end' and e.tag=='Disorder':
            if not get(e,'OrphaCode') or not get(e,'Name'):raise ValueError('Disorder missing identity')
            count+=1;e.clear()
    if not count:raise ValueError('No disease records in source XML')
    return dict(xml_metadata=root,disorder_count=count,license_declaration=license_data)

def fetch_source(spec,cache,offline=False):
    cache.mkdir(parents=True,exist_ok=True)
    target=cache/spec['file'];receipt=target.with_suffix('.source.json')
    if target.exists() and receipt.exists():
        meta=load(receipt)
        if meta['sha256']!=digest(target):raise ValueError('Cached source hash changed: '+str(target))
        if meta['source_url']!=spec['url']:raise ValueError('Cached source URL changed')
        print('Cache verified: '+spec['file'],flush=True)
        return target,meta
    if offline:raise ValueError('Missing verified cache; first run without --offline: '+str(target))
    if target.exists():raise ValueError('Unreceipted XML exists; preserve it and choose a new --cache-dir')
    part=target.with_suffix('.part')
    print('Downloading '+spec['file']+' ...',flush=True)
    req=Request(spec['url'],headers={'User-Agent':'RareMed-Pairs/OrphadataKG research'})
    with urlopen(req,timeout=120) as r,part.open('wb') as out:
        final_url=r.url;headers={k:r.headers.get(k) for k in ['Content-Type','Last-Modified','ETag']}
        shutil.copyfileobj(r,out)
    inspected=inspect_xml(part)
    meta=dict(source_url=spec['url'],source_page=spec['page'],response_url=final_url,
        retrieved_at_utc=now(),sha256=digest(part),bytes=part.stat().st_size,
        response_headers=headers,acquisition='https_download',**inspected)
    part.replace(target);save(receipt,meta)
    return target,meta

def disease_records(path,wrapper=False):
    """Keep phenotype set metadata outside Disorder, including validation date."""
    tag='HPODisorderSetStatus' if wrapper else 'Disorder'
    for _,e in ET.iterparse(path,events=('end',)):
        if e.tag!=tag:continue
        d=e.find('Disorder') if wrapper else e
        if d is not None:
            yield d,({c.tag:xml_object(c) for c in e if c.tag!='Disorder'} if wrapper else {})
        e.clear()

def resolve(path,query,code=None):
    matches=[]
    for d,_ in disease_records(path,True):
        name=get(d,'Name');oid=get(d,'OrphaCode')
        if (code and oid==str(code).removeprefix('ORPHA:')) or (not code and query.casefold() in name.casefold()):
            matches.append(dict(orpha_code=oid,name=name))
    if not matches:raise ValueError('No phenotype-source disease matched; use search or an exact ORPHA code')
    exact=[r for r in matches if r['name'].casefold()==query.casefold()]
    if not code and len(exact)==1:return exact[0]
    if len(matches)!=1:raise ValueError('Ambiguous disease. Select --orpha-code from: '+json.dumps(matches,ensure_ascii=True))
    return matches[0]

def parse_phenotypes(d):
    result=[];seen=set()
    group=d.find('HPODisorderAssociationList')
    for a in d.findall('./HPODisorderAssociationList/HPODisorderAssociation'):
        hid=get(a,'HPO/HPOId');term=get(a,'HPO/HPOTerm')
        if not re.fullmatch(r'HP:\d{7}',hid) or not term:raise ValueError('Invalid HPO identity')
        aid=a.get('id')
        if aid in seen:raise ValueError('Duplicate phenotype association ID')
        seen.add(aid);freq=get(a,'HPOFrequency/Name')
        result.append(dict(association_id=aid,hpo_id=hid,label=term,frequency_label=freq or None,
            frequency_id=a.find('HPOFrequency').get('id') if a.find('HPOFrequency') is not None else None,
            relation='excludes_phenotype' if re.search(r'^excluded\b',freq,re.I) else 'has_phenotype',
            diagnostic_criteria=xml_object(a.find('DiagnosticCriteria')),source_record=xml_object(a)))
    if group is not None and group.get('count') and int(group.get('count'))!=len(result):raise ValueError('Phenotype count mismatch')
    return result

def parse_genes(d):
    result=[]
    for a in d.findall('./DisorderGeneAssociationList/DisorderGeneAssociation'):
        g=a.find('Gene')
        if g is None or not get(g,'Symbol'):raise ValueError('Missing gene identity')
        result.append(dict(orphanet_gene_id=g.get('id'),symbol=get(g,'Symbol'),name=get(g,'Name'),
            association_type=get(a,'DisorderGeneAssociationType/Name'),status=get(a,'DisorderGeneAssociationStatus/Name'),
            validation_source=get(a,'SourceOfValidation'),
            external_references=[dict(source=get(v,'Source'),identifier=get(v,'Reference')) for v in g.findall('./ExternalReferenceList/ExternalReference')],
            source_record=xml_object(a)))
    return result

def extract(paths,metas,selected):
    code=selected['orpha_code'];result=dict(disease=selected,phenotypes=[],genes=[],natural_history={},source_record_metadata={},sources=metas)
    for kind,path in paths.items():
        found=[]
        for d,outer in disease_records(path,kind=='phenotypes'):
            if get(d,'OrphaCode')!=code:continue
            if get(d,'Name')!=selected['name']:raise ValueError('Cross-source disease name disagreement')
            found.append(True);result['source_record_metadata'][kind]=outer
            result['disease']['expert_link']=get(d,'ExpertLink')
            if kind=='phenotypes':result['phenotypes']=parse_phenotypes(d)
            elif kind=='genes':result['genes']=parse_genes(d)
            else:
                result['natural_history']={
                    'age_of_onset':[dict(id=a.get('id'),label=get(a,'Name')) for a in d.findall('./AverageAgeOfOnsetList/AverageAgeOfOnset')],
                    'inheritance':[dict(id=a.get('id'),label=get(a,'Name')) for a in d.findall('./TypeOfInheritanceList/TypeOfInheritance')],
                    'source_record':xml_object(d)}
        if len(found)>1:raise ValueError('Multiple records for the same ORPHA code in '+kind)
        result.setdefault('source_coverage',{})[kind]='found' if found else 'not_available'
    result.update(schema_version='orphadata-subgraph-v1',knowledge_scope='disease_level_not_patient_facts',
        generated_at_utc=now(),language='en',synthetic_patient_generation='not_run',
        frequency_policy='Verbatim source categories; no exact probabilities inferred. Missing does not mean absent.',
        clinical_rules_status='No joint symptom probabilities, MRI sequence rules or patient-specific evidence inferred.')
    return result

def graph(data):
    oid='ORPHA:'+data['disease']['orpha_code'];nodes={oid:dict(id=oid,type='disease',label=data['disease']['name'])};edges=[]
    def edge(rel,tid,label,kind,record,evidence):
        nodes.setdefault(tid,dict(id=tid,type=kind,label=label))
        src=data['sources'][record]
        edges.append(dict(subject=oid,relation=rel,object=tid,source_url=src['source_url'],source_sha256=src['sha256'],
            source_xml_date=src['xml_metadata'].get('date'),orpha_code=data['disease']['orpha_code'],evidence=evidence))
    for a in data['phenotypes']:edge(a['relation'],a['hpo_id'],a['label'],'phenotype','phenotypes',a)
    for a in data['genes']:edge('gene_association','ORPHAGENETIC:'+a['orphanet_gene_id'],a['symbol'],'gene','genes',a)
    for key,rel in [('age_of_onset','has_onset_category'),('inheritance','has_inheritance')]:
        for a in data['natural_history'].get(key,[]):edge(rel,'ORPHA_CATEGORY:'+a['id'],a['label'],key,'natural_history',a)
    return dict(nodes=list(nodes.values()),edges=edges,scope='disease-level associations; no patient assertions')

def export(data,out):
    out.mkdir(parents=True,exist_ok=True);kg=graph(data)
    # Refuse to silently replace a frozen version with different source data.
    if (out/'disease.json').exists():
        old=load(out/'disease.json')
        if {k:v['sha256'] for k,v in old['sources'].items()}!={k:v['sha256'] for k,v in data['sources'].items()}:
            raise ValueError('Output version has different sources; choose a new --output directory')
    save(out/'disease.json',data);save(out/'graph.json',kg)
    for fn,rr in [('nodes.jsonl',kg['nodes']),('edges.jsonl',kg['edges'])]:
        (out/fn).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr),encoding='utf-8')
    with (out/'phenotypes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['hpo_id','label','frequency_label','relation','association_id'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data['phenotypes'])
    frequencies={}
    for a in data['phenotypes']:frequencies[a['frequency_label'] or 'not_reported']=frequencies.get(a['frequency_label'] or 'not_reported',0)+1
    summary=dict(disease=data['disease'],phenotype_associations=len(data['phenotypes']),gene_associations=len(data['genes']),
        frequency_counts=frequencies,graph_nodes=len(kg['nodes']),graph_edges=len(kg['edges']),
        source_coverage=data['source_coverage'],natural_history={k:v for k,v in data['natural_history'].items() if k!='source_record'},
        knowledge_scope=data['knowledge_scope'],synthetic_patient_generation='not_run',new_llm_api_calls=0)
    save(out/'summary.json',summary)
    def table(headers,rs):
        return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v or 'Not reported'))+'</td>' for v in r)+'</tr>' for r in rs)+'</table>'
    title=data['disease']['name']+' / '+ 'ORPHA:'+data['disease']['orpha_code']
    html='<!doctype html><meta charset="utf-8"><title>'+escape(title)+'</title><style>body{font:17px/1.7 Arial,"Microsoft YaHei";max-width:1100px;margin:35px auto;padding:0 22px;color:#222}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left}th{background:#eef5f4}pre{white-space:pre-wrap;background:#f5f5f5;padding:16px}</style>'
    html+='<h1>'+escape(title)+'：Orphadata 知识子图</h1><p>疾病层面知识，不代表任何真实患者或图像已经具备这些表现。未生成合成病历，未调用大模型。</p>'
    html+='<p>'+str(len(data['phenotypes']))+' 条表型关联，'+str(len(data['genes']))+' 条基因关联。保留来源原始频率，不转换为精确概率。未列出不等于阴性。</p>'
    html+='<h2>表型与频率（原文）</h2>'+table(['HPO ID','表型','来源频率','关系'],[(a['hpo_id'],a['label'],a['frequency_label'],a['relation']) for a in data['phenotypes']])
    html+='<h2>基因关联</h2>'+table(['基因','来源关系类型','状态','证据'],[(a['symbol'],a['association_type'],a['status'],a['validation_source']) for a in data['genes']])
    html+='<h2>发病年龄与遗传方式</h2><pre>'+escape(json.dumps(summary['natural_history'],ensure_ascii=False,indent=2))+'</pre>'
    html+='<h2>版本与来源</h2>'+table(['文件','XML日期','SHA-256'],[(Path(s['source_url']).name,s['xml_metadata'].get('date'),s['sha256']) for s in data['sources'].values()])
    html+='<p>XML 文件日期不等于该疾病全部关系的核验日期。记录级状态和日期保存在 disease.json 的 source_record_metadata。</p>'
    html+='<h2>原始记录元数据</h2><pre>'+escape(json.dumps(data['source_record_metadata'],ensure_ascii=False,indent=2))+'</pre>'
    html+='<p>Attribution: Orphanet / INSERM. Source XML licence declarations retained in disease.json. <a href="https://sciences.orphadata.com/orphanet-scientific-knowledge-files/">Orphadata</a>.</p>'
    (out/'report.html').write_text(html,encoding='utf-8')
    hashes={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    save(out/'manifest.json',dict(schema_version='orphadata-subgraph-v1',source_sha256={k:v['sha256'] for k,v in data['sources'].items()},output_sha256=hashes,script_sha256=digest(Path(__file__))))
    print(json.dumps({**summary,'output':str(out),'report':str(out/'report.html')},ensure_ascii=True,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['fetch','search','build'],nargs='?',default='build')
    ap.add_argument('--query',default='CADASIL');ap.add_argument('--orpha-code');ap.add_argument('--offline',action='store_true')
    ap.add_argument('--cache-dir',type=Path,default=BASE/'raw');ap.add_argument('--output',type=Path)
    args=ap.parse_args();config=load(CONFIG);paths={};metas={}
    for kind,spec in config['sources'].items():paths[kind],metas[kind]=fetch_source(spec,args.cache_dir,args.offline)
    if args.command=='fetch':return
    if args.command=='search':
        matches=[dict(orpha_code=get(d,'OrphaCode'),name=get(d,'Name')) for d,_ in disease_records(paths['phenotypes'],True) if args.query.casefold() in get(d,'Name').casefold()]
        print(json.dumps(matches,ensure_ascii=True,indent=2));return
    selected=resolve(paths['phenotypes'],args.query,args.orpha_code)
    out=args.output or BASE/('ORPHA'+selected['orpha_code']+'_v1')
    export(extract(paths,metas,selected),out)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,ET.ParseError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);sys.exit(1)
