"""After review exclusion, inspect full article text for disease/MRI evidence.

No model calls or image downloads. References are excluded from signal matching.
Missing text is held, never treated as evidence of irrelevance.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from collect import ROOT, Downloader, parse_bioc, digest, dump
from screen import PATTERNS

VERSION='fulltext-evidence-v0.1'


def assess(passages):
    hits={name:[] for name in ('cadasil','sporadic_csvd','mri')}
    for idx,p in enumerate(passages):
        section=p['metadata'].get('section_type','')
        if section in {'REF','COMP_INT','AUTH_CONT','ACK_FUND','ACKNOWLEDGMENTS'}:
            continue
        for name in hits:
            match=re.search(PATTERNS[name],p['text'],re.I)
            if match:
                hits[name].append({'passage_index':idx,'section':section,'source_offset':p['offset'],
                    'match_start_in_passage':match.start(),'match_end_in_passage':match.end(),
                    'matched_text':match.group(),'context':p['text'][max(0,match.start()-100):match.end()+160]})
    topic=bool(hits['cadasil'] or hits['sporadic_csvd'])
    # Body/figure presence verifies that this wasn't just an abstract response.
    body=any(p['metadata'].get('section_type') not in {'TITLE','ABSTRACT','REF','COMP_INT','AUTH_CONT'}
             and p['text'].strip() for p in passages)
    if not body:
        return 'held_fulltext_incomplete',hits
    return ('fulltext_passed' if topic and hits['mri'] else 'excluded_fulltext_signals_insufficient'),hits


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/fulltext_screening.json')
    ap.add_argument('--cached-only',action='store_true',help='Process verified cached files, leave remaining articles pending')
    ap.add_argument('--limit',type=int,default=0,help='Maximum non-review articles this invocation; 0 means all')
    args=ap.parse_args()
    if args.limit<0:
        ap.error('limit cannot be negative')
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    run_id=cfg['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise ValueError('Invalid run ID')
    source=(ROOT/cfg['metadata_manifest']).resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError('Manifest must be inside project')
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').split('\n') if line.strip()]
    checkpoint=ROOT/'data/manifests'/(run_id+'_checkpoint.json')
    state=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {
        'manifest_sha256':digest(source),'rule_version':VERSION,'completed':{}}
    if state['manifest_sha256']!=digest(source) or state['rule_version']!=VERSION:
        raise ValueError('Checkpoint input/rule mismatch; choose a new run ID')
    dl=Downloader(offline=args.cached_only)
    output=[]
    attempts=0
    for row in rows:
        aid=row['article_id']
        if not re.fullmatch(r'PMC\d+',aid):
            raise ValueError('Invalid article ID')
        record=dict(row)
        record['metadata_queue']=row['queue']
        record['fulltext_rule_version']=VERSION
        if row['queue'] in {'excluded_review','excluded_license_by_nc_nd'}:
            record['queue']=row['queue']
            record['fulltext_status']='not_fetched_'+row['queue']
            output.append(record)
            continue
        prior=state['completed'].get(aid)
        raw=ROOT/'data/raw'/aid/'article.bioc.xml'
        if prior and raw.exists() and prior.get('source_sha256')==digest(raw):
            record.update(prior)
            output.append(record)
            continue
        if (args.limit and attempts>=args.limit) or (args.cached_only and not raw.with_name(raw.name+'.download.json').exists()):
            record.update(queue='fulltext_pending',fulltext_status='not_processed')
            output.append(record)
            continue
        attempts+=1
        print('Full-text screening '+aid,flush=True)
        url='https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/'+aid+'/unicode'
        try:
            dl.get(url,raw)
            if '[Error] : No result can be found.' in raw.read_text(encoding='utf-8'):
                result={'queue':'held_fulltext_unavailable','fulltext_status':'bioc_no_result'}
            else:
                passages=parse_bioc(raw,aid)
                queue,hits=assess(passages)
                result={'queue':queue,'fulltext_status':'parsed','fulltext_evidence':hits,
                        'fulltext_license_evidence':next((p['metadata'].get('license') for p in passages
                            if p['metadata'].get('license')),None)}
                # Separate file avoids overwriting prior model inputs or parsed evidence.
                dump(ROOT/'data/processed/fulltext_screening'/(aid+'.json'),
                     {'article_id':aid,'passages':passages,'source_sha256':digest(raw),'rule_version':VERSION})
            result.update(source_sha256=digest(raw),fulltext_source_path=raw.relative_to(ROOT).as_posix())
            state['completed'][aid]=result
            dump(checkpoint,state)
        except Exception as exc:
            result={'queue':'held_fulltext_error','fulltext_status':'fetch_or_parse_failed','error':str(exc)}
            # Failures stay retryable; no permanent exclusion based on access problems.
        record.update(result)
        output.append(record)
    target=ROOT/'data/manifests'/(run_id+'.jsonl')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    report={'rule_version':VERSION,'input_articles':len(rows),'cached_only':args.cached_only,
            'new_attempts_this_invocation':attempts,'queues':dict(Counter(r['queue'] for r in output)),
            'source_manifest_sha256':digest(source),'output_sha256':digest(target),
            'meaning':'Disease and MRI text signals only; not a patient diagnosis or image confirmation.',
            'training_ready':0}
    dump(ROOT/'reports'/(run_id+'_receipt.json'),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
