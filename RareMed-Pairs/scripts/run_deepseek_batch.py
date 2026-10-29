"""Sequential bounded extraction with checkpoint reuse and a cumulative summary."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from collect import ROOT, dump, digest
from link_cases import validate
from run_deepseek import MODEL, build_request, call_one


def article_ids(manifest, limit):
    seen = set()
    ordered = []
    for line in manifest.read_text(encoding='utf-8').split('\n'):
        if not line.strip():
            continue
        article = json.loads(line)['article_id']
        if not re.fullmatch(r'PMC\d+',article):
            raise ValueError('Invalid article ID')
        if article not in seen:
            seen.add(article)
            ordered.append(article)
    return ordered[:limit]


def result_location(article, max_tokens):
    payload = build_request(ROOT/'data/model_inputs'/article,max_tokens)
    key = hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode('utf-8')).hexdigest()
    return ROOT/'data/model_outputs'/MODEL/article/key[:16]


def usage_total(receipts):
    totals = {'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    for receipt in receipts:
        for name in totals:
            value = receipt.get('usage',{}).get(name,0)
            if isinstance(value,int) and value>=0:
                totals[name]+=value
    return totals


def summarize(ids, max_tokens, manifest, run_id='deepseek_batch'):
    statuses, receipts, links = [], [], []
    for article in ids:
        try:
            folder = result_location(article,max_tokens)
        except (ValueError,FileNotFoundError) as exc:
            statuses.append({'article_id':article,'status':'preflight_failed','error':str(exc)})
            continue
        path = folder/'receipt.json'
        if not path.exists():
            statuses.append({'article_id':article,'status':'not_attempted'})
            continue
        receipt = json.loads(path.read_text(encoding='utf-8'))
        receipts.append(receipt)
        status = receipt.get('status','unknown')
        errors = receipt.get('errors',[])
        repaired_path=folder/'normalized_response.json'
        repair_log=folder/'format_repair_receipt.json'
        use_repaired=False
        if status=='validation_failed' and repair_log.exists() and repaired_path.exists():
            repair=json.loads(repair_log.read_text(encoding='utf-8'))
            if repair.get('status')=='passed_after_format_repair' and repair.get('candidate_response_sha256')==digest(repaired_path) and repair.get('input_sha256')==digest(ROOT/'data/model_inputs'/article/'input.json'):
                status='validated'
                use_repaired=True
        if status == 'validated':
            inp = ROOT/'data/model_inputs'/article/'input.json'
            response_path = repaired_path if use_repaired else folder/'response.json'
            try:
                request=json.loads(inp.read_text(encoding='utf-8'))
                response=json.loads(response_path.read_text(encoding='utf-8'))
                errors,_=validate(response,request)
                if receipt.get('input_sha256') != digest(inp):
                    errors.append('Prepared input changed since API call')
                if errors:
                    status='local_validation_failed'
                else:
                    for row in response['figures']:
                        links.append({'article_id':article,'figure_id':row['figure_id'],
                            'proposed_linkage':row,'model_requested':MODEL,
                            'model_returned':receipt.get('model_returned'),
                            'response_path':response_path.relative_to(ROOT).as_posix(),
                            'response_sha256':digest(response_path),
                            'input_sha256':digest(inp),'human_review':'skipped_by_user',
                            'semantic_validation':'not_performed','training_ready':False})
            except (OSError,ValueError,TypeError,KeyError) as exc:
                status='local_validation_failed'
                errors=[type(exc).__name__+': output could not be revalidated']
        statuses.append({'article_id':article,'status':status,'errors':errors,
                         'http_status':receipt.get('http_status'),'error':receipt.get('error')})
    output=ROOT/'data/processed'/(run_id+'_links.jsonl')
    output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in links),encoding='utf-8')
    report={'model':MODEL,'updated_at':datetime.now(timezone.utc).isoformat(),
            'source_manifest_sha256':digest(manifest),'selected_articles':len(ids),
            'article_status_counts':dict(Counter(r['status'] for r in statuses)),
            'figure_link_proposals':len(links),'usage_total_including_reused_calls':usage_total(receipts),
            'usage_note':'Cumulative recorded usage, including the earlier single-article call; not all newly billed this run.',
            'training_ready':0,'human_review':'skipped_by_user','articles':statuses,
            'link_proposals_sha256':digest(output)}
    dump(ROOT/'reports'/(run_id+'_summary.json'),report)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figures',type=Path,default=ROOT/'data/processed/fulltext_batch_66_v4_figures.jsonl')
    ap.add_argument('--limit',type=int,default=21)
    ap.add_argument('--max-tokens',type=int,default=8192)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--summary-only',action='store_true',help='No API calls, rebuild aggregate outputs only')
    ap.add_argument('--run-id',default='deepseek_batch',help='Separate summary and link output names')
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        ap.error('Invalid run ID')
    if not 1<=args.limit<=1000 or not 256<=args.max_tokens<=16384:
        ap.error('limit must be 1..1000; max-tokens must be 256..16384')
    ids=article_ids(args.figures,args.limit)
    if not ids:
        ap.error('No articles in figure manifest')
    # Preflight every prepared input before any paid request.
    pending=0
    preflight_issues={}
    for article in ids:
        try:
            folder=result_location(article,args.max_tokens)
            prepared=json.loads((ROOT/'data/model_inputs'/article/'input.json').read_text(encoding='utf-8'))
            if prepared.get('article_id')!=article:
                raise ValueError('Prepared article mismatch')
        except (ValueError,FileNotFoundError) as exc:
            preflight_issues[article]=str(exc)
            continue
        if not (folder/'receipt.json').exists():
            pending+=1
    plan={'selected_articles':len(ids),'new_unattempted_requests':pending,
                      'preflight_issues':preflight_issues,
                      'max_output_tokens_per_call':args.max_tokens,'thinking':'disabled',
                      'dry_run':args.dry_run,'summary_only':args.summary_only}
    dump(ROOT/'reports'/(args.run_id+'_plan.json'),plan)
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return
    key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not args.summary_only and pending and not key:
        ap.exit(1,'DEEPSEEK_API_KEY missing. Use the same VS Code terminal where it was configured.\n')
    if not args.summary_only:
        for index,article in enumerate(ids,1):
            print(f'[{index}/{len(ids)}] {article}',flush=True)
            if article in preflight_issues:
                print('Skipped before API call: '+preflight_issues[article],flush=True)
                continue
            try:
                call_one(article,key,args.max_tokens)
            except (ValueError,FileNotFoundError) as exc:
                print(str(exc),flush=True)
            report=summarize(ids,args.max_tokens,args.figures,args.run_id)
            # Stop billing attempts on account failures or throttling.
            if any(r.get('http_status') in (401,402,429) for r in report['articles']):
                print('Stopped on account/balance/rate-limit error. Inspect summary before continuing.')
                break
    report=summarize(ids,args.max_tokens,args.figures,args.run_id)
    print(json.dumps({k:v for k,v in report.items() if k!='articles'},indent=2))
    if report['article_status_counts'].get('validated',0)!=len(ids):
        raise SystemExit(1)


if __name__=='__main__':
    main()
