"""Acquire a bounded queue from an existing screened manifest. No new search."""
import argparse
from html import escape
import json
from pathlib import Path
import re
from collect import ROOT, Downloader, collect_article, dump, digest


def select_candidates(candidates, queues, excluded, limit):
    seen=set(excluded)
    selected=[]
    for queue in queues:
        for row in candidates:
            if row['queue']==queue and row['article_id'] not in seen:
                selected.append(row)
                seen.add(row['article_id'])
    return selected[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/fulltext_pilot.json')
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='Show article selection without fetching')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    run_id = cfg['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run ID')
    source = (ROOT / cfg['screened_manifest']).resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError('Manifest must be inside project')
    candidates = [json.loads(line) for line in source.read_text(encoding='utf-8').split('\n') if line.strip()]
    queues = cfg.get('queues', [cfg.get('queue')])
    excluded=set()
    exclusion_hashes={}
    for name in cfg.get('exclude_articles_from_receipts',[]):
        receipt_path=(ROOT/name).resolve()
        if not receipt_path.is_relative_to(ROOT):
            raise ValueError('Exclusion receipt must be inside project')
        prior_receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        excluded.update(r['article_id'] for r in prior_receipt['results'])
        exclusion_hashes[name]=digest(receipt_path)
    selected = select_candidates(candidates,queues,excluded,cfg['max_articles'])
    if args.dry_run:
        print(json.dumps({'selected_articles':len(selected),'previously_attempted_excluded':len(excluded),
                          'queues':queues},indent=2))
        return
    dl = Downloader(args.offline)
    records, results = [], []
    checkpoint = ROOT / 'data/manifests' / (run_id + '_checkpoint.json')
    previous = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
    if previous and (previous['config'] != cfg or previous['manifest_sha256'] != digest(source) or previous.get('exclusion_hashes',{})!=exclusion_hashes):
        raise ValueError('Checkpoint does not match config or manifest; choose a new run_id')
    completed = previous.get('completed', {}) if previous else {}
    for row in selected:
        pmcid = row['article_id']
        if not re.fullmatch(r'PMC\d+', pmcid):
            raise ValueError('Invalid PMCID')
        print('Processing ' + pmcid, flush=True)
        if pmcid in completed and completed[pmcid]['status']['status'] != 'failed':
            cached = completed[pmcid]
            # Recheck actual image bytes before reusing a completed article.
            for figure in cached['figures']:
                for asset in figure['assets']:
                    if digest(ROOT / asset['path']) != asset['sha256']:
                        raise ValueError('Checkpoint image hash mismatch: ' + asset['path'])
            records.extend(cached['figures'])
            results.append(cached['status'])
            print('  resumed: ' + cached['status']['status'], flush=True)
            continue
        try:
            figures, status = collect_article(pmcid, dl, mri_only=True,
                license_policy=cfg.get('license_policy','by4_only'),license_hint=row.get('article_license_hint',''))
            records.extend(figures)
            results.append(status)
        except Exception as exc:
            figures = []
            status = {'article_id': pmcid, 'status': 'failed', 'error': str(exc)}
            results.append(status)
        completed[pmcid] = {'status': status, 'figures': figures}
        dump(checkpoint, {'config': cfg, 'manifest_sha256': digest(source), 'exclusion_hashes':exclusion_hashes,'completed': completed})
        print('  ' + status['status'] + ': ' + str(len(figures)) + ' figures', flush=True)
    out = ROOT / 'data/processed' / (run_id + '_figures.jsonl')
    out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    receipt = {'run_id': run_id, 'config': cfg, 'manifest_sha256': digest(source),
        'previously_attempted_excluded':len(excluded),'exclusion_hashes':exclusion_hashes,
        'attempted_articles': len(selected), 'results': results, 'figure_candidates': len(records),
        'decoded_images': sum(len(r['assets']) for r in records), 'training_ready': 0,
        'output_sha256': digest(out), 'downloads': dl.events}
    dump(ROOT / 'reports' / (run_id+'_receipt.json'), receipt)
    cards = []
    for row in records:
        # Inline image payloads let the browser display raw .image files reliably.
        import base64
        imgs = []
        for asset in row['assets']:
            kind = {'JPEG':'jpeg', 'PNG':'png', 'GIF':'gif', 'WEBP':'webp'}.get(asset['format'])
            if kind:
                data = base64.b64encode((ROOT / asset['path']).read_bytes()).decode('ascii')
                imgs.append(f'<img src="data:image/{kind};base64,{data}">')
        cards.append('<section><h2>'+escape(row['sample_id'])+'</h2><a href="'+escape(row['source_url'])+'">'
            +escape(row['article_title'])+'</a>'+''.join(imgs)+'<h3>原文图注</h3><p>'+escape(row['caption'])
            +'</p><p>图注证据偏移：'+str(row['caption_evidence']['offset'])+'</p><p>风险标记：'
            +escape(', '.join(row['quality_flags']))+'</p></section>')
    summaries = ''.join('<li>'+escape(json.dumps(s, ensure_ascii=False))+'</li>' for s in results)
    (ROOT / 'reports' / (run_id+'_review.html')).write_text(
        '<!doctype html><meta charset="utf-8"><title>全文采集试跑</title>'
        '<style>body{font:17px/1.7 sans-serif;max-width:1100px;margin:40px auto}img{display:block;max-width:100%;max-height:800px;margin:20px 0}section{border-top:1px solid #aaa;padding:20px 0}</style>'
        f'<h1>全文采集：{len(selected)}篇试跑</h1><p>{len(records)}条MRI线索图注，'
        f'{receipt["decoded_images"]}幅已解码图像，训练就绪0。</p>'
        '<p>图注词语匹配不是视觉确认；多面板、混合模态、图像隐私和图级权利均待审核。原图未裁剪或修改。</p><ul>'
        +summaries+'</ul>'+''.join(cards), encoding='utf-8')
    from collections import Counter
    print(json.dumps({**{k:v for k,v in receipt.items() if k not in ('config','downloads','results')},
                      'article_status_counts':dict(Counter(r['status'] for r in results))}, indent=2))
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
