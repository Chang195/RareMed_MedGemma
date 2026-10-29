"""Disease-adapted original BioC fulltext screening. No article HTML/image scraping."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import run_other_disease_search_1000 as search
from pipeline_runtime_repair import atomic_dump

PATTERNS = {'gbm': r'\bglioblastoma\b|\bGBM\b', 'astrocytoma': r'\bastrocytomas?\b'}


def adapted_source(disease):
    source = search.original.SOURCE / 'scripts/screen_fulltext.py'
    text = source.read_text(encoding='utf8')
    anchor = 'from screen import PATTERNS'
    assert text.count(anchor) == 1
    text = text.replace(anchor, "from screen import PATTERNS as ORIGINAL_PATTERNS\n"
        + "PATTERNS = {'"+disease+"': "+repr(PATTERNS[disease])
        + ", 'mri': ORIGINAL_PATTERNS['mri']}" )
    text = text.replace("('cadasil','sporadic_csvd','mri')", repr((disease, 'mri')))
    text = text.replace("bool(hits['cadasil'] or hits['sporadic_csvd'])", "bool(hits['"+disease+"'])")
    return text


def seed_bioc(root, candidates):
    other = 'astrocytoma' if 'gbm' in root.name else 'gbm'
    other_root = search.original.HOME / ('original-pipeline-'+other+'-first1000-v1')
    copied = 0
    for row in candidates:
        aid = row['article_id']
        src = other_root / 'data/raw' / aid / 'article.bioc.xml'
        side = src.with_name(src.name+'.download.json')
        dest = root / 'data/raw' / aid / src.name
        if dest.exists() or not src.exists() or not side.exists():
            continue
        meta = search.original.read(side)
        expected = 'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/'+aid+'/unicode'
        if meta['url'] != expected or meta['sha256'] != search.original.sha(src):
            raise ValueError('Shared BioC download provenance mismatch: '+aid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src,dest)
        shutil.copy2(side,dest.with_name(dest.name+'.download.json'))
        copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('disease', choices=PATTERNS)
    ap.add_argument('stage', choices=['check', 'fulltext-plan', 'fulltext'])
    a = ap.parse_args()
    root, cfg, historical = search.prepare(a.disease)
    receipt = search.original.read(root/'local_run/history_filter_receipt.json')
    manifest = root / 'data/manifests' / (cfg['run_id']+'_sources.json')
    if search.original.sha(manifest) != receipt['filtered_sha256']:
        raise ValueError('Frozen filtered search manifest changed')
    candidates = search.original.read(manifest)
    if any(r['article_id'] in historical['article_ids'] for r in candidates) or len(candidates)>1000:
        raise ValueError('History/first-page limit violation')
    script = root / 'scripts' / ('screen_fulltext_'+a.disease+'.py')
    text = adapted_source(a.disease)
    if script.exists() and script.read_text(encoding='utf8') != text:
        raise ValueError('Adapted fulltext code changed')
    script.write_text(text, encoding='utf8')
    sys.path.insert(0,str(root/'scripts'))
    original_stage = importlib.import_module('run_collection_v3')
    metadata, input_hashes = importlib.import_module('audit_licenses').read_metadata(cfg)
    rows = original_stage.build_candidates(candidates, metadata)
    target = root / 'data/manifests/brain_mri_all_candidates_v3.jsonl'
    contents = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if target.exists() and target.read_text(encoding='utf8') != contents:
        raise ValueError('Fulltext input queue changed')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(contents,encoding='utf8')
    audit = {'disease':a.disease,'candidate_articles':len(rows),'first_page_only':True,
        'history_overlap':0,'target_pattern':PATTERNS[a.disease],
        'mri_pattern_unchanged':True,'review_retention_unchanged':True,'license_rule_unchanged':True,
        'prompt_schema_unchanged':True,'input_sha256':search.original.sha(target),
        'original_fulltext_sha256':search.original.sha(search.original.SOURCE/'scripts/screen_fulltext.py'),
        'adapted_fulltext_sha256':search.original.sha(script),
        'paid_api_calls':0,'image_acquisition_started':False}
    atomic_dump(root/'local_run/fulltext_disease_adaptation.json',audit)
    print(json.dumps(audit,indent=2),flush=True)
    if a.stage != 'fulltext':
        return
    print('Verified BioC cache reused from other tumor search: '+str(seed_bioc(root,rows)),flush=True)
    # Original config, same review/license/MRI rules and checkpoint implementation.
    subprocess.run([sys.executable,str(script),'--config',str(root/'configs/fulltext_screening_v3.json')],
                   cwd=root,check=True)


if __name__=='__main__':
    main()
