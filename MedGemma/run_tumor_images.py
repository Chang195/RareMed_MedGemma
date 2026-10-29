"""Original collector logic with official cloud image-location/download adapter."""
import argparse
import inspect
import json
from pathlib import Path
import runpy
import sys
import types
from urllib.parse import urlparse
import run_other_disease_search_1000 as search
from pmc_cloud_images import CloudClient, CloudHold
from pipeline_runtime_repair import atomic_dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('disease',choices=search.TOPICS)
    ap.add_argument('stage',choices=['plan','smoke','collect','prepare-inputs'])
    a=ap.parse_args()
    root,cfg,history=search.prepare(a.disease)
    receipt=search.original.read(root/'reports/brain_mri_fulltext_v3_receipt.json')
    manifest=root/'data/manifests/brain_mri_fulltext_v3.jsonl'
    if search.original.sha(manifest)!=receipt['output_sha256']:
        raise ValueError('Fulltext receipt/hash mismatch')
    config=search.original.read(root/'configs/acquire_v3.json')
    sys.path.insert(0,str(root/'scripts'))
    import collect
    if a.stage=='smoke':
        selected=[]
        for line in manifest.read_text(encoding='utf8').split('\n'):
            if not line.strip():
                continue
            row=json.loads(line)
            if row['queue']!='fulltext_passed':
                continue
            passages=collect.parse_bioc(root/row['fulltext_source_path'],row['article_id'])
            if any(p['metadata'].get('section_type')=='FIG' and p['metadata'].get('type') in {'fig_title_caption','fig_caption'}
                   and collect.caption_signals(p['text'])['mri'] for p in passages):
                selected.append(row)
            if len(selected)==3:
                break
        smoke=root/'data/manifests/cloud_smoke_mri_caption_candidates.jsonl'
        smoke.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf8')
        config={**config,'run_id':'collection_cloud_smoke_v3','max_articles':3,
                'screened_manifest':smoke.relative_to(root).as_posix()}
    config_path=root/'configs'/('acquire_cloud_smoke_v3.json' if a.stage=='smoke' else 'acquire_cloud_v1.json')
    atomic_dump(config_path,config)
    if a.stage=='prepare-inputs':
        import link_cases
        link_cases.prepare(root/'data/processed/collection_v3_figures.jsonl')
        return
    code=inspect.getsource(collect.collect_article)
    start=code.index('    try:\n        html = dl.get(')
    end=code.index('    # Preserve every source passage',start)
    code=code[:start]+'''    parser = types.SimpleNamespace(figures=cloud_client.mapping(pmcid, raw))
    if not parser.figures:
        raise CloudHold('held_cloud_no_figure_map', 'No exact JATS figure/media mapping')
'''+code[end:]
    client=CloudClient(search.original.HOME/'pmc-cloud-cache-v1')
    namespace={**collect.__dict__,'types':types,'cloud_client':client,'CloudHold':CloudHold}
    exec(compile(code,'cloud-adapted-original-collector','exec'),namespace)
    adapted=namespace['collect_article']
    def acquire(*args,**kwargs):
        try:
            figures,status=adapted(*args,**kwargs)
            for fig in figures:
                fig['acquisition_channel']='official_pmc_cloud'
                fig['cloud_mapping_receipt']='data/raw/'+fig['article_id']+'/cloud_mapping_receipt.json'
            return figures,status
        except CloudHold as exc:
            return [],{'article_id':args[0],'status':exc.status,'reason':str(exc),'training_ready':False}
    original_downloader=collect.Downloader
    class Downloader(original_downloader):
        def get(self,url,path):
            if urlparse(url).hostname=='pmc-oa-opendata.s3.amazonaws.com':
                result=client.materialize(url,path)
                self.events.append({'url':url,'path':str(path.relative_to(root)),
                    'sha256':search.original.sha(path),'channel':'official_pmc_cloud'})
                return result
            if urlparse(url).hostname=='pmc.ncbi.nlm.nih.gov':
                raise ValueError('Normal article HTML fetching disabled in cloud adapter')
            return super().get(url,path)
    collect.collect_article=acquire
    collect.Downloader=Downloader
    adapter_record={
        'original_collect_sha256':search.original.sha(root/'scripts/collect.py'),
        'adapter_sha256':search.original.sha(Path(__file__)),
        'cloud_client_sha256':search.original.sha(Path(__file__).with_name('pmc_cloud_images.py')),
        'config':config,'caption_and_mri_filter_unchanged':True,'license_filter_unchanged':True,
        'image_pixels_modified':False,'paid_api_calls':0,'channel':'official_pmc_cloud',
        'documentation':'https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/'}
    audit_path=root/'local_run'/('cloud_adapter_'+config['run_id']+'.json')
    if audit_path.exists() and search.original.read(audit_path)!=adapter_record:
        raise ValueError('Frozen cloud acquisition implementation/config changed; use a new run version')
    atomic_dump(audit_path,adapter_record)
    sys.argv=['acquire_screened.py','--config',str(config_path)]
    if a.stage=='plan':
        sys.argv.append('--dry-run')
    runpy.run_path(str(root/'scripts/acquire_screened.py'),run_name='__main__')


if __name__=='__main__':
    main()
