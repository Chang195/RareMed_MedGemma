"""Independent read-only validation of exported mixtures, not model training."""
import collections,csv,hashlib,json,re
from pathlib import Path
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').split('\n') if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest={r['sample_id']:r for r in csv.DictReader((BASE/'manifests/sample_manifest.csv').open(encoding='utf-8-sig'))}
    ocr={r['sample_id']:r for r in csv.DictReader((BASE/'manifests/pixel_answer_leakage.csv').open(encoding='utf-8-sig'))}
    original={r['sample_id']:r['split'] for r in rows(BASE/'manifests/original_split_inventory.jsonl')}
    shared=collections.defaultdict(set);sizes=[];results=[];all_questions=set();training={}
    for arm,pct in [('A',20),('B',50),('C',80)]:
        folder=BASE/f'dataset_{arm}_{pct}pct';found=collections.defaultdict(lambda:collections.defaultdict(set))
        counts={};training[arm]={}
        for split in ['train','val','test']:
            p=folder/(split+'.jsonl');rr=rows(p);counts[split]=len(rr)
            if split!='train':shared[split].add(sha(p))
            ids=[r['sample_id'] for r in rr];assert len(ids)==len(set(ids)),(arm,split,'duplicate_sample')
            assert {r['binary_target'] for r in rr}=={'CADASIL','non-CADASIL'},(arm,split,'single_class')
            for r in rr:
                raw=manifest[r['sample_id']];img=(folder/r['image']).resolve()
                assert img.parent==(BASE/'images').resolve()
                assert re.fullmatch(r'[0-9a-f]{64}\.[a-z]+',img.name)
                assert sha(img)==r['metadata']['image_sha256']==raw['image_sha256']
                with Image.open(img) as im:im.load()
                assert r['diagnosis'] and r['diagnosis_evidence_level']
                assert r['messages'][0]['role']=='user' and r['messages'][1]['role']=='assistant'
                content=r['messages'][0]['content'];assert len(content)==2 and content[0]=={'type':'image'}
                assert content[1]['type']=='text';all_questions.add(content[1]['text'])
                assert r['messages'][1]['content']==[{'type':'text','text':r['binary_target']}]
                assert ocr[r['sample_id']]['diagnosis_term_flag']=='False'
                if split=='train':
                    assert original.get(raw['original_sample_id']) not in {'validation','test'}
                    training[arm][r['sample_id']]=r['binary_target']
                for field in ['source_case_id','source_article_id']:
                    if r.get(field):found[field][r[field]].add(split)
                for field in ['split_group_id','image_sha256','pixel_sha256']:
                    found[field][r['metadata'][field]].add(split)
            if split=='train':
                positives=sum(r['binary_target']=='CADASIL' for r in rr)
                assert positives*100==len(rr)*pct
        assert not [(k,v) for field,values in found.items() for k,v in values.items() if len(v)>1]
        sizes.append(counts['train']);results.append({'arm':arm,'counts':counts,'cross_split_failures':0})
    assert len(set(sizes))==1 and all(len(v)==1 for v in shared.values()) and len(all_questions)==1
    for label,order in [('CADASIL',['A','B','C']),('non-CADASIL',['C','B','A'])]:
        sets=[{k for k,v in training[a].items() if v==label} for a in order];assert sets[0]<=sets[1]<=sets[2]
    result={'status':'PASS_implemented_technical_checks','arms':results,'shared_heldout_hashes':{k:list(v)[0] for k,v in shared.items()},'limitations':'No claim of globally verified patient independence, clinical label gold status, perfect OCR or rights clearance.'}
    (BASE/'manifests/independent_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
