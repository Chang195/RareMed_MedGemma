"""Create a portable ZIP and verify every exported image inside it. No training."""
import hashlib,io,json,posixpath,zipfile
from pathlib import Path
from PIL import Image

BASE=Path(__file__).resolve().parents[1]

def main():
    target=BASE.parent/(BASE.name+'.zip')
    files=sorted(p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.log','.pyc'})
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:
            name=BASE.name+'/'+p.relative_to(BASE).as_posix()
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        checks=0
        for code,pct in [('A',20),('B',50),('C',80)]:
            for split in ['train','val','test']:
                path=f'{BASE.name}/dataset_{code}_{pct}pct/{split}.jsonl'
                for line in z.read(path).decode('utf8').split('\n'):
                    if not line.strip():continue
                    row=json.loads(line);image=posixpath.normpath(posixpath.join(posixpath.dirname(path),row['image']))
                    assert image.startswith(BASE.name+'/images/')
                    data=z.read(image);assert hashlib.sha256(data).hexdigest()==row['metadata']['image_sha256']
                    with Image.open(io.BytesIO(data)) as im:im.verify()
                    checks+=1
    checksum=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(checksum+'  '+target.name+'\n',encoding='ascii')
    print(json.dumps({'zip':str(target),'bytes':target.stat().st_size,'files':len(files),'portable_image_references_verified':checks,'sha256':checksum},indent=2))

if __name__=='__main__':main()
