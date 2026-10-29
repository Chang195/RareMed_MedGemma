"""Read-only compatibility example. Does not import/train MedGemma or change SFT code."""
import argparse, collections, hashlib, json
from pathlib import Path
from PIL import Image

def load_split(path):
    path=Path(path)
    for line in path.read_text(encoding='utf8').split('\n'):
        if not line.strip():continue
        row=json.loads(line);image_path=(path.parent/row['image']).resolve()
        assert hashlib.sha256(image_path.read_bytes()).hexdigest()==row['metadata']['image_sha256']
        with Image.open(image_path) as im:image=im.convert('RGB').copy()
        # Only these two fields should enter the processor/template.
        # Do not serialize diagnosis/history/metadata into a user message.
        yield {'image':image,'messages':row['messages'],'sample_id':row['sample_id']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('jsonl',type=Path);args=ap.parse_args()
    n=0
    for item in load_split(args.jsonl):n+=1;item['image'].close()
    print(json.dumps({'readable_records':n,'file':str(args.jsonl)}))
