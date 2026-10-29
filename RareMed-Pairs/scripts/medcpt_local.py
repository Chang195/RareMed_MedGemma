"""Download pinned official MedCPT weights and run offline retrieval + reranking."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT/'models/medcpt'
# Keep all caches inside this project; public weights need no authentication.
os.environ.setdefault('HF_HOME', str(MODELS/'cache'))
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
REPOS = {name: 'ncbi/MedCPT-' + suffix for name, suffix in (
    ('query', 'Query-Encoder'), ('article', 'Article-Encoder'), ('cross', 'Cross-Encoder'))}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download():
    from huggingface_hub import HfApi, snapshot_download
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest_path = MODELS/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, repo in REPOS.items():
        if name in manifest:
            revision = manifest[name]['revision']
        else:
            revision = HfApi().model_info(repo).sha
        info = HfApi().model_info(repo, revision=revision)
        files = [f.rfilename for f in info.siblings]
        weight = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
        allowed = [f for f in files if f in {
            weight, 'config.json', 'tokenizer_config.json', 'tokenizer.json',
            'special_tokens_map.json', 'vocab.txt', 'README.md', 'LICENSE', 'LICENSE.txt'}]
        print(f'Downloading {repo} @ {revision}', flush=True)
        folder = MODELS/name
        snapshot_download(repo_id=repo, revision=revision, local_dir=folder,
                          allow_patterns=allowed, max_workers=2)
        manifest[name] = {'repo_id': repo, 'revision': revision, 'weight_file': weight,
                          'files_sha256': {f: sha(folder/f) for f in allowed}}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('All three official MedCPT models are downloaded.', flush=True)


def load(name, device):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    manifest = json.loads((MODELS/'manifest.json').read_text(encoding='utf-8'))
    folder = MODELS/name
    for filename, expected in manifest[name]['files_sha256'].items():
        if sha(folder/filename) != expected:
            raise ValueError('Model file checksum mismatch: ' + name + '/' + filename)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    cls = AutoModelForSequenceClassification if name == 'cross' else AutoModel
    model = cls.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                               use_safetensors=manifest[name]['weight_file'].endswith('.safetensors'))
    return tokenizer, model.to(device).eval()


def encode(texts, name, device):
    import torch
    tokenizer, model = load(name, device)
    output = []
    with torch.inference_mode():
        for i in range(0, len(texts), 4):
            batch = tokenizer(texts[i:i+4], padding=True, truncation=True,
                              max_length=64 if name == 'query' else 512, return_tensors='pt').to(device)
            output.append(model(**batch).last_hidden_state[:, 0, :].cpu())
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(output)


def corpus(article, section=None):
    if not article:
        # Synthetic fixtures only: this smoke test is not a retrieval accuracy benchmark.
        return [
            {'passage_id': 'synthetic_genetics', 'text': 'Genetic testing identified a heterozygous pathogenic NOTCH3 variant, supporting the diagnosis of CADASIL.'},
            {'passage_id': 'synthetic_mri', 'text': 'Brain MRI demonstrated bilateral white matter hyperintensities on FLAIR images.'},
            {'passage_id': 'synthetic_history', 'text': 'The patient presented with recurrent headaches and a family history of stroke.'},
            {'passage_id': 'synthetic_unrelated', 'text': 'The patient was treated with insulin for diabetes mellitus.'}]
    import re
    if not re.fullmatch(r'PMC\d+', article):
        raise ValueError('Expected an article ID such as PMC12675991')
    source = ROOT/'data/model_inputs'/article/'input.json'
    article_data = json.loads(source.read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS/'article', local_files_only=True)
    rows = []
    for passage in article_data['passages']:
        if section and passage.get('section') != section:
            continue
        text = passage['text']
        if not text.strip():
            continue
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        for start in range(0, len(offsets), 350):
            end = min(start + 400, len(offsets))
            a, b = offsets[start][0], offsets[end-1][1]
            rows.append({'article_id': article, 'passage_id': passage['passage_id'],
                         'section': passage.get('section'), 'start_in_passage': a, 'end_in_passage': b,
                         'passage_source_offset': passage['source_offset'], 'text': text[a:b]})
            if end == len(offsets):
                break
    if not rows:
        raise ValueError('No text passages')
    return rows


def search(query, article, device, top_k, section=None):
    # Enforce offline inference: no source text goes to any provider.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = corpus(article, section)
    print(f'Encoding {len(rows)} chunks on {device}; section={section or "all"}', flush=True)
    query_vector = encode([query], 'query', device)
    # Official article encoder uses text pairs (title, abstract). Here an empty title
    # avoids propagating a diagnosis-bearing paper title into every evidence chunk.
    doc_vectors = encode([['', row['text']] for row in rows], 'article', device)
    scores = (query_vector @ doc_vectors.T)[0]
    selected = torch.argsort(scores, descending=True)[:min(20, len(rows))].tolist()
    tokenizer, model = load('cross', device)
    reranked = []
    with torch.inference_mode():
        for i in range(0, len(selected), 4):
            ids = selected[i:i+4]
            pairs = [[query, rows[j]['text']] for j in ids]
            batch = tokenizer(pairs, padding=True, truncation='only_second', max_length=512,
                              return_tensors='pt').to(device)
            logits = model(**batch).logits.squeeze(-1).cpu().tolist()
            for j, score in zip(ids, logits):
                reranked.append({**rows[j], 'retrieval_inner_product': float(scores[j]),
                                 'reranker_logit': score})
    reranked.sort(key=lambda x: x['reranker_logit'], reverse=True)
    result = {'query': query, 'device': device, 'offline_inference': True,
              'corpus': article or 'synthetic_smoke_test', 'corpus_chunks': len(rows),
              'article_scope_only': True, 'case_scope_verified': False,
              'section_filter': section,
              'score_note': 'Ranking scores are not probabilities or evidence validation.',
              'model_manifest_sha256': sha(MODELS/'manifest.json'),
              'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'huggingface-hub')},
              'results': reranked[:top_k]}
    if article:
        result['source_sha256'] = sha(ROOT/'data/model_inputs'/article/'input.json')
    output = ROOT/'reports'/('medcpt_' + (article or 'smoke_test') + '.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Saved: ' + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'search'])
    parser.add_argument('--query', default='genetic testing evidence for CADASIL diagnosis')
    parser.add_argument('--article', help='Optional local PMCID; otherwise uses synthetic smoke-test text')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--section', help='Optional exact BioC section filter, e.g. CASE; this is not patient identity verification')
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error('top-k must be 1..20')
    if args.command == 'download':
        download()
    else:
        search(args.query, args.article, args.device, args.top_k, args.section)


if __name__ == '__main__':
    main()
