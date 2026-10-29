"""Local-only pilot workflow. Never creates medical targets or an evaluation split."""
import os
from pathlib import Path
import argparse, copy, hashlib, json, math, os, random, re, sys
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('WANDB_DISABLED', 'true')
ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))))))
os.environ.setdefault('HF_HOME', str(LOCAL / 'huggingface'))

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def validate(root, out):
    from PIL import Image
    root = root.resolve()
    for name in ['images', 'sft_pilot.jsonl', 'manifest.json', 'preview.html', 'held.json', 'loading_check.json']:
        if not (root/name).exists(): raise ValueError(f'Missing {name} in {root}')
    for name in ['manifest.json', 'held.json', 'loading_check.json']:
        json.loads((root/name).read_text(encoding='utf-8-sig'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    dataset_digest=hashlib.sha256((root/'sft_pilot.jsonl').read_bytes()).hexdigest()
    if manifest.get('dataset_sha256') != dataset_digest: raise ValueError('Manifest dataset SHA256 mismatch')
    rows, checks = [], []
    for line, text in enumerate((root/'sft_pilot.jsonl').read_text(encoding='utf-8-sig').splitlines(), 1):
        if not text.strip(): continue
        r = json.loads(text)
        rel = Path(r['image'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe image path')
        p = (root/rel).resolve()
        if not p.is_relative_to(root): raise ValueError('Image escapes dataset')
        raw = p.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load(); size, mode = im.size, im.mode
        messages = r['messages']
        if [m['role'] for m in messages] != ['user','assistant']:
            raise ValueError('Pilot requires exactly user + assistant')
        if not isinstance(r['metadata'], dict): raise ValueError('metadata must be an object')
        if digest.lower() != str(r['metadata'].get('image_sha256','')).lower():
            raise ValueError(f'Line {line}: SHA256 absent or mismatched in metadata; inspect schema')
        content = messages[0]['content']
        if not isinstance(content, list) or sum(c.get('type') == 'image' for c in content) != 1:
            raise ValueError('Expected one structured image placeholder')
        for c in content:
            if c.get('type') not in ('image', 'text'): raise ValueError('Unsupported user content')
            if c.get('type') == 'text' and not c.get('text', '').strip(): raise ValueError('Empty question')
        target = messages[1]['content']
        if isinstance(target, list):
            if any(c.get('type') != 'text' for c in target): raise ValueError('Assistant must be text only')
            target = ''.join(c['text'] for c in target)
        if not isinstance(target, str) or not target.strip(): raise ValueError('Empty target')
        meta=r['metadata']; evidence=meta['target_evidence']
        if target != evidence['text']: raise ValueError('Target differs from recorded evidence')
        start=evidence['caption_start']
        if meta['original_caption'][start:start+len(target)] != target: raise ValueError('Caption substring mismatch')
        if meta.get('training_ready') is not False: raise ValueError('Expected unreviewed pilot training_ready=false')
        if meta.get('split') != 'unassigned': raise ValueError('Pilot must remain unsplit')
        checks.append(dict(line=line, image=r['image'], sha256=digest, size=size, mode=mode,
                           hash_verified=True, transforms='processor defaults only; no manual crop/resize/arrow removal'))
        rows.append(r)
    if not rows: raise ValueError('Empty dataset')
    if manifest.get('examples') != len(rows): raise ValueError('Manifest sample count mismatch')
    write(out/'dataset-audit.json', dict(samples=checks, dataset_sha256=hashlib.sha256((root/'sft_pilot.jsonl').read_bytes()).hexdigest(),
          training_ready=False, purpose='smoke/overfit only; no clinical reference validation; no train/test split'))
    return rows

def conversation(row, root):
    from PIL import Image
    m = copy.deepcopy(row['messages'])
    # Preserve pixels/annotations. RGB conversion is recorded; never save over source.
    with Image.open(root/row['image']) as im: image = im.convert('RGB')
    for i,c in enumerate(m[0]['content']):
        if c['type']=='image': m[0]['content'][i] = {'type':'image', 'image':image}
    if isinstance(m[1]['content'], str): m[1]['content']=[{'type':'text','text':m[1]['content']}]
    return m

def encode(processor, messages):
    import torch
    kw = dict(tokenize=True, return_dict=True, return_tensors='pt', padding=True, truncation=False)
    prompt = processor.apply_chat_template(messages[:1], add_generation_prompt=True, **kw)
    full = processor.apply_chat_template(messages, add_generation_prompt=False, **kw)
    n = prompt['input_ids'].shape[1]
    assert torch.equal(prompt['input_ids'], full['input_ids'][:,:n]), 'Chat prefix mismatch; mask unsafe'
    ids = full['input_ids']; labels = torch.full_like(ids, -100)
    # Only exact assistant text tokens; exclude turn delimiters and all image/special tokens.
    target = ''.join(c['text'] for c in messages[1]['content'])
    target_ids = processor.tokenizer(target, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    assert len(target_ids)>0
    assert torch.equal(ids[0,n:n+len(target_ids)], target_ids), 'Answer boundary mismatch; inspect template'
    labels[0,n:n+len(target_ids)] = target_ids
    labels[full['attention_mask']==0]=-100
    for token in processor.tokenizer.all_special_ids: labels[ids==token]=-100
    image_id = processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')
    assert (ids == image_id).any(), 'Image tokens missing'
    assert (labels[ids==image_id]==-100).all()
    assert (labels[:,:n]==-100).all() and (labels!=-100).any()
    assert (labels[full['attention_mask']==0]==-100).all()
    full['labels']=labels
    return full, prompt, n

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage', choices=['validate','processor','forward','smoke'])
    p.add_argument('--data', type=Path, default=LOCAL/'data/caption-pilot-v0.1')
    p.add_argument('--out', type=Path, default=LOCAL/'runs/pilot')
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--precision', choices=['fp16','fp32'], default='fp32')
    a=p.parse_args(); a.data=a.data.resolve()
    if not 1<=a.steps<=5: raise ValueError('Smoke limited to 1..5 steps')
    if a.stage=='smoke' and any((a.out/name).exists() for name in ('baseline.json','adapter','smoke-log.json')):
        raise ValueError('This output already contains baseline/training results. Choose a new --out for a new run.')
    a.out.mkdir(parents=True, exist_ok=True)
    rows=validate(a.data,a.out)
    print(f'Dataset PASS: {len(rows)} samples', flush=True)
    if a.stage=='validate': return
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
    random.seed(42); torch.manual_seed(42)
    model_id='google/medgemma-1.5-4b-it'
    revision_file=a.out/'model-revision.json'
    if revision_file.exists(): revision=json.loads(revision_file.read_text())['sha']
    else:
        access_file=LOCAL/'runs/model-access/access.json'
        access=json.loads(access_file.read_text()) if access_file.exists() else {}
        revision=access['sha'] if access.get('status')=='PASS' else HfApi().model_info(model_id).sha
        write(revision_file,dict(model=model_id,sha=revision))
    processor=AutoProcessor.from_pretrained(model_id,revision=revision,use_fast=False)
    processor.save_pretrained(a.out/'processor')
    encoded=[]; prompts=[]
    for i,r in enumerate(rows):
        full,prompt,n=encode(processor,conversation(r,a.data)); encoded.append(full); prompts.append(prompt)
        ids=full['input_ids'][0]; labels=full['labels'][0]
        write(a.out/f'mask-{i}.json',dict(prefix_tokens=n, supervised_tokens=int((labels!=-100).sum()),
            input_tokens=len(ids), tokens=[dict(index=j,id=int(t),token=processor.tokenizer.convert_ids_to_tokens(int(t)),label=int(labels[j])) for j,t in enumerate(ids)],
            image_processing=processor.image_processor.to_dict(), rgb_conversion=True, truncation=False))
    assert processor.tokenizer.pad_token_id is not None
    # Check the same masking rule on actual processor padding for unequal prompts.
    padding_probe = processor.tokenizer(['padding probe', 'padding probe with a longer suffix'], padding=True, return_tensors='pt')
    probe_labels = padding_probe['input_ids'].clone()
    probe_labels[padding_probe['attention_mask']==0] = -100
    assert (padding_probe['attention_mask']==0).any()
    assert (probe_labels[padding_probe['attention_mask']==0]==-100).all()
    write(a.out/'processor-check.json',dict(status='PASS', samples=len(encoded),
          template=processor.chat_template, supervision='exact assistant text only; special/image/user/padding excluded'))
    print('Processor and assistant mask PASS', flush=True)
    if a.stage=='processor': return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    compute_dtype=torch.float32 if a.precision=='fp32' else torch.float16
    torch.cuda.reset_peak_memory_stats()
    write(a.out/'invocation.json',dict(argv=sys.argv, python=sys.version, seed=42, model=model_id, revision=revision,
          steps=a.steps,precision=a.precision,attention='sdpa', image_processing=processor.image_processor.to_dict(), truncation=False, manual_crop=False,
          arrow_removal=False, rgb_conversion=True, training_ready=False, purpose='smoke/overfit only'))
    model=Gemma3ForConditionalGeneration.from_pretrained(model_id, revision=revision,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=compute_dtype),
        dtype=compute_dtype, device_map={'':0}, attn_implementation='sdpa')
    model.config.use_cache=False
    def device_batch(b): return {k:v.to('cuda') for k,v in b.items()}
    model.eval()
    with torch.no_grad(): loss=model(**device_batch(encoded[0])).loss
    write(a.out/f'forward-{a.precision}.json',dict(loss=float(loss) if torch.isfinite(loss) else str(float(loss)),finite=bool(torch.isfinite(loss)),precision=a.precision,peak_vram=torch.cuda.max_memory_allocated()))
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite forward loss')
    write(a.out/'forward.json',dict(loss=float(loss),finite=True,precision=a.precision,attention='sdpa',peak_vram=torch.cuda.max_memory_allocated()))
    print(f'Forward PASS loss={float(loss)}',flush=True)
    if a.stage=='forward': return
    # Baseline must exist before any adapter is attached or optimization starts.
    baseline=[]
    with torch.no_grad():
        for r,b in zip(rows,prompts):
            b=device_batch(b)
            generated=model.generate(**b,max_new_tokens=128,do_sample=False,use_cache=True)
            baseline.append(dict(image=r['image'], generated=processor.decode(generated[0,b['input_ids'].shape[1]:],skip_special_tokens=True)))
    write(a.out/'baseline.json',dict(results=baseline, interpretation='training-sample smoke only; no generalization metric'))
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    # Restrict adapters to language attention. Freeze vision tower explicitly.
    targets=[name for name,module in model.named_modules() if name.endswith(('q_proj','v_proj')) and 'language_model' in name]
    assert targets, 'No language adapter targets'
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=targets,task_type='CAUSAL_LM'))
    model.train(); model.print_trainable_parameters()
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=2e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=a.precision=='fp16')
    log=[]
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=a.precision=='fp16'): loss=model(**device_batch(encoded[step%len(encoded)])).loss
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.0,error_if_nonfinite=True)
        scaler.step(opt); scaler.update()
        log.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),peak_vram=torch.cuda.max_memory_allocated()))
        write(a.out/'smoke-log.json',log); print(log[-1],flush=True)
    model.save_pretrained(a.out/'adapter',safe_serialization=True)
    torch.save(dict(optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(),step=a.steps),a.out/'adapter/training-state.pt')
    write(a.out/'run-config.json',dict(seed=42,steps=a.steps,learning_rate=2e-5,rank=4,alpha=8,
          targets=targets,quantization=f'NF4 double quant {a.precision} compute',attention='sdpa',max_new_tokens=128,do_sample=False,
          training_ready=False,purpose='smoke/overfit only',split=None,argv=sys.argv))

if __name__=='__main__': main()
