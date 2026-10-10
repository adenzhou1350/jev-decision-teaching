"""Train a frozen-language-model candidate scorer for a local browser sandbox."""
import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

from .mini_jev import Encoder, fingerprint, fit_head, make_head, metrics, validate
from .browser_data import records, lexical_choice


def decision_prompt(row, description):
    return (f"规则：{row['question']}\n{row['state']}\n"
            f"待评估按钮：{description}\n这个按钮是否完成当前要求？")


def encode(encoder, rows, batch_size=4):
    texts = [decision_prompt(row, desc) for row in rows for desc in row['candidates'].values()]
    flat = encoder.encode_texts(texts, batch_size=batch_size, max_length=512)
    return list(flat.split([len(row['candidates']) for row in rows]))


def percentile(values, fraction):
    """Empirical nearest-rank quantile; for 12 requests P95 is the maximum."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError('A non-empty sample and a fraction between 0 and 1 are required.')
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered)*fraction)-1)]


def public_model_identity(model):
    """Record the actual snapshot name without publishing a machine-local path."""
    path = Path(model)
    return path.name if path.is_dir() else model


def score_head(head, feature, row):
    with torch.no_grad():
        probs = head(feature).squeeze(-1).softmax(-1).tolist()
    keys = list(row['candidates'])
    choice = keys[max(range(len(keys)), key=lambda i: probs[i])]
    return {'choice': choice, 'probabilities': dict(zip(keys, probs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-0.8B',
                        help='Complete local snapshot directory; see docs/getting-started.md')
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--out', type=Path, default=Path('runs/browser-first'))
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error('--model must be a local snapshot directory, not a Hugging Face ID. '
                     'Download the model first (docs/getting-started.md); its file fingerprint must match the demo service.')
    if args.out.exists():
        parser.error('Choose a new --out; existing evidence is never overwritten.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rows = records()
    validate(rows)
    print(f'Loading frozen backbone; {len(rows)} decisions', flush=True)
    encoder = Encoder(args.model, None, args.device, args.local_only)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    features = encode(encoder, rows, args.batch_size)
    feature_seconds = time.perf_counter()-started
    print(f'Features extracted in {feature_seconds:.2f}s', flush=True)
    idx = {split: [i for i,r in enumerate(rows) if r['split']==split]
           for split in ('train','dev','test','challenge')}
    pick = lambda values, split: [values[i] for i in idx[split]]
    targets = [list(r['candidates']).index(r['target']) for r in rows]
    before = make_head(features[0].shape[-1]).eval()
    started = time.perf_counter()
    after, training = fit_head(pick(features,'train'),pick(targets,'train'),
                              pick(features,'dev'),pick(targets,'dev'),steps=args.steps,lr=0.003)
    train_seconds = time.perf_counter()-started
    result = {'scope':'authored local browser sandbox; not arbitrary website automation',
              'seed':42, 'model':public_model_identity(args.model), 'model_fingerprint':fingerprint(args.model),
              'dataset_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
              'python':platform.python_version(), 'torch':torch.__version__,
              'feature_seconds':feature_seconds,'head_train_seconds':train_seconds,
              'training':training, 'trainable_parameters':sum(p.numel() for p in after.parameters()),
              'backbone_trainable_parameters':0,'splits':{}}
    if args.device=='cuda':
        result.update(gpu=torch.cuda.get_device_name(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    for split in idx:
        rs,xs = pick(rows,split),pick(features,split)
        lex = [{'id':r['id'],'target':r['target'],'choice':lexical_choice(r)} for r in rs]
        result['splits'][split] = {
            'before':metrics(before,xs,rs), 'after':metrics(after,xs,rs),
            'lexical':{'count':len(rs),'correct':sum(p['choice']==p['target'] for p in lex),'predictions':lex}}
        print(split, {k:f"{v['correct']}/{v['count']}" for k,v in result['splits'][split].items()},flush=True)
    # Same resident encoder, fresh forward passes, serial request latency; no cached-feature claim.
    probes, workflow_counts = [], {}
    for row in pick(rows,'test'):
        workflow = row['ui']['workflow']
        if workflow_counts.get(workflow, 0) < 2:
            probes.append(row)
            workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1
    for row in probes[:2]:
        encode(encoder,[row],args.batch_size)
    latencies = []
    for row in probes:
        started = time.perf_counter()
        x = encode(encoder,[row],args.batch_size)[0]
        score_head(after,x,row)
        latencies.append((time.perf_counter()-started)*1000)
    result['resident_inference_ms'] = {'count':len(latencies),'p50':statistics.median(latencies),
                                       'p95':percentile(latencies,.95),'raw':latencies,
                                       'p95_method':'empirical nearest-rank',
                                       'probe_ids':[row['id'] for row in probes],
                                       'probe_workflows':workflow_counts,
                                       'scope':'tokenize + frozen backbone + head, single request, excludes loading and HTTP'}
    args.out.mkdir(parents=True)
    (args.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save({'kind':'frozen-qwen-browser-head-v1','model_id':public_model_identity(args.model),
                'model_fingerprint':result['model_fingerprint'], 'width':features[0].shape[-1],
                'before':before.state_dict(),'after':after.state_dict(),
                'dataset_sha256':result['dataset_sha256']},args.out/'heads.pt')
    print(f'Saved evidence and heads to {args.out}',flush=True)


if __name__=='__main__':
    main()
