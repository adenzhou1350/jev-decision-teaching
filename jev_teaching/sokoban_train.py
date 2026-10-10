"""An actual closed-loop Sokoban policy: board encoder + shared candidate scorer.

The exact solver labels training states and evaluates optimal lengths offline.
The deployed policy only receives four board planes and legal candidate moves.
"""
import argparse
import copy
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .sokoban_env import ACTIONS, State, encode_planes, is_solved, make_dataset, step


class CandidatePolicy(nn.Module):
    def __init__(self, size=6):
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Flatten(),
            nn.Linear(64*size*size,128),nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(130,64),nn.ReLU(),nn.Linear(64,1))
        self.register_buffer('candidates',torch.tensor([[0.,-1.],[1.,0.],[0.,1.],[-1.,0.]]))

    def forward(self, boards):
        h = self.encoder(boards)
        c = self.candidates[None].expand(len(h),-1,-1)
        return self.scorer(torch.cat((h[:,None].expand(-1,4,-1),c),dim=-1)).squeeze(-1)


def legal_mask(layout,state):
    return [step(layout,state,a)[1] for a in range(4)]


def tensors(examples):
    x = torch.tensor([encode_planes(e.layout,e.state) for e in examples],dtype=torch.float32)
    legal = torch.tensor([legal_mask(e.layout,e.state) for e in examples],dtype=torch.bool)
    labels = torch.tensor([[float(a in e.optimal_actions) for a in range(4)] for e in examples])
    labels /= labels.sum(-1,keepdim=True)
    return x,legal,labels


def augment(x,legal,labels,rng):
    k = rng.randrange(4)
    x = torch.rot90(x,k,(-2,-1))
    legal,labels = torch.roll(legal,-k,-1),torch.roll(labels,-k,-1)
    if rng.random()<.5:
        x = x.flip(-1)
        legal,labels = legal[:,[0,3,2,1]],labels[:,[0,3,2,1]]
    return x,legal,labels


@torch.no_grad()
def action_metrics(model,data,device):
    x,legal,labels = data
    correct,loss = 0,0.
    for start in range(0,len(x),512):
        xx,ll,yy = x[start:start+512].to(device),legal[start:start+512].to(device),labels[start:start+512].to(device)
        logits = model(xx).masked_fill(~ll,-1e9)
        choice = logits.argmax(-1)
        correct += int((yy[torch.arange(len(xx),device=device),choice]>0).sum())
        loss += float(-(yy*logits.log_softmax(-1)).sum())
    return {'count':len(x),'correct':correct,'optimal_action_accuracy':correct/len(x),'nll':loss/len(x)}


def manhattan(a,b,width):
    return abs(a%width-b%width)+abs(a//width-b//width)


def greedy_action(layout,state):
    scored = []
    for a in range(4):
        nxt,moved = step(layout,state,a)
        if not moved:
            continue
        # Deliberately simple rule, no solver/lookahead/deadlock detector.
        distance = sum(min(manhattan(b,g,layout.width) for g in layout.goals) for b in nxt.boxes)
        approach = min(manhattan(nxt.player,b,layout.width) for b in nxt.boxes)
        scored.append((distance*10+approach,a))
    return min(scored)[1] if scored else 0


@torch.no_grad()
def rollout(model,examples,device='cpu',policy='after',max_steps=96,seed=42):
    """Fresh model decisions on reached states; never call an oracle here."""
    rng = random.Random(seed)
    states = [e.state for e in examples]
    seen = [{e.state} for e in examples]
    paths = [[] for _ in examples]
    outcomes = [None for _ in examples]
    for _ in range(max_steps):
        active = [i for i in range(len(examples)) if outcomes[i] is None]
        if not active:
            break
        if model is not None:
            boards = torch.tensor([encode_planes(examples[i].layout,states[i]) for i in active],dtype=torch.float32,device=device)
            legal = torch.tensor([legal_mask(examples[i].layout,states[i]) for i in active],device=device)
            choices = model(boards).masked_fill(~legal,-1e9).argmax(-1).cpu().tolist()
        else:
            choices = []
            for i in active:
                layout,state = examples[i].layout,states[i]
                allowed = [a for a,b in enumerate(legal_mask(layout,state)) if b]
                choices.append(rng.choice(allowed) if policy=='random' and allowed else greedy_action(layout,state))
        for i,a in zip(active,choices):
            e = examples[i]
            states[i],moved = step(e.layout,states[i],a)
            paths[i].append(ACTIONS[a])
            if is_solved(e.layout,states[i]):
                outcomes[i] = 'solved'
            elif not moved:
                outcomes[i] = 'no_legal_move'
            elif policy!='random' and states[i] in seen[i]:
                outcomes[i] = 'deterministic_loop'
            seen[i].add(states[i])
    trajectories = []
    for e,state,path,outcome in zip(examples,states,paths,outcomes):
        trajectories.append({'id':e.level_id,'layout_key':e.layout_key,'outcome':outcome or 'step_limit',
            'steps':len(path),'optimal_steps':e.distance,'actions':path,
            'final_player':state.player,'final_boxes':list(state.boxes)})
    wins = [t for t in trajectories if t['outcome']=='solved']
    return {'count':len(examples),'solved':len(wins),'success_rate':len(wins)/len(examples),
            'mean_steps_solved':statistics.mean(t['steps'] for t in wins) if wins else None,
            'mean_excess_steps_solved':statistics.mean(t['steps']-t['optimal_steps'] for t in wins) if wins else None,
            'max_steps':max_steps,'trajectories':trajectories}


def example_json(e,split):
    return {'id':e.level_id,'split':split,'width':e.layout.width,'height':e.layout.height,
            'walls':sorted(e.layout.walls),'goals':list(e.layout.goals),'player':e.state.player,
            'boxes':list(e.state.boxes),'layout_key':e.layout_key,'optimal_steps':e.distance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/sokoban-first'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--train-layouts',type=int,default=256)
    parser.add_argument('--train-starts',type=int,default=64)
    parser.add_argument('--eval-layouts',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261009)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new --out; previous evidence is never overwritten.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = random.Random(42)
    started = time.perf_counter()
    dataset = make_dataset(seed=args.seed,train_layouts=args.train_layouts,dev_layouts=args.eval_layouts,
                          test_layouts=args.eval_layouts,ood_layouts=12,train_starts=args.train_starts,eval_starts=4)
    data_seconds = time.perf_counter()-started
    print('Dataset:',{k:len(v) for k,v in dataset.items()},f'{data_seconds:.1f}s',flush=True)
    tensor_sets = {s:tensors(es) for s,es in dataset.items()}
    model = CandidatePolicy().to(args.device)
    before = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    x,legal,labels = tensor_sets['train']
    best,best_key,best_epoch,history = None,(-1.,-float('inf')),0,[]
    started = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        order = torch.randperm(len(x))
        train_loss = 0.
        for start in range(0,len(x),256):
            ids = order[start:start+256]
            xx,ll,yy = augment(x[ids].to(args.device),legal[ids].to(args.device),labels[ids].to(args.device),rng)
            logits = model(xx).masked_fill(~ll,-1e9)
            loss = -(yy*logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())*len(ids)
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            model.eval()
            dev = action_metrics(model,tensor_sets['dev'],args.device)
            episodes = rollout(model,dataset['dev'],args.device)
            key = (episodes['success_rate'],-dev['nll'])
            if key>best_key:
                best_key,best_epoch = key,epoch
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append({'epoch':epoch,'train_nll':train_loss/len(x),
                            'dev_optimal_action_accuracy':dev['optimal_action_accuracy'],
                            'dev_nll':dev['nll'],'dev_solved':episodes['solved'],'dev_total':episodes['count']})
            print(history[-1],flush=True)
    train_seconds = time.perf_counter()-started
    model.load_state_dict(best)
    model.eval()
    result = {'scope':'6x6 generated Sokoban; one-box train/dev/test and two-box OOD; compact board CNN, not an LLM',
              'seed':args.seed,'initialization_seed':42,'train_parameters':sum(p.numel() for p in model.parameters()),
              'architecture':'4 board planes -> CNN -> shared action-conditioned scalar scorer',
              'selection':'highest dev rollout success, then lowest dev NLL; test/OOD unused for selection',
              'selected_epoch':best_epoch,'history':history,'data_seconds':data_seconds,
              'train_seconds':train_seconds,'python':platform.python_version(),'torch':torch.__version__,
              'device':args.device,'splits':{}}
    all_levels = [example_json(e,s) for s,es in dataset.items() for e in es]
    result['dataset_sha256'] = hashlib.sha256(json.dumps(all_levels,sort_keys=True).encode()).hexdigest()
    if args.device=='cuda':
        result['gpu'] = torch.cuda.get_device_name()
    for split,examples in dataset.items():
        if split=='train':
            result['splits'][split] = {'states':len(examples),'layouts':len({e.layout_key for e in examples}),
                                      'after_actions':action_metrics(model,tensor_sets[split],args.device)}
            continue
        result['splits'][split] = {'states':len(examples),'layouts':len({e.layout_key for e in examples}),
            'before':rollout(before,examples,args.device,'before'),
            'after':rollout(model,examples,args.device,'after'),
            'random':rollout(None,examples,policy='random'),
            'greedy':rollout(None,examples,policy='greedy'),
            'after_actions':action_metrics(model,tensor_sets[split],args.device)}
        print(split,{k:f"{result['splits'][split][k]['solved']}/{len(examples)}" for k in ('before','after','random','greedy')},flush=True)
    # Deployment uses CPU to keep the tiny policy independent of the language-model GPU.
    model.cpu()
    probe = dataset['test'][0]
    latency = []
    for j in range(110):
        started = time.perf_counter()
        xx = torch.tensor([encode_planes(probe.layout,probe.state)],dtype=torch.float32)
        ll = torch.tensor([legal_mask(probe.layout,probe.state)])
        with torch.no_grad():
            model(xx).masked_fill(~ll,-1e9).softmax(-1)
        if j>=10:
            latency.append((time.perf_counter()-started)*1000)
    result['cpu_resident_inference_ms'] = {'count':len(latency),'p50':statistics.median(latency),
        'p95':sorted(latency)[94],'raw':latency,'scope':'board encoding + legal mask + forward; excludes HTTP and loading'}
    args.out.mkdir(parents=True)
    (args.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out/'levels.json').write_text(json.dumps(all_levels,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save({'kind':'sokoban-shared-scorer-v1','size':6,'actions':list(ACTIONS),
                'before':{k:v.detach().cpu() for k,v in before.state_dict().items()},
                'after':best,'dataset_sha256':result['dataset_sha256']},args.out/'policy.pt')
    print('Saved',args.out,flush=True)


if __name__=='__main__':
    main()
