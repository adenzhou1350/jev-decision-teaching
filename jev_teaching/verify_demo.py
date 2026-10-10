"""Exercise the actual HTTP service, replay its actions, retain every result."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen

from .browser_data import records
from .sokoban_env import ACTIONS, Layout, State, is_solved, step


def request(base,path,payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(base+path,data=data,headers={'Content-Type':'application/json'})
    started = time.perf_counter()
    with urlopen(req,timeout=180) as response:
        result = json.load(response)
    return result,(time.perf_counter()-started)*1000


def timing(samples):
    ordered = sorted(samples)
    return {'count':len(samples),'p50':statistics.median(samples),
            'p95':ordered[math.ceil(.95*len(ordered))-1],'raw_ms':samples,
            'scope':'resident local HTTP round trip, includes encoding, inference and serialization; excludes initial loading'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--out',type=Path,default=Path('runs/http-check.json'))
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new --out to preserve previous evidence.')
    info,_ = request(args.url,'/api/info')
    assert info['ready']
    levels,_ = request(args.url,'/api/levels')
    levels = levels['levels']
    result = {'scope':'actual localhost service; full held-out games and browser cases',
              'games':{},'browser':{},'failures':[]}
    game_times = []
    for policy in ('before','after'):
        for split in ('test','ood'):
            trajectories = []
            for record in (r for r in levels if r['split']==split):
                width = record['size']
                cell = lambda xy:xy[1]*width+xy[0]
                xy = lambda i:[i%width,i//width]
                layout = Layout(width,width,frozenset(map(cell,record['walls'])),tuple(map(cell,record['goals'])))
                state = State(cell(record['player']),tuple(map(cell,record['boxes'])))
                seen = {state}
                actions = []
                for _ in range(96):
                    response,elapsed = request(args.url,'/api/step',{'level_id':record['id'],
                        'player':xy(state.player),'boxes':[xy(b) for b in state.boxes],'policy':policy})
                    game_times.append(elapsed)
                    action = response['action']
                    state,moved = step(layout,state,ACTIONS.index(action))
                    assert moved
                    assert response['next_state']=={'player':xy(state.player),'boxes':[xy(b) for b in state.boxes]}
                    assert response['solved']==is_solved(layout,state)
                    assert abs(sum(response['probabilities'].values())-1)<1e-5
                    actions.append(action)
                    if is_solved(layout,state) or state in seen:
                        break
                    seen.add(state)
                trajectories.append({'id':record['id'],'solved':is_solved(layout,state),'actions':actions})
            result['games'][f'{policy}_{split}'] = {'solved':sum(t['solved'] for t in trajectories),
                'count':len(trajectories),'trajectories':trajectories}
            print('HTTP game',policy,split,result['games'][f'{policy}_{split}']['solved'],len(trajectories),flush=True)
    result['game_http_ms'] = timing(game_times)
    if info['browser_available']:
        rows = [r for r in records() if r['split'] in ('test','challenge')]
        cold,elapsed = request(args.url,'/api/browser-predict',{'case_id':rows[0]['id'],'policy':'after'})
        result['browser_initial_request'] = {'http_ms':elapsed,'includes_model_loading':cold['includes_model_loading']}
        browser_times = []
        for policy in ('before','after','lexical'):
            predictions = []
            for row in rows:
                response,elapsed = request(args.url,'/api/browser-predict',{'case_id':row['id'],'policy':policy})
                if policy=='after':
                    browser_times.append(elapsed)
                assert response['correct']==(response['choice']==row['target'])
                predictions.append({'id':row['id'],'split':row['split'],'target':row['target'],
                                    'choice':response['choice'],'correct':response['correct']})
            result['browser'][policy] = {'splits':{split:{'count':sum(p['split']==split for p in predictions),
                'correct':sum(p['correct'] and p['split']==split for p in predictions)} for split in ('test','challenge')},
                'predictions':predictions}
            print('HTTP browser',policy,result['browser'][policy]['splits'],flush=True)
        result['browser_http_ms'] = timing(browser_times)
        custom,_ = request(args.url,'/api/browser-predict',{'case_id':rows[0]['id'],'policy':'after','task':'这次先不要操作任何按钮。'})
        assert 'correct' not in custom and 'reference_choice' not in custom
        result['custom_task_suppresses_stale_gold'] = True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified actual HTTP results:',args.out,flush=True)


if __name__=='__main__':
    main()
