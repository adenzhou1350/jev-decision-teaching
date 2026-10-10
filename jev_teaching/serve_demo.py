"""Serve real learned policies and a local interactive sandbox (stdlib HTTP)."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch

from .browser_data import apply_button, lexical_choice, records, visible_state
from .browser_train import encode, score_head
from .mini_jev import Encoder, fingerprint, make_head
from .sokoban_env import ACTIONS, Layout, State, encode_planes, is_solved, step, validate_state
from .sokoban_train import CandidatePolicy, greedy_action, legal_mask
from .paths import ARTIFACTS, WEB


class Demo:
    def __init__(self,artifacts,model_path=None,device='cpu'):
        torch.set_num_threads(4)
        self.artifacts = Path(artifacts)
        game_artifacts = self.artifacts / 'sokoban'
        self.browser_artifacts = self.artifacts / 'browser'
        self.model_path,self.device = model_path,device
        if model_path and not Path(model_path).is_dir():
            raise ValueError('--model must be a local snapshot directory so its weight fingerprint can be verified.')
        for name in ('sokoban-demo.pt','sokoban-levels.json','sokoban-results.json'):
            if not (game_artifacts/name).is_file():
                raise ValueError(f'Missing {game_artifacts/name}. --artifacts must contain a sokoban/ subdirectory. '
                                 'Use the bundled artifacts/ directory or export a complete demo first.')
        if model_path and not (self.browser_artifacts/'browser-demo.pt').is_file():
            raise ValueError(f'Missing {self.browser_artifacts / "browser-demo.pt"}; export a trained browser head first.')
        checkpoint = torch.load(game_artifacts/'sokoban-demo.pt',map_location='cpu',weights_only=True)
        self.policies = {}
        for name in ('before','after'):
            policy = CandidatePolicy(checkpoint['size'])
            policy.load_state_dict(checkpoint[name])
            self.policies[name] = policy.eval()
        self.levels = {r['id']:r for r in json.loads((game_artifacts/'sokoban-levels.json').read_text(encoding='utf-8'))
                       if r['split'] in ('test','ood')}
        self.game_report = json.loads((game_artifacts/'sokoban-results.json').read_text(encoding='utf-8'))
        self.browser_rows = {r['id']:r for r in records() if r['split'] in ('test','challenge')}
        self.browser_enabled = bool(model_path)
        self.browser_report = json.loads((self.browser_artifacts/'browser-results.json').read_text(encoding='utf-8')) if (self.browser_artifacts/'browser-results.json').exists() else None
        self.encoder = None
        self.heads = {}
        self.lock = threading.Lock()
        self.rng = random.Random(42)

    def info(self):
        test = self.game_report['splits']['test']
        browser_name = self.browser_report.get('model','语言模型') if self.browser_report else '语言模型'
        return {'title':'小模型决策实验室','ready':True,
                'model':'棋盘 CNN + 共享动作打分头（从零训练）',
                'metrics':{k:{'solved':test[k]['solved'],'total':test[k]['count']} for k in ('before','after','random','greedy')},
                'browser_available':self.browser_enabled,
                'browser_model':f'冻结 {browser_name} + 共享候选打分头',
                'browser_metrics':{k:{'correct':v['correct'],'total':v['count']} for k,v in self.browser_report['splits']['test'].items()} if self.browser_report else None,
                'browser_loaded':self.encoder is not None,
                'notes':'每次 AI 步进均重新运行策略；关卡来自留出集；求解器仅离线标注。'}

    @staticmethod
    def xy(cell,width):
        return [cell%width,cell//width]

    def public_levels(self):
        return [{'id':r['id'],'name':f"{'双箱挑战' if r['split']=='ood' else '单箱测试'} · {r['id']}",
                 'size':r['width'],'walls':[self.xy(c,r['width']) for c in r['walls']],
                 'goals':[self.xy(c,r['width']) for c in r['goals']],
                 'player':self.xy(r['player'],r['width']),
                 'boxes':[self.xy(c,r['width']) for c in r['boxes']],'split':r['split']}
                for r in self.levels.values()]

    def game_step(self,body):
        record = self.levels[body['level_id']]
        width,height = record['width'],record['height']
        layout = Layout(width,height,frozenset(record['walls']),tuple(record['goals']))
        def cell(pair):
            if not isinstance(pair,list) or len(pair)!=2 or any(type(v) is not int for v in pair):
                raise ValueError('Coordinates must be [integer x, integer y].')
            x,y = pair
            if not (0<=x<width and 0<=y<height):
                raise ValueError('Coordinates outside board.')
            return y*width+x
        state = State(cell(body['player']),tuple(cell(p) for p in body['boxes']))
        validate_state(layout,state)
        if is_solved(layout,state):
            raise ValueError('This level is already solved; reset or choose another.')
        kind = body.get('policy','after')
        if kind not in (*self.policies,'random','greedy'):
            raise ValueError('Unknown policy.')
        started = time.perf_counter()
        allowed = legal_mask(layout,state)
        if not any(allowed):
            raise ValueError('No legal move remains. Reset this level.')
        if kind in self.policies:
            with torch.no_grad():
                boards = torch.tensor([encode_planes(layout,state)],dtype=torch.float32)
                scores = self.policies[kind](boards)[0].masked_fill(~torch.tensor(allowed),-1e9)
                probabilities = scores.softmax(-1).tolist()
            action = max(range(4),key=lambda i:probabilities[i])
        else:
            ids = [i for i,v in enumerate(allowed) if v]
            action = self.rng.choice(ids) if kind=='random' else greedy_action(layout,state)
            probabilities = [float(v)/sum(allowed) for v in allowed] if kind=='random' else [float(i==action) for i in range(4)]
        latency = (time.perf_counter()-started)*1000
        new,moved = step(layout,state,action)
        return {'action':ACTIONS[action],'probabilities':dict(zip(ACTIONS,probabilities)),
                'latency_ms':latency,'model':kind,'moved':moved,'solved':is_solved(layout,new),
                'next_state':{'player':self.xy(new.player,width),'boxes':[self.xy(c,width) for c in new.boxes]}}

    def load_browser(self):
        if self.encoder is not None:
            return
        checkpoint = torch.load(self.browser_artifacts/'browser-demo.pt',map_location='cpu',weights_only=True)
        if fingerprint(self.model_path)!=checkpoint['model_fingerprint']:
            raise ValueError('Backbone files do not match this demo head; retrain the head for your snapshot.')
        encoder = Encoder(self.model_path,None,self.device,local_only=True)
        heads = {}
        for kind in ('before','after'):
            heads[kind] = make_head(checkpoint['width'])
            heads[kind].load_state_dict(checkpoint[kind])
            heads[kind].eval()
        self.encoder,self.heads = encoder,heads

    def browser_predict(self,body):
        if not self.browser_enabled:
            raise ValueError('Start server with --model PATH to enable the Qwen browser lesson.')
        row = deepcopy(self.browser_rows[body['case_id']])
        original_task = row['ui']['task']
        task = body.get('task',original_task)
        if not isinstance(task,str) or not task.strip() or len(task)>1000:
            raise ValueError('Task must be 1–1000 characters.')
        row['ui']['task'] = task
        row['state'] = visible_state(row['ui'])
        kind = body.get('policy','after')
        if kind not in ('before','after','lexical'):
            raise ValueError('Unknown browser policy.')
        with self.lock:
            was_loaded = self.encoder is not None
            started = time.perf_counter()
            if kind=='lexical':
                choice = lexical_choice(row)
                result = {'choice':choice,'probabilities':None}
            else:
                self.load_browser()
                feature = encode(self.encoder,[row])[0]
                result = score_head(self.heads[kind],feature,row)
            elapsed = (time.perf_counter()-started)*1000
        outcome = apply_button(row['ui'],result['choice'])
        result.update(latency_ms=elapsed,includes_model_loading=(kind!='lexical' and not was_loaded),
                      model=kind,outcome=outcome,task=task)
        # For user edits we show the effect, but never invent a reference label.
        if task==original_task:
            expected = apply_button(row['ui'],row['target'])
            result.update(correct=outcome==expected,reference_choice=row['target'])
        return result


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,payload,content_type='application/json; charset=utf-8'):
            data = json.dumps(payload,ensure_ascii=False).encode('utf-8') if content_type.startswith('application/json') else payload
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ('/','/demo.html'):
                self.reply(200,(WEB/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif path=='/api/info':
                self.reply(200,demo.info())
            elif path=='/api/levels':
                self.reply(200,{'levels':demo.public_levels()})
            elif path=='/api/browser-cases':
                self.reply(200,{'cases':[{'id':r['id'],'split':r['split'],'ui':r['ui']} for r in demo.browser_rows.values()]})
            elif path=='/favicon.ico':
                self.reply(204,b'','image/x-icon')
            else:
                self.reply(404,{'error':'Not found'})

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:
                    raise ValueError('Invalid request size.')
                body = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path=='/api/step':
                    self.reply(200,demo.game_step(body))
                elif path=='/api/browser-predict':
                    self.reply(200,demo.browser_predict(body))
                else:
                    self.reply(404,{'error':'Not found'})
            except (KeyError,ValueError,TypeError,json.JSONDecodeError) as error:
                self.reply(400,{'error':str(error)})
            except Exception as error:
                self.reply(500,{'error':f'{type(error).__name__}: {error}'})

        def log_message(self,format,*args):
            if args and str(args[1]) not in ('200','204'):
                super().log_message(format,*args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--artifacts',type=Path,default=ARTIFACTS,
                        help='Demo artifact root containing sokoban/ and optional browser/ (default: bundled artifacts/)')
    parser.add_argument('--model',help='Local Qwen snapshot directory; omit for the standalone Sokoban lesson')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    try:
        demo = Demo(args.artifacts,args.model,args.device)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    try:
        server = ThreadingHTTPServer((args.host,args.port),handler_for(demo))
    except OSError as error:
        parser.error(f'{error} (If the port is in use, try --port 8766.)')
    print(f'Demo ready at http://{args.host}:{args.port}; browser lesson={demo.browser_enabled}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()
