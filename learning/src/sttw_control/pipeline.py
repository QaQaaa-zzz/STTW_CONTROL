"""Bounded training and declared fixed-alpha evaluation using canonical CLIs."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]


def evaluation_alphas(task,panel):
    values=panel.get('priority_alphas')
    if task.get('priority') is None:
        if values is not None:raise ValueError('priority alphas require a conditioned task')
        return [None]
    if not isinstance(values,list) or not values:raise ValueError('conditioned panel requires explicit priority_alphas')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in values) or len(set(values))!=len(values):
        raise ValueError('priority alphas must be finite, unique and within [0,1]')
    return list(map(float,values))


def select_endpoint(result,*,conditioned):
    return result['last_checkpoint'] if conditioned else result['best_checkpoint'] or result['last_checkpoint']


def source_identity():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('learning/src/sttw_control','learning/cli') for p in sorted((ROOT/folder).glob('*.py'))}


def run(task_path,training_path,panel_path,output):
    from .env import load_config
    from .training import TrainingConfig
    from .disturbance import plot_panel
    task_path,training_path,panel_path=map(Path,(task_path,training_path,panel_path))
    raw=json.loads(task_path.read_text());panel=json.loads(panel_path.read_text())
    cfg=load_config(task_path);training=TrainingConfig(**json.loads(training_path.read_text()))
    alphas=evaluation_alphas(raw,panel);conditioned=cfg.priority is not None
    seeds=panel['evaluation_seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):raise ValueError('evaluation seeds must be distinct integers')
    if set(seeds)&set(training.validation_seeds) or training.seed in set(seeds)|set(training.validation_seeds):raise ValueError('training, development and standard seeds must be separate')
    cases=panel.get('cases',[])
    count=1+len(cases)+len(panel.get('steer_rate_pulses',[]))+len(panel.get('lateral_forces',[]))
    horizon=max(cfg.horizon_seconds,panel['start_seconds']+max([panel['duration_seconds']]+[c['duration'] for c in cases])+panel.get('minimum_post_event_seconds',0.))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    frozen=output/'frozen';frozen.mkdir()
    for name,p in [('task',task_path),('training',training_path),('panel',panel_path)]:
        (frozen/f'{name}.json').write_bytes(p.read_bytes())
    sources=source_identity()
    declaration={'source_sha256':sources,'priority_alphas':alphas,
                 'endpoint':'fixed final update; priority controllability development' if conditioned else 'legacy best eligible or last fallback',
                 'training_budget':training.num_envs*training.rollout_steps*training.updates,
                 'standard_budget':len(alphas)*len(seeds)*count*2*math.ceil(horizon/cfg.controller.dt),
                 'media_selection':'first predeclared seed, all events, every declared alpha',
                 'role':'development; no holdout, energy saving or hardware claim'}
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    def status(phase,**extra):
        (output/'pipeline_status.json').write_text(json.dumps({'phase':phase,**extra},indent=2)+'\n')
    def invoke(args,log,cpu=False):
        if source_identity()!=sources:raise RuntimeError('source changed since stage declaration; refusing mixed-source execution')
        env=os.environ.copy()
        if cpu:env['JAX_PLATFORMS']='cpu'
        with log.open('x') as stream:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    train=output/'training'
    try:
        status('training',endpoint=declaration['endpoint'])
        invoke(['learning/cli/train.py','--task',frozen/'task.json','--config',frozen/'training.json','--output',train],output/'training.log')
        result=json.loads((train/'status.json').read_text())
        if not result['complete']:raise RuntimeError('training did not finish its declared budget')
        checkpoint=select_endpoint(result,conditioned=conditioned)
        rows=[];media_index=['# Complete standard recovery media','',declaration['media_selection'],'']
        for i,alpha in enumerate(alphas):
            prefix=f'alpha_{i}/' if conditioned else ''
            for seed in seeds:
                status('standard_evaluation',checkpoint=checkpoint,alpha=alpha,seed=seed,endpoint=declaration['endpoint'])
                out=output/'evaluation'/prefix/f'seed_{seed}'
                args=['learning/cli/disturbance.py','--task',frozen/'task.json','--panel',frozen/'panel.json','--training-run',train,'--checkpoint',checkpoint,'--output',out,'--seed',seed]
                if conditioned:args+=['--priority-alpha',alpha]
                invoke(args,output/f'evaluation_alpha_{i}_seed_{seed}.log',True)
                for line in (out/'results.jsonl').read_text().splitlines():
                    rows.append({'seed':seed,'priority_alpha':alpha,**json.loads(line)})
            status('media',checkpoint=checkpoint,alpha=alpha)
            first=output/'evaluation'/prefix/f'seed_{seeds[0]}'
            if source_identity()!=sources:raise RuntimeError('source changed before media generation')
            plot_panel(first)
            destination=output/'complete_media'/prefix if conditioned else output/'complete_media'
            invoke(['learning/cli/panel_media.py','--panel-run',first,'--output',destination],output/f'media_alpha_{i}.log',True)
            if conditioned:media_index.append(f'- alpha={alpha}: [all scenarios](alpha_{i}/INDEX.md)')
        if conditioned:(output/'complete_media'/'INDEX.md').write_text('\n'.join(media_index)+'\n')
        (output/'standard_results.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        status('complete',checkpoint=checkpoint,episodes=len(rows),endpoint=declaration['endpoint'],
               legacy_eligible_candidate_exists=result['best_checkpoint'] is not None,
               **({'development_eligible':result['best_checkpoint'] is not None} if not conditioned else {}),
               eligibility_scope='legacy speed/nominal gate, not priority controllability evidence')
    except Exception as exc:
        status('error',error=str(exc));raise
