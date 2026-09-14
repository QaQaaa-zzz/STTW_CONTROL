#!/usr/bin/env python3
"""Bounded command-policy training, fixed-alpha panels and reward diagnostics."""
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
from dataclasses import replace,asdict
import jax
from sttw_control.env import RecoveryEnv,load_config
from sttw_control.network import make_policy_identity,load_policy
from sttw_control.evaluation import evaluate
from sttw_control.command_diagnostics import generate

def evaluate_panel(root,task,panel,checkpoint):
    cfg=load_config(task);env=RecoveryEnv(cfg)
    identity=make_policy_identity(env.bundle.identity,asdict(cfg),cfg.observation.history_steps)
    policy=load_policy(checkpoint,expected=identity)
    p=json.loads(panel.read_text())
    (root/'alpha_values.json').write_text(json.dumps({f'alpha_{i}':a for i,a in enumerate(p['alphas'])},indent=2)+'\n')
    for i,a in enumerate(p['alphas']):
        for seed in p['seeds']:
            for case in p['cases']:
                mc=replace(cfg.motion_commands,fixed=case['commands'])
                c=replace(cfg,motion_commands=mc)
                for label,pi in [('baseline',None),('residual',policy)]:
                    out=root/'evaluation'/f'alpha_{i}'/f'seed_{seed}'/case['name']/label
                    evaluate(RecoveryEnv(c),out,seed=seed,policy=pi,policy_identity=identity if pi else None,priority_alpha=a)
    rows=generate(root)
    if p.get('render',True):
        from sttw_control.media import render_run
        selected=[]
        for case in p['cases']:
            candidates=[x for x in rows if x['case']==case['name'] and x['policy']=='residual']
            best=min(candidates,key=lambda x:(x['failed'],-x['observed_seconds'] if x['failed'] else (x['speed_rmse']/cfg.motion_commands.speed_scale)**2+(x['yaw_rmse']/cfg.motion_commands.yaw_scale)**2))
            i=p['alphas'].index(best['alpha']);where=root/'evaluation'/f'alpha_{i}'/best['seed']/case['name']
            for label in ('baseline','residual'):render_run(where/label,fps=20)
            selected.append(dict(case=case['name'],alpha=best['alpha'],seed=best['seed'],directory=str(where)))
        (root/'analysis/media_selection.json').write_text(json.dumps(dict(criterion='Prefer full surviving episodes, then minimum fixed normalized speed+yaw squared error; failed-only cases choose longest survival. Display selection, not a training checkpoint ranking.',selected=selected),indent=2)+'\n')
        with (root/'analysis/INDEX.md').open('a') as f:
            f.write('\n## Recorded videos\n\n')
            for x in selected:
                for label in ('baseline','residual'):
                    path=Path(x['directory'])/label/'media/replay.mp4'
                    f.write(f"- [{x['case']} alpha={x['alpha']} {label}]({os.path.relpath(path,root/'analysis')})\n")

def run(a):
    cfg=load_config(a.task)
    if cfg.horizon_seconds>a.max_episode_seconds:raise ValueError('episode exceeds declared experiment duration cap')
    root=a.output;root.mkdir(parents=True,exist_ok=False);frozen=root/'frozen';frozen.mkdir()
    for name in ('task','training','panel'):(frozen/f'{name}.json').write_bytes(getattr(a,name).read_bytes())
    declaration=dict(role='Development command tracking, not holdout; conditioned lower policy only',endpoint='best_development' if json.loads((frozen/'training.json').read_text()).get('command_selection',False) else 'last',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('learning/src/sttw_control').glob('*.py')})
    (root/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    def status(phase,**kw):(root/'status.json').write_text(json.dumps(dict(phase=phase,**kw),indent=2)+'\n')
    env=dict(os.environ,PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl');env.pop('JAX_PLATFORMS',None)
    try:
        panel=json.loads((frozen/'panel.json').read_text())
        if not panel['alphas'] or any(not 0<=x<=1 for x in panel['alphas']):raise ValueError('invalid evaluation alpha')
        for case in panel['cases']:
            replace(cfg.motion_commands,fixed=case['commands'])
            if case['commands'][-1][0]>=cfg.horizon_seconds:raise ValueError('command changes must precede the episode endpoint')
        status('training')
        with (root/'training.log').open('x') as f:subprocess.run([sys.executable,'learning/cli/train.py','--task',str(frozen/'task.json'),'--config',str(frozen/'training.json'),'--output',str(root/'training')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        result=json.loads((root/'training/status.json').read_text())
        use_best=json.loads((frozen/'training.json').read_text()).get('command_selection',False)
        if use_best and not result['best_checkpoint']:raise RuntimeError('No finite development-selected checkpoint')
        checkpoint=Path(result['best_checkpoint'] if use_best else result['last_checkpoint'])
        status('evaluation',checkpoint=str(checkpoint));env['JAX_PLATFORMS']='cpu'
        with (root/'evaluation.log').open('x') as f:subprocess.run([sys.executable,__file__,'--evaluate','--output',str(root),'--task',str(frozen/'task.json'),'--panel',str(frozen/'panel.json'),'--checkpoint',str(checkpoint)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        status('complete',checkpoint=str(checkpoint))
    except Exception as e:status('error',error=repr(e));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true')
    p.add_argument('--max-episode-seconds',type=float,default=10.)
    for name in ('output','task','training','panel','checkpoint'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.evaluate:evaluate_panel(a.output,a.task,a.panel,a.checkpoint)
    else:run(a)
