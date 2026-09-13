#!/usr/bin/env python3
"""Execute every declared comparison arm sequentially; preserve failed artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from sttw_control.deferred_training import verify_files


def run(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    from sttw_control.env import load_config
    from sttw_control.training import TrainingConfig
    for arm in plan['arms']:
        load_config(arm['task']);TrainingConfig(**json.loads(Path(arm['training']).read_text()))
        panel=json.loads(Path(arm['panel']).read_text())
        for key in ('evaluation_seeds','start_seconds','duration_seconds','path_tolerance_m','extra_tolerance_m','heading_tolerance_rad','hold_seconds'): 
            if key not in panel:raise ValueError('panel missing '+key)
    root=Path(plan['output']);root.mkdir(exist_ok=False)
    completed=[]
    def status(phase,**extra):
        temp=root/'status.tmp';temp.write_text(json.dumps(dict(phase=phase,completed=completed,**extra),indent=2)+'\n');temp.replace(root/'status.json')
    env=dict(os.environ,PYTHONPATH=str(Path(plan['repository'])/'learning/src'),MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1');env.pop('JAX_PLATFORMS',None)
    def invoke(args,log,cpu=False):
        verify_files(plan['input_sha256']); e=dict(env)
        if cpu:e['JAX_PLATFORMS']='cpu'
        with log.open('x') as f:subprocess.run([sys.executable,*map(str,args)],cwd=plan['repository'],env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        for arm in plan['arms']:
            verify_files(plan['input_sha256']); out=root/arm['name'];out.mkdir();frozen=out/'frozen';frozen.mkdir()
            for key in ('task','training','panel'):(frozen/f'{key}.json').write_bytes(Path(arm[key]).read_bytes())
            (out/'declaration.json').write_text(json.dumps(dict(priority_alphas=arm['alphas'],arm=arm,source_sha256=plan['input_sha256'],role='development comparison; not holdout'),indent=2)+'\n')
            status('training',arm=arm['name']);invoke(['learning/cli/train.py','--task',frozen/'task.json','--config',frozen/'training.json','--output',out/'training'],out/'training.log')
            st=json.loads((out/'training/status.json').read_text());assert st['complete']
            checkpoint=(st['best_checkpoint'] or st['last_checkpoint']) if arm['endpoint']=='best_or_last' else st['last_checkpoint']
            panel=json.loads((frozen/'panel.json').read_text())
            for i,a in enumerate(arm['alphas']):
                for seed in panel['evaluation_seeds']:
                    status('evaluation',arm=arm['name'],alpha=a,seed=seed)
                    args=['learning/cli/disturbance.py','--task',frozen/'task.json','--panel',frozen/'panel.json','--training-run',out/'training','--checkpoint',checkpoint,'--output',out/'evaluation'/f'alpha_{i}'/f'seed_{seed}','--seed',seed]
                    if a is not None:args+=['--priority-alpha',a]
                    invoke(args,out/f'evaluation_{i}_{seed}.log',True)
            status('diagnostics',arm=arm['name']);invoke(['learning/cli/reward_breakdown.py','--run',out],out/'diagnostics.log',True)
            completed.append(dict(arm=arm['name'],checkpoint=checkpoint,training_steps=st['control_transitions']))
        status('analysis')
        from sttw_control.campaign_analysis import summarize
        summarize(plan_path)
        status('complete')
        (root/'INDEX.md').write_text('# Comparison campaign\n\n'+ '\n'.join(f"- [{x['arm']}]({x['arm']}/analysis/reward_breakdown/INDEX.md)" for x in completed)+'\n\nCompare physical metrics within matched tasks; returns across reward definitions are not comparable.\n')
    except Exception as exc:
        status('error',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);run(p.parse_args().plan)
