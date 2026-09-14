#!/usr/bin/env python3
"""Post-hoc frozen command checkpoint audit on a declared development seed."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from sttw_control.env import RecoveryEnv,load_config
from sttw_control.network import make_policy_identity,load_policy
from sttw_control.evaluation import evaluate
from sttw_control.command_diagnostics import reconstruct,generate
from sttw_control.selection import rank_command_candidate


def metrics(path):
    x=reconstruct(path);t=x['trace'];end=min(1.,float(json.loads((path/'commands.json').read_text())['schedule'][1][0]))
    first=t['time'][:-1]<end
    return dict(failed=bool(t['terminated'][-1]),episode_return=float(sum(v.sum() for v in x['rewards'].values())),initial_speed_rmse=float(np.sqrt(np.mean(x['speed_error'][first]**2))),initial_yaw_rmse=float(np.sqrt(np.mean(x['yaw_error'][first]**2))))


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    def status(phase,**extra):(a.output/'status.json').write_text(json.dumps(dict(phase=phase,**extra),indent=2)+'\n')
    try:
        task=a.run/'frozen/task.json';cfg=load_config(task);env=RecoveryEnv(cfg)
        identity=make_policy_identity(env.bundle.identity,asdict(cfg),cfg.observation.history_steps)
        checkpoints=sorted((a.run/'training/checkpoints').glob('update_*'))
        (a.output/'declaration.json').write_text(json.dumps(dict(role='Post-hoc development audit; old frozen reward, not new-task results or holdout',seed=a.seed,alphas=list(cfg.priority.validation_alphas),task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),checkpoints=[str(p) for p in checkpoints]),indent=2)+'\n')
        for ck in checkpoints:
            status('evaluation',checkpoint=ck.name)
            policy=load_policy(ck,expected=identity);baseline=[];candidate=[]
            for i,alpha in enumerate(cfg.priority.validation_alphas):
                shared=a.output/'baseline'/f'alpha_{i}'
                if not shared.exists():evaluate(env,shared,seed=a.seed,priority_alpha=alpha)
                dest=a.output/ck.name/'evaluation'/f'alpha_{i}'/f'seed_{a.seed}'/'command_sequence'
                dest.mkdir(parents=True)
                (dest/'baseline').symlink_to(shared.resolve(),target_is_directory=True)
                evaluate(env,dest/'residual',seed=a.seed,priority_alpha=alpha,policy=policy,policy_identity=identity)
                baseline.append(metrics(shared));candidate.append(metrics(dest/'residual'))
            pack=lambda values:{k:[v[k] for v in values] for k in values[0]}
            rank,reason=rank_command_candidate(pack(candidate),pack(baseline))
            rows.append(dict(checkpoint=ck.name,rank=rank,reason=reason,metrics=candidate))
            (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            generate(a.output/ck.name)
        best=min((r for r in rows if r['rank'] is not None),key=lambda r:r['rank'])
        (a.output/'INDEX.md').write_text('# Frozen checkpoint development comparison\n\nPost-hoc audit using the original reward. Not independent test selection.\n\n'+ '\n'.join(f"- [{r['checkpoint']}]({r['checkpoint']}/analysis/INDEX.md): rank={r['rank']}" for r in rows)+f"\n\nSelected by declared development rank: {best['checkpoint']}\n")
        status('complete',best_checkpoint=best['checkpoint'])
    except Exception as e:status('error',error=repr(e));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=46001);run(p.parse_args())
