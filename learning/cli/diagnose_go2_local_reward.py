"""Reconstruct the frozen new reward on saved candidate and zero-ECBC traces."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--training-run',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--additional-evaluation',type=Path)
p.add_argument('--selected-model',type=Path)
a=p.parse_args()
task=json.loads((a.training_run/'config.json').read_text())['task']
s=json.loads(a.config.read_text())
r=task['reward'];dt=task['control_dt']
selection=json.loads((a.training_run/'best_training.json').read_text())
best=a.selected_model or a.evaluation/Path(selection['best']['checkpoint']).stem
folders=sorted(a.evaluation.glob('model_*'))+[a.evaluation/'zero_ecbc']
if a.additional_evaluation:folders+=sorted(a.additional_evaluation.glob('model_*'))
for folder in folders:
    d=np.load(folder/'trajectory.npz')
    previous=np.concatenate([np.zeros_like(d['action'][:1]),d['action'][:-1]])
    ev=d['speed']-d['reference'][:,:,0]
    ed=d['measurement'][:,:,2]-d['reference'][:,:,1]
    values=dict(speed=np.exp(-(ev/r['speed_scale'])**2),
        steer=np.exp(-(ed/r['steer_scale'])**2),
        roll=-np.maximum(d['peak_roll']-r['roll_start'],0)**2,
        roll_rate=-d['measurement'][:,:,1]**2,
        action=-np.sum(d['action']**2,axis=-1),
        action_change=-np.sum((d['action']-previous)**2,axis=-1))
    parts={k:np.where(d['valid']&~d['failed'],dt*r['weights'][k]*v,0.) for k,v in values.items()}
    parts['failure']=np.where(d['valid']&d['failed'],-r['failure_penalty'],0.)
    total=sum(parts.values())
    np.savez_compressed(folder/'reward_parts.npz',**parts,total=total,cumulative=np.cumsum(total,axis=0),
        speed_error=ev,steer_error=ed,valid=d['valid'])
policy=np.load(best/'trajectory.npz');pr=np.load(best/'reward_parts.npz')
zero=np.load(a.evaluation/'zero_ecbc/trajectory.npz');zr=np.load(a.evaluation/'zero_ecbc/reward_parts.npz')
for i,case in enumerate(s['cases']):
    fig,axs=plt.subplots(3,2,figsize=(13,11))
    ref=np.load(Path(s['mujoco_reference'])/(case+'.npz'))
    axs[0,0].plot(*ref['reference_xy'].T,':',color='gray',label='Original reference')
    for d,rew,label,color,style in [(policy,pr,'Selected student','#0072B2','-'),(zero,zr,'ECBC+ESO zero residual','#D55E00','--')]:
        n=int(d['valid'][:,i].sum());t=d['time'][:n,i]+d['stop_tick'][:n,i]*s['physics_dt']
        xy=d['pose'][:n,i,:2]
        axs[0,0].plot(*xy.T,style,color=color,label=label);axs[0,0].scatter(*xy[-1],marker='x',color=color)
        axs[0,1].plot(np.arange(1,n+1),rew['total'][:n,i],style,color=color,label=label)
        axs[1,0].plot(np.arange(1,n+1),rew['cumulative'][:n,i],style,color=color,label=label)
        axs[1,1].plot(t,d['speed'][:n,i],style,color=color,label=label+' body')
        axs[1,1].plot(t,d['measurement'][:n,i,5]*.1,':',color=color,label=label+' wheel proxy')
        axs[2,0].plot(t,rew['steer_error'][:n,i],style,color=color,label=label)
        axs[2,1].plot(t,d['measurement'][:n,i,0],style,color=color,label=label)
        if d['failed'][:,i].any():
            for ax in axs.flat:
                if ax is not axs[0,0]:ax.axvline(n if ax in (axs[0,1],axs[1,0]) else t[-1],color=color,alpha=.3)
    axs[1,1].plot(policy['time'][:,i]+dt,policy['reference'][:,i,0],':',color='gray',label='Requested speed')
    axs[0,0].set_aspect('equal',adjustable='box');axs[0,0].set(xlabel='World x (m)',ylabel='World y (m)')
    for ax,title in zip(axs.flat,['XY trajectory','Per-step reward','Cumulative reward','Speed (m/s)','Steer error (rad)','Roll (rad)']):
        ax.set_title(title);ax.legend(fontsize=7);ax.grid(alpha=.2)
        if ax is not axs[0,0]:ax.set_xlabel('Control step' if ax in (axs[0,1],axs[1,0]) else 'Time (s)')
    fig.suptitle(case+' | '+best.name+' | nominal DEV, no injected disturbance\nNew reward reconstructed on both traces; endpoints stop at observed failure')
    fig.tight_layout();fig.savefig(best/(case+'_diagnostics.png'),dpi=150);plt.close(fig)
    names=list(r['weights'])+['failure']
    fig,axs=plt.subplots(len(names),2,figsize=(12,15))
    for j,key in enumerate(names):
        for d,rew,label,color,style in [(policy,pr,'Student','#0072B2','-'),(zero,zr,'Zero ECBC','#D55E00','--')]:
            n=int(d['valid'][:,i].sum());steps=np.arange(1,n+1);value=rew[key][:n,i]
            axs[j,0].plot(steps,value,style,color=color,label=label)
            axs[j,1].plot(steps,np.cumsum(value),style,color=color,label=label)
        axs[j,0].set_ylabel(key)
        for ax in axs[j]:ax.grid(alpha=.2);ax.legend(fontsize=7);ax.set_xlabel('Control step')
    axs[0,0].set_title('Signed per-step reward components');axs[0,1].set_title('Cumulative components')
    fig.suptitle(case+' | '+best.name+' | new reward, nominal DEV')
    fig.tight_layout();fig.savefig(best/(case+'_reward_components.png'),dpi=150);plt.close(fig)
(a.evaluation/'reward_diagnostics.json').write_text(json.dumps(dict(config=str(a.training_run/'config.json'),
    candidates=[str(f) for f in folders],selected=str(best),cases=s['cases'],
    limit='Offline reward reconstruction; evaluation did not log an independent online reward. No reward-log parity claim.'),indent=2))
