"""Read-only all-branch diagnostics for the 80-lower-tick collector probe."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_best import atomic_json, digest
from sttw_control.path_command_reward import configured_cost_keys


def render(root):
    if json.loads((root/'status.json').read_text())['phase'] != 'completed':
        raise ValueError('complete engineering probe required')
    plan=json.loads((root/'plan.json').read_text()); receipt=json.loads((root/'receipt.json').read_text())
    cfg=json.loads((Path(plan['training_run'])/'effective_config.json').read_text())
    labels=plan['branches']; data={}
    for label in labels:
        with np.load(root/(label+'.npz')) as z: data[label]={k:z[k] for k in z.files}
    with np.load(root/'reference_path.npz') as z: reference=z['xy']
    fig, axes=plt.subplots(1,len(labels),figsize=(19,4.2),squeeze=False)
    for ax,label in zip(axes[0],labels):
        d=data[label]; active=d['active_tick']; xy=d['actual_xy'][active]
        ax.plot(reference[:,0],reference[:,1],'--',color='.55',lw=1,label='Original path')
        ax.plot(xy[:,0],xy[:,1],color='tab:blue',marker='.',ms=3,label='Actual XY')
        ax.scatter(*xy[-1],marker='x' if d['contact_failure'][active].any() else 'o',color='tab:red',s=25,label='Actual probe end')
        center=(xy.max(axis=0)+xy.min(axis=0))/2; span=max(float(np.ptp(xy,axis=0).max()),.02)*.7
        ax.set(xlim=(center[0]-span,center[0]+span),ylim=(center[1]-span,center[1]+span),
               title=label,xlabel='World X (m)',ylabel='World Y (m)')
        ax.set_aspect('equal',adjustable='box'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Collector engineering: same complete task0 start; 0.08 s per branch, no disturbance')
    fig.tight_layout(); fig.savefig(root/'all_xy.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(6,len(labels),figsize=(23,19),squeeze=False)
    costs=configured_cost_keys(cfg); scale=cfg['path_reward']['scale']*cfg['timing']['lower_dt_s']
    for ci,label in enumerate(labels):
        d=data[label]; mask=d['active_tick']; t=d['time'][mask]
        def line(row,key,name,style='-'):
            axes[row,ci].plot(t,d[key][mask],style,label=name,lw=1)
        line(0,'actual_forward_speed','True forward');line(0,'wheel_speed_proxy','Wheel proxy','--')
        axes[0,ci].plot(t,d['reference'][mask,0],label='Given reference')
        axes[0,ci].axhline(1.8,color='.3',ls=':',label='Original request')
        line(1,'actual_delta','Actual steering')
        axes[1,ci].plot(t,d['reference'][mask,1],label='Given reference')
        line(2,'phi','Signed roll');line(2,'peak_roll','Peak absolute roll','--')
        line(3,'phi_dot','Roll rate')
        valid=mask & d['task_reward_valid']; tv=d['time'][valid]
        axes[4,ci].plot(tv,d['path_cross_track'][valid],label='Path error (m)')
        axes[4,ci].plot(tv,d['path_heading_error'][valid],label='Heading error (rad)',ls='--')
        upper_valid=valid.reshape(-1,4).any(axis=1)
        upper_time=d['time'].reshape(-1,4)[:,0][upper_valid]
        aggregate=lambda x:np.where(valid,x,0.).reshape(-1,4).sum(axis=1)[upper_valid]
        axes[5,ci].plot(upper_time,aggregate(d['scored_tick_reward']),color='black',lw=1.8,label='Scored reward')
        axes[5,ci].plot(upper_time,aggregate(d['tick_reward']),color='.5',ls='--',label='Running reward')
        for ki,key in enumerate(costs):
            axes[5,ci].plot(upper_time,aggregate(-scale*d['effective_costs'][:,ki]),label='-'+key,lw=.8)
        axes[0,ci].set_title(label)
        for ri,unit in enumerate(['Speed (m/s)','Steering (rad)','Roll (rad)',
                                  'Roll rate (rad/s)','Task errors','Per-20ms-policy-step reward']):
            ax=axes[ri,ci];ax.set_ylabel(unit);ax.grid(alpha=.2)
            if ri==5:ax.set_xlabel('Actual task time (s)')
            if ci==0:ax.legend(fontsize=6,ncol=2 if ri==5 else 1)
    fig.suptitle('All branches: 5 ms physical logs; rewards summed over each actual 20 ms decision; no post-end continuation',y=.999)
    fig.tight_layout();fig.savefig(root/'all_diagnostics.png',dpi=130);plt.close(fig)
    sources=[root/(name+'.npz') for name in labels]+[root/'receipt.json',root/'reference_path.npz']
    atomic_json(root/'plot_receipt.json',dict(covered_branches=labels,physics_steps=0,optimizer_steps=0,
        source_sha256={str(p):digest(p) for p in sources}, reward_interval_s=.02,
        physical_log_interval_s=.005,figures=['all_xy.png','all_diagnostics.png']))
    (root/'REPORT.md').write_text('''# Actual endpoint collector engineering probe

[All five original/actual XY trajectories](all_xy.png)

[All five speed, steering, roll, errors and reward-component traces](all_diagnostics.png)

Purpose: verify real collection plumbing before the fixed terminal policy comparison. Same complete bank3 task0 state, original H75/local300, alpha0, left route requested1.8 m/s, no external force. New Gaussian, original Gaussian, deterministic mean, exact zero and parent with a1.5 m/s reference floor each executed only four20 ms policy intervals.

This is at most80 actual5 ms lower-control transitions total, not80 MuJoCo integration substeps, and zero optimizer updates. Raw actions/logs and complete final states were saved before assertions. The Gaussian differences and actual per-branch costs are in receipt.json; deterministic mean/zero use no Gaussian density or random draws. The floor restricts only references through the original clipping/filter back-calculation, retaining the requested task and rewards; this very short start window need not activate its limit.

These trajectories check an initial collector interface. They do not establish10 s survival, task recovery, force tolerance, preference preservation or a training benefit. The64-update training and488-case terminal evaluation remain separate.
''')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    render(p.parse_args().run.resolve())
