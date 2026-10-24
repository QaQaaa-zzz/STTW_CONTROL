"""All-case20ms signed reward/components/errors and true speed from raw logs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_best import digest,atomic_json
from sttw_control.path_command_reward import configured_cost_keys

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
assert json.loads((root/'status.json').read_text())['phase']=='completed'
plan=json.loads((root/'plan.json').read_text());results=json.loads((root/'results.json').read_text())
cfg=json.loads(Path(plan['source_declaration']).read_text())['config'];keys=configured_cost_keys(cfg)
metrics=[];hashes={};plots=[]
for index in range(4):
    fig,axes=plt.subplots(4,3,figsize=(16,12),layout='constrained')
    case=plan['cases'][index]
    for column,model in enumerate(['parent','task_only','physical_credit']):
        row=results['models'][model][index];raw=root/model/f'case_{index:03d}.npz'
        assert digest(raw)==row['raw_sha256'];hashes[str(raw)]=digest(raw)
        with np.load(raw) as z:
            active=z['active_tick'];task=z['task_reward_valid']&active
            task_blocks=task.reshape(-1,4).any(axis=1)
            times=z['time'].reshape(-1,4)[:,0]
            reward=np.where(task,z['scored_tick_reward'],0.).reshape(-1,4).sum(axis=1)
            components=(-cfg['path_reward']['scale']*.005*np.where(task[:,None],z['effective_costs'],0.)).reshape(-1,4,len(keys)).sum(axis=1)
            axes[0,column].plot(times[task_blocks],reward[task_blocks],label='actual20ms scored reward')
            for k,key in enumerate(keys):axes[1,column].plot(times[task_blocks],components[task_blocks,k],label=key)
            for field,label in [('path_cross_track','cross-track(m)'),('path_heading_error','heading(rad)'),('phi','roll(rad)')]:
                axes[2,column].plot(z['time'][task],z[field][task],label=label)
            for field,label in [('actual_forward_speed','true body speed'),('wheel_speed_proxy','wheel speed proxy')]:
                axes[3,column].plot(z['time'][active],z[field][active],label=label)
            axes[3,column].plot(z['time'][active],z['reference'][active,0],label='governed speed reference')
            axes[3,column].axhline(case['speed'],color='black',ls='--',label='original requested speed')
            axes[0,column].set_title(f"{model} | {row['outcome']} | hold={row['hold_ticks']*.005:.2f}s")
            for ax in axes[:,column]:
                if case['force_N']:ax.axvspan(case['pulse_onset_task_tick']*.005,case['pulse_onset_task_tick']*.005+.1,color='grey',alpha=.2)
                hold=np.flatnonzero(z['fallback_active']&active)
                if len(hold):ax.axvline(z['time'][hold[0]],c='purple',ls=':',lw=1)
                if row['event_contact']:ax.axvline(row['event_time_s'],c='red',ls='--',lw=1)
                ax.set_xlabel('task time(s)');ax.legend(fontsize=6,ncol=2)
            terminal=float(np.where(task,z['scored_tick_reward'],0.).sum(dtype=np.float64)-components.sum(dtype=np.float64))
            metrics.append(dict(model=model,case=index,task_scored_return=row['task_scored_return'],
                running_component_return=float(components.sum(dtype=np.float64)),terminal_penalty_return=terminal,
                max_abs_cross_track_task_active=float(np.max(np.abs(z['path_cross_track'][task]))),
                mean_true_speed_physical_window=float(z['actual_forward_speed'][active].mean()),
                mean_true_speed_task_active=float(z['actual_forward_speed'][task].mean()),
                hold_s=row['hold_ticks']*.005,contact_time_s=row['event_time_s']))
    for ax,label in zip(axes[:,0],['20ms scored reward','signed20ms components','task-active errors','speed(m/s)']):ax.set_ylabel(label)
    fig.suptitle(f"case{index}: {case['route']}, request{case['speed']}m/s, F={case['force_N']}N, alpha0\nPurple dotted: path-domain handover; red dashed: actual contact. No task reward/error is invented during hold.")
    target=root/f'paired_case_{index:03d}_rewards.png'
    if target.exists():raise FileExistsError('preserve prior plot')
    fig.savefig(target,dpi=150);plt.close(fig);plots.append(str(target))
atomic_json(root/'continuous_metrics.json',dict(cases=metrics,raw_sha256=hashes,plots=plots,new_physics_ticks=0,
    scope='Task errors only while task-valid; physical mean speed includes actual hold tail. Last partial interval sums actual ticks only.'))
print(json.dumps(metrics))
