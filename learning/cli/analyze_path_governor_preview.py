"""Read one numerical preview batch; never execute or extend a controller."""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def main(root,source):
    root=root.resolve();source=source.resolve()
    receipt=json.loads((root/'receipt.json').read_text());n=json.loads((root/'numerical.json').read_text())
    if not receipt['preview_results_exist'] or n['governor_control_episodes']!=0:raise ValueError('one completed engineering preview required')
    if json.loads((root/'status.json').read_text())['phase']!='completed':raise ValueError('numerical check not complete')
    with np.load(root/'preview_traces.npz') as z:raw={k:z[k] for k in z.files}
    data={k:v.reshape((5,-1)+v.shape[3:]) for k,v in raw.items()}
    if int(data['active_tick'].sum())!=n['actual_lower_ticks']:raise ValueError('actual preview count differs')
    forecasts=json.loads((root/'forecasts.json').read_text())
    for i,row in enumerate(forecasts):
        a=data['active_tick'][i].astype(bool)
        if int(a.sum())!=row['observed_lower_ticks']:raise ValueError('lane count differs')
        if row['original_user_cost'] is not None:
            np.testing.assert_allclose(-data['scored_tick_reward'][i,a].sum(),row['original_user_cost'],atol=3e-5,rtol=3e-5)
    source_rows=json.loads((source/'analysis_partial_block_fix_v1/results.json').read_text())
    first=next(r for r in source_rows if r['prefix_index']==0 and r['role']=='full_current')
    if digest(first['raw_file'])!=first['raw_sha256']:raise ValueError('changed inherited prefix')
    with np.load(first['raw_file']) as z:prefix=z['actual_xy'][:first['fork_task_tick']]
    plan=json.loads((source/'plan.json').read_text());context=plan['contexts'][first['context']]
    with np.load(context['path']) as z:path={k:z[k] for k in z.files}
    goal=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in range(2)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors=['#333333','#E69F00','#56B4E9','#009E73','#D55E00'];styles=['-','--','-.',':','-']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].plot(*path['xy'][path['s']<=path['goal']].T,c='black',ls=':',label='Original task path')
    axes[0].add_patch(Circle(goal,1,fill=False,ec='black',lw=.8));axes[0].scatter(*goal,c='black',marker='*',s=20)
    axes[0].plot(*prefix.T,c='#999999',lw=1,label='Inherited actual prefix')
    all_xy=[]
    for i,(r,c,style) in enumerate(zip(forecasts,colors,styles)):
        xy=np.vstack([prefix[-1],data['actual_xy'][i,data['active_tick'][i].astype(bool)]])
        all_xy.append(xy)
        for ax in axes:
            ax.plot(*xy.T,c=c,ls=style,label=f'fraction {r["fraction"]:g}: {r["status"]}')
            ax.scatter(*xy[-1],c=c,marker='x' if r['status']=='contact' else 'o',s=20)
            ax.scatter(*xy[0],c=c,marker='s',s=15)
    for ax in axes:ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='World X (m)',ylabel='World Y (m)');ax.grid(alpha=.15)
    axes[0].set_title('Original geometry / inherited prefix');axes[1].set_title('Forecast suffix only: one origin, five hypotheses');axes[1].legend(fontsize=7)
    fig.suptitle('One-second candidate model forecasts; NOT executed online governor trajectories\nOriginal user alpha retained; assumed zero future force, no future event input')
    fig.savefig(root/'preview_xy.png',dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for i,(r,c,style) in enumerate(zip(forecasts,colors,styles)):
        a=data['active_tick'][i].astype(bool);t=(np.arange(len(a))+1)*.005
        label=f'fraction {r["fraction"]:g}'
        axes[0,0].plot(t[a],data['actual_forward_speed'][i,a],c=c,ls=style,label=label+' body')
        axes[0,0].plot(t[a],data['wheel_speed_proxy'][i,a],c=c,ls='--',alpha=.5)
        axes[0,0].plot(t[a],data['reference'][i,a,0],c=c,ls=':',alpha=.7)
        axes[0,1].plot(t[a],data['actual_delta'][i,a],c=c,ls=style,label=label)
        axes[0,1].plot(t[a],data['reference'][i,a,1],c=c,ls=':',alpha=.7)
        axes[1,0].plot(t[a],data['phi'][i,a],c=c,ls=style,label=label)
        axes[1,1].plot(t[a],data['scored_tick_reward'][i,a],c=c,ls=style,label=label)
    for ax,label in zip(axes.flat,['Speed (m/s)','Steering (rad)','Roll endpoint (rad)','Scored reward / lower tick']):ax.set(xlabel='Forecast time (s)',ylabel=label);ax.grid(alpha=.15)
    axes[0,0].set_title('Body / wheel proxy (faint dashed) / reference (dotted)');axes[0,1].set_title('Actual joint / reference (dotted)');axes[1,0].legend(fontsize=7)
    fig.suptitle('One numerical engineering batch: local model forecasts, no anti-disturbance success claim')
    fig.savefig(root/'preview_diagnostics.png',dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    atomic_json(root/'analysis_receipt.json',dict(actual_lower_ticks=n['actual_lower_ticks'],allocated_lower_slots=n['allocated_lower_slots'],
        source_prefix_raw_sha256=first['raw_sha256'],trace_cost_reconstructed=True,analysis_physics_steps=0,
        real_time_control_period_s=.02,execution_s=n['execution_s'],compile_s=n['compile_s'],
        warning='One measured batch, not warm repeated latency benchmark; no closed-loop governor evaluation',
        full_reward_components_saved=False,visual_review='pending'))
    print(json.dumps(n))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path);a=p.parse_args();main(a.root,a.source)
