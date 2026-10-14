"""Offline analysis of the frozen five-checkpoint conditional PPO run; no simulator."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKTREE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(WORKTREE/'learning/src'))
from sttw_control.path_command_reporting import reward_series
from sttw_control.path_command_selection import summarize_conditional

RUN=WORKTREE/'runs/path_unified_alpha_200_20261013/training'
OUT=Path(__file__).parent
UPDATES=[0,50,100,150,200]
ALPHAS={'alpha_0':0.,'alpha_0p25':.25,'alpha_0p5':.5,'alpha_0p75':.75,'alpha_1':1.}
manifest=json.loads((RUN/'manifest.json').read_text());cfg=manifest['config']
status=json.loads((RUN/'status.json').read_text());best=json.loads((RUN/'shared/best_conditional_model.json').read_text())
assert status['state']=='completed' and status['completed_updates']==200
assert status['policy_transitions_total']==13_107_200 and status['validation_control_ticks_total']==600_000
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(best['checkpoint'])==best['checkpoint_sha256'] and sha(best['actor'])==best['actor_sha256']
last=json.loads((RUN/'shared/last_completed.json').read_text());assert sha(last['checkpoint'])==last['checkpoint_sha256']

def bins(d,path):
    lo=cfg['path']['straight_prefix_m'];hi=float(path['turn_end'])
    if hi<=lo:return None
    edges=np.append(np.arange(lo,hi,.5),hi);p=d['path_progress'].astype(float)
    errors=np.stack((d['actual_forward_speed'].astype(float)-d['v_user'],d['path_cross_track']),axis=-1)
    squares=[];counts=[]
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(p>=a)&((p<=b) if i==len(edges)-2 else (p<b))
        counts.append(int(mask.sum()));assert mask.any(),'missing original turn bin'
        squares.append(np.mean(errors[mask]**2,axis=0))
    ev,ey=np.sqrt(np.mean(squares,axis=0))
    turn=(p>=lo)&(p<=hi)
    return dict(speed_rmse_m_s=float(ev),path_rmse_m=float(ey),bin_counts=counts,
                mean_offset_m_s_rad=np.mean(d['offsets'][turn],axis=0).tolist())

result=dict(run=str(RUN),seed=911401,status=status,best_update=best['source_update'],
    best_checkpoint_sha256=best['checkpoint_sha256'],last_checkpoint_sha256=last['checkpoint_sha256'],
    scope='five frozen DEV checkpoints; one training seed; fixed alpha; zero new simulator steps',
    error_windows=dict(full='actual 20 second trajectory',demand='chi>0, original demand',
                       equal_bins='entire original turn, fixed 0.5m bins, equal bin MSE then square root'),
    candidates={},audit=dict(traces=0,lower_ticks=0,max_reward_error=0.))
loaded={}
for u in UPDATES:
    folder=RUN/f'shared/validation/update_{u:04d}'
    source=json.loads((folder/'source.json').read_text())
    assert source['source_update']==u
    for kind in ['checkpoint','actor']:assert sha(source[kind])==source[kind+'_sha256']
    model=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
    assert model['update']==u and model['identity']['source_sha256']==best['source_sha256']
    w=model['policy']['actor.0.weight'].numpy()
    panel={};candidate={'per_alpha':{},'preference':{},'alpha_input_column_l2':float(np.linalg.norm(w[:,336])),
        'other_input_column_median_l2':float(np.median(np.linalg.norm(np.delete(w,336,axis=1),axis=0)))}
    for key,a in ALPHAS.items():
        panel[key]={};candidate['per_alpha'][key]={}
        for trace in sorted((folder/key).glob('*.npz')):
            if trace.stem.endswith('_path'):continue
            name=trace.stem
            with np.load(trace,allow_pickle=False) as z:d=dict(z)
            with np.load(trace.with_name(name+'_path.npz'),allow_pickle=False) as z:path=dict(z)
            with np.load(RUN/'zero_upper'/f'{name}_path.npz',allow_pickle=False) as z:original=dict(z)
            assert set(path)==set(original) and all(np.array_equal(path[k],original[k]) for k in path)
            assert len(d['time'])==4000 and np.all(d['active_tick']) and np.all(d['upper_alpha']==a)
            np.testing.assert_allclose(d['time'],np.arange(4000)*.005,rtol=0,atol=2e-6)
            assert all(np.isfinite(v).all() for v in d.values() if np.issubdtype(v.dtype,np.number))
            assert not any(np.any(d[k]) for k in ('policy_fault','lower_fault','physical_failure','domain_exit'))
            reward,parts=reward_series(d,a,cfg)
            error=float(np.max(np.abs(reward-d['scored_tick_reward'])))
            assert error<2e-6
            result['audit']['max_reward_error']=max(result['audit']['max_reward_error'],error)
            result['audit']['traces']+=1;result['audit']['lower_ticks']+=4000
            ev=d['actual_forward_speed'].astype(float)-d['v_user'];ey=d['path_cross_track']
            demand=d['chi']>0;hold=d['time']>=19
            if not demand.any():demand=np.ones(4000,bool)
            candidate['per_alpha'][key][name]=dict(
                full_speed_rmse_m_s=float(np.sqrt(np.mean(ev**2))),full_path_rmse_m=float(np.sqrt(np.mean(ey**2))),
                demand_speed_rmse_m_s=float(np.sqrt(np.mean(ev[demand]**2))),demand_path_rmse_m=float(np.sqrt(np.mean(ey[demand]**2))),
                peak_roll_rad=float(np.max(d['peak_roll'])),peak_cross_track_m=float(np.max(np.abs(ey))),
                final_hold_independent=bool(np.all(np.abs(ev[hold])<=.1)&np.all(np.abs(ey[hold])<=.1)&np.all(np.abs(d['path_heading_error'][hold])<=.05)),
                hold_speed_max_m_s=float(np.max(np.abs(ev[hold]))),hold_path_max_m=float(np.max(np.abs(ey[hold]))),
                hold_heading_max_rad=float(np.max(np.abs(d['path_heading_error'][hold]))),
                chi_positive_fraction=float(np.mean(d['chi']>0)),strong_fraction=float(np.mean(d['chi']>=.5)),
                turn=bins(d,path),reward_sum=float(np.sum(reward)),component_returns={k:float(np.sum(v)) for k,v in parts.items()})
            panel[key][name]=d;loaded[u,key,name]=(d,path)
    summary=summarize_conditional(panel,cfg);frozen=json.loads((folder/'task_metrics.json').read_text())
    np.testing.assert_allclose(summary['score_tuple'],frozen['score_tuple'],rtol=0,atol=1e-12)
    candidate['summary']={k:v for k,v in summary.items() if k!='per_alpha'}
    for name in ('left90_R2_2.6','right90_R2_2.6'):
        a0=candidate['per_alpha']['alpha_0'][name]['turn'];a1=candidate['per_alpha']['alpha_1'][name]['turn']
        dv=a0['speed_rmse_m_s']-a1['speed_rmse_m_s'];dy=a1['path_rmse_m']-a0['path_rmse_m']
        candidate['preference'][name]=dict(D_v_m_s=dv,D_y_m=dy,direction=(dv>0 and dy>0),legacy_gap_gate=(dv>=.03 and dy>=.02))
    result['candidates'][str(u)]=candidate
rows=[json.loads(line) for line in (RUN/'shared/metrics.jsonl').read_text().splitlines()]
assert [r['update'] for r in rows]==list(range(1,201))
result['training']=dict(accepted_batches=status['accepted_batches'],physical_failures=sum(r['physical_failures'] for r in rows),
    domain_failures=sum(r['tracking_domain_failures'] for r in rows),engineering_faults=sum(r['engineering_faults'] for r in rows),
    hard_rejected_updates=[r['update'] for r in rows if r['optimizer']['accepted_epochs']==0],
    sampling_seconds=sum(r['sample_seconds'] for r in rows),optimization_seconds=sum(r['optimize_seconds'] for r in rows),
    mean_step_reward_windows={f'{lo}-{hi}':float(np.mean([r['mean_step_reward'] for r in rows[lo-1:hi]])) for lo,hi in [(1,25),(26,50),(76,100),(126,150),(176,200)]})
statfiles=sorted((RUN/'shared/statistics').glob('update_*.json'))
result['training']['statistics_windows']={}
for file in statfiles:
    d=json.loads(file.read_text());groups=d['alpha_groups']
    result['training']['statistics_windows'][file.stem]={
        'strong_fraction':d['windows']['strong_demand']['sample_fraction'],
        'demand_fraction':d['windows']['demand_positive']['sample_fraction'],
        'alpha_sample_fraction':{k:v['sample_fraction'] for k,v in groups.items()},
        'max_training_roll_rad':max(v['peak_roll_rad'] for v in groups.values()),
        'primary_effective_cost_mean_by_alpha':{k:v['costs']['primary']['effective_mean'] for k,v in groups.items()}}
(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')

plt.rcParams.update({'font.size':9})
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ri,name in enumerate(('left90_R2_2.6','right90_R2_2.6')):
    for u in UPDATES:
        values=[result['candidates'][str(u)]['per_alpha'][k][name]['turn'] for k in ALPHAS]
        for ci,field in enumerate(('speed_rmse_m_s','path_rmse_m')):
            axes[ri,ci].plot(list(ALPHAS.values()),[v[field] for v in values],'-o',label=f'u{u}')
    for ci in range(2):
        axes[ri,ci].set(title=name,xlabel='Fixed upper alpha',ylabel=('Full-turn speed RMSE (m/s)' if ci==0 else 'Full-turn path RMSE (m)'),xticks=list(ALPHAS.values()));axes[ri,ci].grid(alpha=.25);axes[ri,ci].legend()
fig.suptitle('One shared checkpoint per curve | seed911401 | entire original turn, equal 0.5m bins')
fig.savefig(OUT/'PREFERENCE_BY_ALPHA_AND_UPDATE.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
colors=plt.cm.viridis(np.linspace(0,1,5))
for ri,name in enumerate(('left90_R2_2.6','right90_R2_2.6')):
    for ci,u in enumerate((50,200)):
        ax=axes[ri,ci];_,p=loaded[u,'alpha_0',name];ref=(p['s']>=2.5)&(p['s']<=9.)
        ax.plot(*p['xy'][ref].T,'k--',label='Original fixed path')
        with np.load(RUN/'zero_upper'/f'{name}.npz') as z:b=dict(z)
        m=(b['path_progress']>=2.5)&(b['path_progress']<=9);ax.plot(*b['actual_xy'][m].T,color='.5',label='B0 local300+ECBC/ESO')
        for color,(key,a) in zip(colors,ALPHAS.items()):
            d,_=loaded[u,key,name];m=(d['path_progress']>=2.5)&(d['path_progress']<=9)
            ax.plot(*d['actual_xy'][m].T,color=color,label=f'alpha={a:g}')
        ax.set(title=f'{name} | u{u}',xlabel='World X (m)',ylabel='World Y (m)');ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.suptitle('Turn XY detail | one shared model per column | all five alpha | same original paths')
fig.savefig(OUT/'TURN_XY_U50_U200.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
rewards=np.array([r['mean_step_reward'] for r in rows]);axes[0,0].plot(np.arange(1,201),rewards,alpha=.35,label='Actual TRAIN batch');axes[0,0].plot(np.arange(25,201),np.convolve(rewards,np.ones(25)/25,'valid'),label='25-update mean');axes[0,0].set(xlabel='Update (sample scored on preceding model)',ylabel='Mean policy-interval reward');axes[0,0].legend()
for name in ('left90_R2_2.6','right90_R2_2.6'):
    axes[0,1].plot(UPDATES,[result['candidates'][str(u)]['preference'][name]['D_v_m_s'] for u in UPDATES],'-o',label=name)
    axes[1,0].plot(UPDATES,[result['candidates'][str(u)]['preference'][name]['D_y_m'] for u in UPDATES],'-o',label=name)
axes[0,1].axhline(.03,color='k',ls='--',label='Legacy difference .03m/s');axes[0,1].set(xlabel='Frozen DEV update',ylabel='Ev(alpha0)-Ev(alpha1) (m/s)');axes[0,1].legend()
axes[1,0].axhline(.02,color='k',ls='--',label='Legacy difference .02m');axes[1,0].set(xlabel='Frozen DEV update',ylabel='Ey(alpha1)-Ey(alpha0) (m)');axes[1,0].legend()
for field,label in [('goal_hold_failures','Goal/hold failures /30'),('endpoint_primary_failures','Endpoint primary failures /12')]:
    axes[1,1].plot(UPDATES,[result['candidates'][str(u)]['summary'][field] for u in UPDATES],'-o',label=label)
axes[1,1].set(xlabel='Frozen DEV update',ylabel='Failed cases');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.25);ax.axvline(50,color='.5',ls=':',label='Selected u50')
fig.suptitle('TRAIN and frozen DEV | seed911401 | original thresholds are diagnostics')
fig.savefig(OUT/'TRAINING_AND_DEV.png',dpi=170);plt.close(fig)
print(json.dumps({'audit':result['audit'],'training':result['training'],'preference':{u:c['preference'] for u,c in result['candidates'].items()}},indent=2))

# Last-candidate full six-case XY and actual-step reward coverage; existing best
# pointer stays at u50. Each alpha uses its own rescored B0 reward.
case_names=list(result['candidates']['200']['per_alpha']['alpha_0'])
for kind in ('XY','STEP_REWARD','CUMULATIVE_REWARD'):
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,name in zip(axes.flat,case_names):
        with np.load(RUN/'zero_upper'/f'{name}.npz') as z:b=dict(z)
        _,p=loaded[200,'alpha_0',name]
        if kind=='XY':
            progress=max(float(b['path_progress'].max()),*[float(loaded[200,k,name][0]['path_progress'].max()) for k in ALPHAS])
            ref=p['s']<=progress+2
            ax.plot(*p['xy'][ref].T,'k--',label='Original path');ax.plot(*b['actual_xy'].T,color='.5',label='B0 local300+ECBC/ESO')
        for color,(key,a) in zip(colors,ALPHAS.items()):
            d,_=loaded[200,key,name]
            if kind=='XY':ax.plot(*d['actual_xy'].T,color=color,label=f'alpha={a:g}')
            else:
                y=d['scored_tick_reward'];by,_=reward_series(b,a,cfg)
                if kind=='CUMULATIVE_REWARD':y=np.cumsum(y);by=np.cumsum(by)
                ax.plot(np.arange(1,len(y)+1),y,color=color,label=f'alpha={a:g}')
                ax.plot(np.arange(1,len(by)+1),by,color=color,ls='--',alpha=.65,label=f'B0 scored alpha={a:g}')
        ax.set_title(name);ax.grid(alpha=.2)
        if kind=='XY':ax.set(xlabel='World X (m)',ylabel='World Y (m)');ax.axis('equal')
        else:ax.set(xlabel='Actual lower-control step (5ms)',ylabel='Signed reward' if kind=='STEP_REWARD' else 'Cumulative signed reward')
        ax.legend(fontsize=6,ncol=1 if kind=='XY' else 2)
    fig.suptitle('u200 SHA08df91a0f9d6 | seed911401 | same shared model at all alpha | no external disturbance')
    fig.savefig(OUT/f'U200_{kind}.png',dpi=160);plt.close(fig)
