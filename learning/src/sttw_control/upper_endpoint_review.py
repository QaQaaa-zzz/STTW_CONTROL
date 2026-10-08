"""Matched final endpoint comparison: independent Actors, one frozen lower."""
from pathlib import Path
import pickle,json,time
import numpy as np
from .direct_command_training import write,flatten_logs

def rescore_baseline(trace,alpha,spec,previous):
    import jax,jax.numpy as jp
    from .direct_command_reward import interval_cost
    d={k:np.array(v,copy=True) for k,v in trace.items()};n=len(d['time'])
    before=np.concatenate([np.asarray(previous)[None],d['final_command'][:-1]])
    c=jax.device_get(interval_cost(alpha=jp.full((n,),float(alpha)),chi=d['chi'],g=d['g'],raw=d['limited_command'],actual_speed=d['actual_forward_speed'],actual_steer=d['actual_delta'],heading_error=d['e_psi_unwrapped'],roll=d['phi'],roll_rate=d['phi_dot'],executed_offsets=d['offsets'],final_command=d['final_command'],previous_final_command=before,spec=spec))
    d['alpha']=np.full(n,float(alpha));d['raw_cost']=c['raw_cost'];d['effective_cost']=c['effective_cost'];d['cap_fraction']=c['cap_fraction'];d['tick_reward']=c['reward'];d['scored_tick_reward']=c['reward'].copy()
    for k in c['raw_components']:d['raw_cost_'+k]=c['raw_components'][k];d['effective_cost_'+k]=c['effective_components'][k];d['scored_cost_'+k]=c['effective_components'][k].copy()
    if np.any(d['physical_failure']):
        idx=int(np.flatnonzero(d['physical_failure'])[0]);first=idx//4*4
        d['scored_tick_reward'][first:idx+1]=0.;d['scored_tick_reward'][idx]=trace['scored_tick_reward'][idx]
        for k in c['raw_components']:d['scored_cost_'+k][first:idx+1]=0.
    return d

def select_checkpoint(run):
    """Use explicit best provenance, otherwise the last completed checkpoint."""
    run=Path(run);best=run/'pilot/best_model.json'
    if best.exists():
        info=json.loads(best.read_text());update=int(info['update']);source='training_reward_best'
    else:
        info=json.loads((run/'pilot/last_completed.json').read_text());update=int(info['update']);source='last_completed'
    checkpoint=run/f'pilot/checkpoints/update_{update:04d}.pt'
    actor=run/f'pilot/checkpoints/actor_{update:04d}.pkl'
    if not checkpoint.exists() or not actor.exists():raise FileNotFoundError(f'incomplete selected checkpoint {update}: {run}')
    return dict(update=update,checkpoint=str(checkpoint),actor=str(actor),selection=source,provenance=info)


def training_diagnostics(runs,out,selections):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,2,figsize=(13,10));summary={}
    for alpha,run in enumerate(runs):
        rows=[json.loads(x) for x in (run/'pilot/metrics.jsonl').read_text().splitlines()]
        groups=[r['groups'][f'alpha{alpha}/family-1/all'] for r in rows]
        count=sum(g['count'] for g in groups)
        stats={k:sum(g[k]*g['count'] for g in groups)/max(1,count) for k in ['cost_cap','roll_violations','motor_clip','final_clip','reference_clip','reference_rate_clip']}
        failed=sum(g['failed_episodes'] for g in groups);ends=sum(g['episode_ends'] for g in groups)
        stats.update(failed_episodes=failed,episode_ends=ends,failure_rate=failed/ends if ends else None,updates=len(rows),selection=selections[alpha],sample_seconds=sum(r['sample_seconds'] for r in rows),optimize_seconds=sum(r['optimize_seconds'] for r in rows),hard_kl_stops=sum(bool(r['optimizer']['hard_kl_stop']) for r in rows),nonfinite_stops=sum(bool(r['optimizer']['nonfinite_stop']) for r in rows))
        for k in ['actor_grad_norm','critic_grad_norm','mean_kl']:
            values=[r['optimizer'][k] for r in rows];stats[k]={'min':min(values),'max':max(values),'mean':float(np.mean(values))}
        summary[f'alpha{alpha}']=stats
        values=[[r['mean_step_reward'] for r in rows],[r['optimizer']['mean_kl'] for r in rows],[r['optimizer']['actor_grad_norm'] for r in rows],[r['optimizer']['critic_grad_norm'] for r in rows],[g['cost_cap'] for g in groups],[g['roll_violations'] for g in groups]]
        for ax,y in zip(axes.flat,values):ax.plot([r['update'] for r in rows],y,label=f'alpha{alpha}')
    for ax,title in zip(axes.flat,['Training mean step reward (different alpha weights)','KL per independent alpha model','Actor gradient norm','Critic gradient norm','Cost-cap fraction','Roll work-range violation fraction']):
        ax.set_title(title);ax.set_xlabel('PPO update');ax.legend();ax.grid(alpha=.3)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'training_diagnostics.{ext}',dpi=130)
    plt.close(fig);write(out/'training_summary.json',summary)
    return summary


def review_pair(root,alias):
    import jax,jax.numpy as jp
    from .direct_command_env import DirectCommandEnv
    from .direct_command_policy import DirectCommandActor
    from .direct_command_scenarios import review_rows
    from .fixed_command_panel import load_protocol,schedules,report
    protocol=load_protocol()
    from .direct_command_audit import reconstruct_trace
    root=Path(root);short=alias.split('_')[1];runs=[root/f'{short}_upper{a}' for a in [0,1]];out=root/f'{short}_comparison';start=time.monotonic()
    if (out/'manifest.json').exists():raise FileExistsError(out)
    specs=[json.loads((p/'frozen_config.json').read_text()) for p in runs]
    assert [s['upper_alpha'] for s in specs]==[0,1]
    assert all(s['lower_controller']['alias']==alias and s['lower_controller']['alpha']==1. for s in specs)
    updates=specs[0]['ppo']['default_updates']
    assert specs[1]['ppo']['default_updates']==updates
    for p in runs:
        status=json.loads((p/'status.json').read_text());assert status['completed_updates']==updates and status['state']=='complete'
    selections=[select_checkpoint(p) for p in runs]
    training_diagnostics(runs,out/'review',selections)
    banks=[]
    for p in runs:
        with (p/'prepared_bank.pkl').open('rb') as f:banks.append(pickle.load(f))
    for a,b in zip(jax.tree.leaves(banks[0]),jax.tree.leaves(banks[1])):np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
    sample=jax.tree.map(lambda x:jp.asarray(x[3]),banks[0]);params=[]
    for selected in selections:
        with Path(selected['actor']).open('rb') as f:params.append(pickle.load(f))
    stacked=jax.tree.map(lambda a,b:jp.stack([a,a,b]),*params);spec=specs[0];env=DirectCommandEnv(spec);actor=DirectCommandActor();alphas=jp.array([0.,0.,1.]);bypass=jp.array([True,False,False])
    write(out/'manifest.json',dict(lower_alias=alias,lower_alpha=1.,upper_alphas=[0,1],independent_checkpoints=[x['checkpoint'] for x in selections],checkpoint_selection=selections,bank_parity=True,new_episodes=18,control_ticks_upper=36000,budget_s=1200,protocol=protocol))
    cases=schedules(protocol);main=cases[0][1];sl=np.asarray(protocol['slew']);write(out/'review/frozen_schedules.json',protocol)
    reset=jax.jit(lambda rows:jax.vmap(lambda a:env.reset(sample,jp.int32(77001),jp.int32(0),a,rows,jp.asarray(sl,jp.float32)))(alphas))
    states=reset(jp.asarray(main,jp.float32));jax.block_until_ready(states);env.set_log_template(jax.tree.map(lambda x:x[0],states))
    def chunk(s):
        def step(s,_):
            obs,_,fault,_=jax.vmap(env.observation)(s);mu=jax.vmap(lambda p,x:actor.apply(p,x))(stacked,obs);z=jp.where((bypass|s.physical.failed)[:,None],jp.zeros_like(mu),mu)
            end,_,_,logs,_=jax.vmap(env.policy_step)(s,z,bypass);return end,logs
        return jax.lax.scan(step,s,None,length=50)
    write(out/'status.json',dict(state='running',stage='compile'));execute=jax.jit(chunk).lower(states).compile();audits={}
    for case,rows in cases:
        states=reset(jp.asarray(rows,jp.float32));chunks=[];case_start=time.monotonic()
        case_dir=out/'review'/case;case_dir.mkdir(parents=True,exist_ok=True)
        def persist():
            flat=flatten_logs(jax.tree.map(lambda *x:np.concatenate(x),*chunks))
            for i,name in enumerate(['B0','pi_alpha0','pi_alpha1']):
                d={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()};mask=d['active_tick'];d={k:v[mask] for k,v in d.items()};d['checkpoint_update']=np.asarray(-1 if i==0 else selections[i-1]['update']);np.savez_compressed(case_dir/f'{name}.npz',**d)
        for sec in range(protocol['duration_s']):
            if time.monotonic()-start>1100 or time.monotonic()-case_start>480:raise TimeoutError('declared comparison budget')
            states,logs=execute(states);jax.block_until_ready(states);chunks.append(jax.device_get(logs));persist()
            write(out/'status.json',dict(state='running',case=case,seconds=sec+1,elapsed_s=time.monotonic()-start))
            if np.any(np.asarray(states.fault)):raise RuntimeError('evaluation policy fault')
        controller=json.loads((runs[0]/'manifest.json').read_text())['controller'];audits[case]={}
        for name in ['B0','pi_alpha0','pi_alpha1']:
            d=dict(np.load(case_dir/f'{name}.npz'));audits[case][name]=reconstruct_trace(d,spec,np.asarray(sample.actuator.previous),controller)
        baseline=dict(np.load(case_dir/'B0.npz'));b1=rescore_baseline(baseline,1,spec,np.asarray(sample.actuator.previous));np.savez_compressed(case_dir/'B0_alpha1.npz',**b1);audits[case]['B0_alpha1']=reconstruct_trace(b1,spec,np.asarray(sample.actuator.previous),controller)
        assert all(v['passed'] for v in audits[case].values())
        report(out,spec,protocol)
        # Additional correctly paired reward curves; B0 in inherited overview is alpha0-scored.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,2,figsize=(13,8))
        for a,base in [(0,baseline),(1,b1)]:
            learned=dict(np.load(case_dir/f'pi_alpha{a}.npz'))
            for trace,label,ls in [(base,'same-alpha baseline','--'),(learned,f'upper_alpha{a}','-')]:
                reward=trace['scored_tick_reward'];steps=np.arange(1,len(reward)+1);ax[a,0].plot(steps,reward,ls,label=label);ax[a,1].plot(steps,np.cumsum(reward),ls,label=label)
            for j in [0,1]:ax[a,j].legend();ax[a,j].grid(alpha=.3);ax[a,j].set_title(f'alpha{a} '+('per-step reward' if j==0 else 'cumulative reward'))
        fig.suptitle(alias+' '+case+' paired reward; raw command evaluation');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/'review'/f'{case}_paired_rewards.{ext}',dpi=130)
        plt.close(fig)
    write(out/'review/audit.json',audits);write(out/'status.json',dict(state='complete',episodes=18,elapsed_s=time.monotonic()-start))
    (out/'review/INDEX.md').write_text(
        '# Frozen lower / independent upper endpoint comparison\n\n'
        f"Lower: {alias}, fixed lower_alpha=1. Independent upper selected updates: {[x['update'] for x in selections]}.\n\n"
        '[Training diagnostics](training_diagnostics.png) · [Training metrics](training_summary.json)\n\n'
        '[Physical metrics and plots](REPORT.md) · [Metrics](metrics.json) · [Reward audit](audit.json)\n\n'
        'Six historical fixed command cases; case plots linked in REPORT.md.\n\n'
        'B0 is this frozen residual lower with zero upper correction. Its physical trajectory is shared; '
        'rewards are rescored separately for each upper alpha. These are two independent upper policies.\n')
