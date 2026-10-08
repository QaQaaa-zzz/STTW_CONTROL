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

def review_pair(root,alias):
    import jax,jax.numpy as jp
    from .direct_command_env import DirectCommandEnv
    from .direct_command_policy import DirectCommandActor
    from .direct_command_scenarios import review_rows
    from .direct_command_reporting import generate_report
    from .direct_command_audit import reconstruct_trace
    root=Path(root);short=alias.split('_')[1];runs=[root/f'{short}_upper{a}' for a in [0,1]];out=root/f'{short}_comparison';start=time.monotonic()
    if (out/'manifest.json').exists():raise FileExistsError(out)
    specs=[json.loads((p/'frozen_config.json').read_text()) for p in runs]
    assert [s['upper_alpha'] for s in specs]==[0,1]
    assert all(s['lower_controller']['alias']==alias and s['lower_controller']['alpha']==1. for s in specs)
    for p in runs:
        status=json.loads((p/'status.json').read_text());assert status['completed_updates']==250 and status['state']=='complete'
    banks=[]
    for p in runs:
        with (p/'prepared_bank.pkl').open('rb') as f:banks.append(pickle.load(f))
    for a,b in zip(jax.tree.leaves(banks[0]),jax.tree.leaves(banks[1])):np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
    sample=jax.tree.map(lambda x:jp.asarray(x[3]),banks[0]);params=[]
    for p in runs:
        with (p/'pilot/checkpoints/actor_0250.pkl').open('rb') as f:params.append(pickle.load(f))
    stacked=jax.tree.map(lambda a,b:jp.stack([a,a,b]),*params);spec=specs[0];env=DirectCommandEnv(spec);actor=DirectCommandActor();alphas=jp.array([0.,0.,1.]);bypass=jp.array([True,False,False])
    write(out/'manifest.json',dict(lower_alias=alias,lower_alpha=1.,upper_alphas=[0,1],independent_checkpoints=[str(p/'pilot/checkpoints/update_0250.pt') for p in runs],bank_parity=True,new_episodes=6,control_ticks_upper=19200,budget_s=1200,protocol='original main/random16s,3 methods, raw command evaluation'))
    main,random,sl=review_rows(spec);write(out/'review/frozen_schedules.json',dict(main=main.tolist(),random=random.tolist(),slew=sl.tolist()))
    reset=jax.jit(lambda rows:jax.vmap(lambda a:env.reset(sample,jp.int32(77001),jp.int32(0),a,rows,jp.asarray(sl,jp.float32)))(alphas))
    states=reset(jp.asarray(main,jp.float32));jax.block_until_ready(states);env.set_log_template(jax.tree.map(lambda x:x[0],states))
    def chunk(s):
        def step(s,_):
            obs,_,fault,_=jax.vmap(env.observation)(s);mu=jax.vmap(lambda p,x:actor.apply(p,x))(stacked,obs);z=jp.where((bypass|s.physical.failed)[:,None],jp.zeros_like(mu),mu)
            end,_,_,logs,_=jax.vmap(env.policy_step)(s,z,bypass);return end,logs
        return jax.lax.scan(step,s,None,length=50)
    write(out/'status.json',dict(state='running',stage='compile'));execute=jax.jit(chunk).lower(states).compile();audits={}
    for case,rows in [('main',main),('random',random)]:
        states=reset(jp.asarray(rows,jp.float32));chunks=[];case_start=time.monotonic()
        case_dir=out/'review'/case;case_dir.mkdir(parents=True,exist_ok=True)
        def persist():
            flat=flatten_logs(jax.tree.map(lambda *x:np.concatenate(x),*chunks))
            for i,name in enumerate(['B0','pi_alpha0','pi_alpha1']):
                d={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()};mask=d['active_tick'];d={k:v[mask] for k,v in d.items()};d['checkpoint_update']=np.asarray(250);np.savez_compressed(case_dir/f'{name}.npz',**d)
        for sec in range(16):
            if time.monotonic()-start>1100 or time.monotonic()-case_start>480:raise TimeoutError('declared comparison budget')
            states,logs=execute(states);jax.block_until_ready(states);chunks.append(jax.device_get(logs));persist()
            write(out/'status.json',dict(state='running',case=case,seconds=sec+1,elapsed_s=time.monotonic()-start))
            if np.any(np.asarray(states.fault)):raise RuntimeError('evaluation policy fault')
        controller=json.loads((runs[0]/'manifest.json').read_text())['controller'];audits[case]={}
        for name in ['B0','pi_alpha0','pi_alpha1']:
            d=dict(np.load(case_dir/f'{name}.npz'));audits[case][name]=reconstruct_trace(d,spec,np.asarray(sample.actuator.previous),controller)
        baseline=dict(np.load(case_dir/'B0.npz'));b1=rescore_baseline(baseline,1,spec,np.asarray(sample.actuator.previous));np.savez_compressed(case_dir/'B0_alpha1.npz',**b1);audits[case]['B0_alpha1']=reconstruct_trace(b1,spec,np.asarray(sample.actuator.previous),controller)
        assert all(v['passed'] for v in audits[case].values())
        generate_report(out,spec)
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
    write(out/'review/audit.json',audits);write(out/'status.json',dict(state='complete',episodes=6,elapsed_s=time.monotonic()-start))
    (out/'review/INDEX.md').write_text(
        '# Frozen lower / independent upper endpoint comparison\n\n'
        f'Lower: {alias}, fixed lower_alpha=1. Independent upper Actors at update250.\n\n'
        '[Physical metrics and plots](REPORT.md) · [Metrics](metrics.json) · [Reward audit](audit.json)\n\n'
        '[Main paired rewards](main_paired_rewards.png) · [Random paired rewards](random_paired_rewards.png)\n\n'
        'B0 is this frozen residual lower with zero upper correction. Its physical trajectory is shared; '
        'rewards are rescored separately for each upper alpha. These are two independent upper policies.\n')
