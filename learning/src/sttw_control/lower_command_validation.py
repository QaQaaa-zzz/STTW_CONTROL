"""Fixed late local-command validation; V5.2 safety-first best machinery.

This protocol qualifies the declared local task only. Adoption also requires
comparison at the original parent mismatch states; that is a separate gate.
"""
from pathlib import Path
import json,pickle
import numpy as np
from .preference_best import COUNT_NAMES,selection_key,qualified,is_better,persist_best,load_best,config_hash
from .lower_tracking_audit import held,intervals

def validate_trace(d,dt,horizon):
    required={'time':(), 'limited_command':(2,), 'actual_forward_speed':(), 'actual_delta':(),
              'phi':(), 'peak_roll':(), 'physical_failure':(), 'raw_rates':(2,),
              'final_command_clipped':(), 'actual_delta_rate':()}
    if set(required)-set(d):raise ValueError('missing physical validation fields')
    t=np.asarray(d['time']);n=len(t)
    if not n or n>round(horizon/dt):raise ValueError('invalid validation length')
    for key,tail in required.items():
        x=np.asarray(d[key])
        if x.shape!=(n,)+tail or not np.isfinite(x).all():raise ValueError('invalid/nonfinite '+key)
    if not np.allclose(t,np.arange(n)*dt,atol=2e-6,rtol=0):raise ValueError('noncontiguous validation time')
    for key in ('nonfinite','policy_fault','lower_fault'):
        if key in d and (np.asarray(d[key]).shape!=(n,) or not np.isfinite(d[key]).all() or np.any(d[key])):
            raise ValueError('invalid validation fault '+key)
    failures=np.flatnonzero(d['physical_failure'])
    if failures.size and not (failures.size==1 and failures[0]==n-1):raise ValueError('data after failure')
    if not failures.size and n!=round(horizon/dt):raise ValueError('incomplete validation')


def summarize_local(traces,spec):
    cfg=spec['validation'];dt=spec['dt'];horizon=cfg['seconds']
    if set(traces)!=set(cfg['case_ids']):raise ValueError('all fixed local cases required')
    cases={}
    for name,d in traces.items():
        validate_trace(d,dt,horizon)
        t=np.asarray(d['time']);n=len(t)
        if not n:raise ValueError('empty validation')
        err=np.column_stack([d['actual_forward_speed'],d['actual_delta']])-d['limited_command']
        peak=np.maximum(np.abs(d['phi']),d['peak_roll'])
        if not all(np.isfinite(x).all() for x in [t,err,peak]):raise ValueError('nonfinite validation cannot select best')
        failed=bool(np.any(d['physical_failure']));complete=failed or t[-1]+dt>=horizon-1e-4
        rates=np.asarray(d['raw_rates']);threshold=cfg.get('stable_rate_thresholds',[.1,.02]);stable=held((abs(rates[:,0])<threshold[0])&(abs(rates[:,1])<threshold[1]),dt,cfg['settle_s'])&(t>=1.)
        transition=~stable;final=t>=horizon-cfg['final_hold_s']-1e-6
        tol=np.array([cfg['speed_tolerance'],cfg['steer_tolerance']]);rms=lambda mask:np.sqrt(np.mean(err[mask]**2,axis=0)) if mask.any() else np.full(2,np.inf)
        small_mask=np.zeros(n,bool);small_valid=True;direction=True
        if name.startswith('post_return_small') and 'small_steer_tolerance' in cfg:
            target=cfg['post_small_target'];start=cfg['post_small_target_start_s']
            plateau=(t>=start)&(np.abs(np.abs(d['limited_command'][:,1])-target)<1e-6)&(abs(rates[:,1])<threshold[1])
            hits=np.flatnonzero(plateau)
            if hits.size:small_mask=(t>=t[hits[0]]+cfg['settle_s']-1e-6)
            small_valid=small_mask.sum()*dt>=cfg['minimum_stable_s']
            direction=bool(small_valid and np.all(d['actual_delta'][small_mask]*d['limited_command'][small_mask,1]>0))

        primary=stable.sum()*dt>=cfg['minimum_stable_s'] and bool(np.all(rms(stable)<=tol))
        if name.startswith('post_return_small') and 'small_steer_tolerance' in cfg:
            primary=primary and small_valid and direction and bool(rms(small_mask)[1]<=cfg['small_steer_tolerance'])
            tol[1]=cfg['small_steer_tolerance']
        joint=final.sum()>=round(cfg['final_hold_s']/dt) and bool(np.all(abs(err[final])<=tol)) and not failed
        q=float(np.mean((err/tol)**2)+2*np.mean(peak>cfg['working_roll_limit']))
        windows={}
        for label,mask in [('full',np.ones(n,bool)),('steady',stable),('transition',transition),('return',(t>=4.5)&(t<7.5)),('post_return_small',small_mask)]:
            windows[label]=dict(ticks=int(mask.sum()),rmse=rms(mask).tolist() if mask.any() else None,
                error_min=err[mask].min(0).tolist() if mask.any() else None,error_max=err[mask].max(0).tolist() if mask.any() else None)
        cases[name]=dict(complete=bool(complete),physical_failure=failed,working_limit_failure=bool(np.any(peak>cfg['working_roll_limit'])),
            small_window_valid=bool(small_valid),small_direction_pass=bool(direction),historical_loose_stable_rmse=rms(stable).tolist(),primary_failure=not primary,joint_final_hold_failure=not joint,physical_quality_score=q,peak_roll=float(peak.max()),
            windows=windows,steer_exceed_intervals=intervals(abs(err[:,1])>tol[1],t,dt),speed_exceed_intervals=intervals(abs(err[:,0])>tol[0],t,dt),
            final_clip_fraction=float(np.mean(d['final_command_clipped'])),observed_ticks=n)
    summary={k:sum(int(x[k.removesuffix('_cases')]) for x in cases.values()) for k in COUNT_NAMES}
    summary.update(cases=cases,all_declared_traces_complete=all(x['complete'] for x in cases.values()),physical_quality_score=float(np.mean([x['physical_quality_score'] for x in cases.values()])))
    summary['score_tuple']=list(selection_key(summary));summary['qualified']=qualified(summary)
    summary['qualification_scope']=cfg['protocol']+' only; parent-state repair gate separate'
    return summary

def export_local_actor(policy):
    """Same Flax ELU/tanh residual implementation; fresh210D weights only."""
    import torch
    layers=[x for x in policy.actor.modules() if isinstance(x,torch.nn.Linear)]
    return {'params':{f'Dense_{i}':dict(kernel=l.weight.detach().cpu().numpy().T,bias=l.bias.detach().cpu().numpy()) for i,l in enumerate(layers)}}

def evaluate_fixed(training,algo,update,checkpoint,*,select=True):
    import jax,jax.numpy as jp
    from .lower_command import local_command_rows,publish_local_command
    from .network import ResidualActor
    from .direct_command_training import write
    e=training.env;cfg=training.cfg;v=cfg['validation'];sample=jax.tree.map(lambda x:x[3],training.bank)
    params=jax.tree.map(jp.asarray,export_local_actor(algo.policy));net=ResidualActor(tuple(cfg['hidden_sizes']),activation='elu')
    # Publish once without physics; both controllers consume these exact samples.
    streams=reference_streams(training)
    count=len(v['case_ids'])
    if not hasattr(training,'local_validation_chunk'):
        def active(s,params):
            obs,_=e.observation(s);z=net.apply(params,obs,return_logits=True)
            ns,_,_,log=e.step(s,z);return ns,log
        def chunk(states,params):
            def step(states,_):
                def one(s):
                    return jax.lax.cond(s.physical.failed,lambda s:(s,training.local_validation_zero),lambda s:active(s,params),s)
                return jax.vmap(one)(states)
            return jax.lax.scan(step,states,None,length=200)
        states=local_validation_states(training,streams)
        example=jax.tree.map(lambda x:None if x is None else x[0],states)
        shape=jax.eval_shape(lambda s:e.step(s,jp.zeros(2)),example)[3]
        training.local_validation_zero=jax.tree.map(lambda x:jp.zeros(x.shape,x.dtype),shape)
        training.local_validation_chunk=training.compile('seven local cases 1s batched',chunk,states,params)
    states=local_validation_states(training,streams)
    dest=training.out/'training'/('final_best_validation' if not select else f'validation_{update:04d}');dest.mkdir(parents=True,exist_ok=True);traces={};chunks=[]
    for _ in range(round(v['seconds'])):
        states,logs=training.local_validation_chunk(states,params);jax.block_until_ready(states);chunks.append(jax.device_get(logs))
    from .direct_command_training import flatten_logs
    logs=jax.tree.map(lambda *x:np.concatenate(x),*chunks)
    for i,case in enumerate(v['case_ids']):
        d=flatten_logs(jax.tree.map(lambda x:x[:,i],logs));used=int(states.tick[i]);d={k:x[:used] for k,x in d.items()};traces[case]=d
        np.savez_compressed(dest/f'{case}.npz',**d)
    summary=summarize_local(traces,cfg)
    baseline=ensure_r196_reference(training)
    summary['R196_reference']=summarize_local(baseline,cfg)
    for case in traces:
        n=min(len(traces[case]['time']),len(baseline[case]['time']))
        np.testing.assert_array_equal(traces[case]['limited_command'][:n],baseline[case]['governed'][:n])
        np.testing.assert_array_equal(traces[case]['raw_rates'][:n],baseline[case]['raw_rates'][:n])
    summary['parent_mismatch_same_state_gate']='not_run; local protocol cannot prove repair of the old aggressive state'
    write(dest/'metrics.json',summary)
    if select:
        root=training.out/'training';pointer=root/'best_model.json';old=json.loads(pointer.read_text()) if pointer.exists() else None
        actor=dest/'actor.pkl'
        with actor.open('wb') as f:pickle.dump(export_local_actor(algo.policy),f)
        if is_better(summary,None if old is None else old['metrics']):
            identity=dict(source_update=update,config=cfg,config_sha256=config_hash(cfg),protocol=v,protocol_sha256=config_hash(v),
                reward_version=cfg['reward']['version'],adoption_status='not_adopted',parent_mismatch_same_state_gate='not_run')
            persist_best(root,checkpoint,actor,summary,identity,[x for x in cfg['validation_updates'] if x<=update])
    return summary


def ensure_r196_reference(training):
    """Future validation: matched full preparation, rows, slew and frozen R196.

    Reuse only a checked protocol/physics/bank/controller identity in this run.
    No call occurs in preparation-only mode or at20/25 updates.
    """
    import jax,jax.numpy as jp
    from dataclasses import asdict
    from .direct_command_env import DirectCommandEnv
    from .smooth_command_config import resolve_preference_v51
    from .lower_command import local_command_rows
    from .direct_command_training import flatten_logs,write
    from .preference_best import digest
    cfg=training.cfg;root=training.out/'training'/'R196_reference';spec=resolve_preference_v51(0);spec['prepublished_local_reference']=True
    env=DirectCommandEnv(spec);sample=jax.tree.map(lambda x:x[3],training.bank)
    streams=reference_streams(training)
    identity=dict(reference_stream_sha256=config_hash({k:np.asarray(x).tolist() for k,x in streams.items()}),protocol=cfg['validation'],command_design=cfg['command_design'],command_seed=cfg['command_seed'],
        prepared_sha256=digest(cfg['prepared_bank']),prepared_index=3,lower=env.lower.provenance,
        model=env.physics.bundle.identity,controller=asdict(env.cc),actuator=asdict(env.ac),physics_dt=float(env.physics.model.opt.timestep),control_dt=env.cc.dt)
    marker=root/'identity.json';expected=config_hash(identity)
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['identity_sha256']!=expected:raise ValueError('R196 fixed validation cache identity mismatch')
        traces={case:dict(np.load(root/f'{case}.npz')) for case in cfg['validation']['case_ids']}
        for case in traces:
            if digest(root/f'{case}.npz')!=old['trace_sha256'][case]:raise ValueError('R196 trace cache SHA mismatch')
            validate_trace(traces[case],cfg['dt'],cfg['validation']['seconds'])
        return traces
    states=[]
    for i,case in enumerate(cfg['validation']['case_ids']):
        states.append(env.reset(sample,jp.int32(88001+i),jp.int32(0),jp.asarray(0.),streams[case],jp.array([.5,.3])))
    env.set_log_template(states[0])
    states=jax.tree.map(lambda *x:jp.stack(x),*states)
    def chunk(st):
        def step(st,_):
            ns,_,_,log,_=jax.vmap(lambda s:env.policy_step(s,jp.zeros(2),True))(st);return ns,log
        return jax.lax.scan(step,st,None,length=50)
    fn=training.compile('matched R196 seven cases 1s batched',chunk,states);traces={};root.mkdir(parents=True,exist_ok=True);chunks=[]
    for _ in range(round(cfg['validation']['seconds'])):
        states,log=fn(states);jax.block_until_ready(states);chunks.append(jax.device_get(log))
    logs=jax.tree.map(lambda *x:np.concatenate(x),*chunks)
    for i,case in enumerate(cfg['validation']['case_ids']):
        d=flatten_logs(jax.tree.map(lambda x:x[:,i],logs));d={k:v.reshape((-1,)+v.shape[2:]) for k,v in d.items()};active=d['active_tick'];d={k:v[active] for k,v in d.items()}
        validate_trace(d,cfg['dt'],cfg['validation']['seconds'])
        traces[case]=d;np.savez_compressed(root/f'{case}.npz',**d)
    write(marker,dict(identity=identity,identity_sha256=expected,trace_sha256={case:digest(root/f'{case}.npz') for case in traces}))
    return traces


def reference_streams(training):
    import jax,jax.numpy as jp
    from .lower_command import fixed_reference_stream
    if not hasattr(training,'issued_streams'):
        raw=jax.tree.map(lambda x:x[3],training.bank).raw
        training.issued_streams={case:jax.jit(lambda:fixed_reference_stream(raw,training.cfg,case,training.env.cc))() for case in training.cfg['validation']['case_ids']}
        root=training.out/'training'/'reference_streams';root.mkdir(parents=True,exist_ok=True)
        for case,stream in training.issued_streams.items():np.save(root/(case+'.npy'),np.asarray(stream))
    return training.issued_streams


def local_validation_states(training,streams):
    import jax,jax.numpy as jp
    sample=jax.tree.map(lambda x:x[3],training.bank);e=training.env;states=[]
    for i,(case,stream) in enumerate(streams.items()):
        st=e.reset(sample,jp.int32(88001+i),jp.int32(0))
        st=st.replace(issued_stream=stream,rates=stream[0,3:5],physical=st.physical.replace(raw=stream[0,1:3]),frames=jp.zeros_like(st.frames),mask=jp.zeros_like(st.mask))
        states.append(e.record(st))
    return jax.tree.map(lambda *x:jp.stack(x),*states)
