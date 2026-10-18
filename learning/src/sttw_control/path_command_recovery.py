"""Explicit full-state takeover using the existing scan50 path executor.

Snapshots contain every JAX leaf, including MJX warm starts, controller/ESO,
actuator delay, lower and upper histories, filters, projection and absolute time.
A live template supplies static pytree metadata; incompatible templates fail.
No function changes plant state or creates an artificial roll perturbation.
"""
from pathlib import Path
from dataclasses import asdict
import json
import math
import numpy as np
import jax
import jax.numpy as j
from .preference_best import atomic_json
from .path_command_policy import config_key
from .path_command_env import PathState
from .geometric_path import Path as FixedPath

SCHEMA='path_complete_state_recovery_v1'
REQUIRED_IDENTITIES=('schema','lower','model','controller','actuator','timing','bank','physical_config','terminal_contract')
ROUTES=('left90_R2','right90_R2')


def _identity(identity):
    if not isinstance(identity,dict) or any(k not in identity or identity[k] in (None,'',{},[]) for k in REQUIRED_IDENTITIES):
        raise ValueError('missing recovery identity signatures')
    if identity['terminal_contract']!='terminal_contract_v2':
        raise ValueError('recovery requires terminal_contract_v2')
    return identity


def make_identity(env,source):
    """Record live physical settings and source paths without file hashing."""
    from .path_command_fork import physical_config_identity

    return _identity(dict(schema=env.schema,lower=source['candidate']['actor_sha256'],
        model=config_key(env.physics.bundle.identity),controller=config_key(asdict(env.cc)),
        actuator=config_key(asdict(env.ac)),timing=config_key(env.config['timing']),
        bank=dict(sha256=source['prepared_bank_sha256'],index=source['bank_index']),
        physical_config=physical_config_identity(env.config),terminal_contract=env.terminal_contract))


def _tree_signature(tree):
    leaves,structure=jax.tree.flatten(tree)
    return dict(structure=str(structure),leaves=[dict(shape=list(np.shape(x)),dtype=str(np.asarray(x).dtype),weak_type=bool(getattr(x,'weak_type',False))) for x in leaves])


def _restore_leaf(value,weak_type=False):
    array=np.asarray(value)
    restored=j.asarray(array.item()) if weak_type and array.shape==() else j.asarray(array)
    if restored.dtype!=array.dtype or bool(getattr(restored,'weak_type',False))!=weak_type:
        raise ValueError('restored state precision/weak-type signature mismatch')
    return restored


def _valid_state(state):
    if not isinstance(state,PathState):raise ValueError('complete PathState required')
    for x in jax.tree.leaves(state):
        if not np.isfinite(np.asarray(x)).all():raise ValueError('nonfinite recovery state')
    if bool(state.physical.failed)|bool(state.fault)|bool(state.domain_exit):raise ValueError('faulted recovery state')
    if int(state.tick)<=0 or int(state.tick)%4:raise ValueError('snapshot must be an upper boundary')


def _validate_trace(data,start_tick,dt,expected_end=None):
    required=('time','peak_roll','phi','actual_forward_speed','v_user','path_cross_track','path_heading_error',
        'path_progress','goal_progress','goal_section_signed_distance','chi','physical_failure','domain_exit','policy_fault','lower_fault')
    if any(k not in data for k in required):raise ValueError('missing recovery trace measurements')
    n=len(data['time'])
    if not n or any(np.asarray(data[k]).shape!=(n,) for k in required):raise ValueError('invalid recovery trace shape')
    if any(not np.isfinite(np.asarray(data[k])).all() for k in required):raise ValueError('nonfinite recovery trace')
    if np.any(np.asarray(data['peak_roll'])<np.abs(np.asarray(data['phi']))-1e-7):raise ValueError('inconsistent actual peak roll')
    if not np.allclose(data['time'],np.arange(start_tick,start_tick+n)*dt,rtol=0,atol=dt*.002):raise ValueError('recovery trace must be contiguous absolute ticks')
    if 'active_tick' in data and not np.all(data['active_tick']):raise ValueError('failure padding cannot be saved')
    terminal=np.asarray(data['physical_failure'],bool)|np.asarray(data['domain_exit'],bool)
    if terminal[:-1].any():raise ValueError('trace continues after failure')
    if expected_end is not None and start_tick+n!=expected_end:raise ValueError('missing snapshot prefix/time context')
    return n


def save_snapshot(output,state,path,prefix,identity,*,route,v_user,horizon,boundary_phi,config_key=None):
    """Immutable numeric archive plus checked schema/identity/checksum receipt."""
    _identity(identity);_valid_state(state)
    if route not in ROUTES or v_user!=2.6 or horizon!=20.:raise ValueError('recovery fixed high-turn contract changed')
    if not math.isfinite(boundary_phi) or not .302<abs(boundary_phi)<=.40:raise ValueError('no qualifying natural boundary')
    _validate_trace(prefix,0,.005,int(state.tick))
    boundary_phis=np.asarray(prefix['phi'])[3::4]
    qualifying=(abs(boundary_phis)>.302)&(abs(boundary_phis)<=.40)
    if not qualifying[-1] or qualifying[:-1].any() or not np.isclose(boundary_phis[-1],boundary_phi,rtol=0,atol=1e-7):raise ValueError('not first natural qualifying upper boundary')
    if np.any(prefix['policy_fault']) or np.any(prefix['lower_fault']) or np.any(prefix['physical_failure']) or np.any(prefix['domain_exit']):raise ValueError('faulted snapshot prefix')
    if not isinstance(path,FixedPath) or not np.all(np.diff(path.s)>0):raise ValueError('invalid fixed original path')
    if int(state.tick)>=4000:raise ValueError('snapshot has no original horizon remaining')
    out=Path(output)
    if out.exists():raise FileExistsError('immutable recovery snapshot exists')
    out.mkdir(parents=True)
    leaves=jax.tree.leaves(state)
    arrays={f'state_{i:05d}':np.asarray(x) for i,x in enumerate(leaves)}
    for key in ('s','xy','heading','curvature','turn_end','goal'):arrays['path_'+key]=np.asarray(getattr(path,key))
    for key,x in prefix.items():arrays['prefix_'+key]=np.asarray(x)
    np.savez_compressed(out/'leaves.npz',**arrays)
    meta=dict(schema=SCHEMA,identity=identity,state_signature=_tree_signature(state),
        path_signature=_tree_signature(path),payload_sha256=None,route=route,v_user=v_user,
        original_horizon_s=horizon,lower_dt_s=.005,upper_dt_s=.02,start_tick=int(state.tick),
        absolute_takeover_s=int(state.tick)*.005,physical_tick=int(state.physical.physical_tick),
        boundary_phi_rad=float(boundary_phi),source_control='B0_zero_upper_frozen_local300_ECBC_ESO',
        capture_rule='first active upper boundary .302<abs(phi)<=.40',config_key=config_key)
    meta['receipt_sha256']=None;atomic_json(out/'snapshot.json',meta)
    return meta


def load_snapshot(output,template,identity):
    _identity(identity);out=Path(output);meta=json.loads((out/'snapshot.json').read_text())
    if meta.get('schema')!=SCHEMA or meta.get('identity')!=identity:raise ValueError('snapshot identity mismatch')
    template_signature=_tree_signature(template);saved_signature=meta['state_signature']
    compatible=template_signature['structure']==saved_signature['structure'] and len(template_signature['leaves'])==len(saved_signature['leaves'])
    compatible=compatible and all(a['shape']==b['shape'] and a['dtype']==b['dtype'] for a,b in zip(template_signature['leaves'],saved_signature['leaves']))
    if not compatible:raise ValueError('complete state template signature mismatch')
    leaves,tree=jax.tree.flatten(template)
    with np.load(out/'leaves.npz',allow_pickle=False) as archive:
        expected={f'state_{i:05d}' for i in range(len(leaves))}|{'path_'+k for k in ('s','xy','heading','curvature','turn_end','goal')}
        if not expected<=set(archive.files) or any(k not in expected and not k.startswith('prefix_') for k in archive.files):raise ValueError('snapshot leaf coverage mismatch')
        restored=[]
        for i,original in enumerate(leaves):
            x=archive[f'state_{i:05d}']
            if x.shape!=original.shape or x.dtype!=original.dtype:raise ValueError('snapshot leaf signature mismatch')
            restored.append(_restore_leaf(x,saved_signature['leaves'][i]['weak_type']))
        state=jax.tree.unflatten(tree,restored)
        path=FixedPath(*(_restore_leaf(archive['path_'+k],meta['path_signature']['leaves'][i]['weak_type']) for i,k in enumerate(('s','xy','heading','curvature','turn_end','goal'))))
        prefix={k[7:]:archive[k].copy() for k in archive.files if k.startswith('prefix_')}
    _valid_state(state)
    if _tree_signature(state)!=meta['state_signature']:raise ValueError('restored state precision signature mismatch')
    if _tree_signature(path)!=meta['path_signature'] or int(state.tick)!=meta['start_tick'] or int(state.physical.physical_tick)!=meta['physical_tick']:
        raise ValueError('snapshot path/time signature mismatch')
    _validate_trace(prefix,0,meta['lower_dt_s'],int(state.tick))
    if not .302<abs(meta['boundary_phi_rad'])<=.40 or meta['original_horizon_s']!=20. or meta['v_user']!=2.6 or meta['route'] not in ROUTES:
        raise ValueError('snapshot natural boundary contract mismatch')
    return state,path,meta,prefix


def clone_for_alpha(state,alpha):
    if alpha not in (0,1):raise ValueError('recovery endpoint upper alpha required')
    _valid_state(state)
    copied=jax.tree.map(lambda x:_restore_leaf(np.asarray(x).copy(),bool(getattr(x,'weak_type',False))),state)
    return copied.replace(alpha=_restore_leaf(np.asarray(alpha,dtype=state.alpha.dtype),bool(getattr(state.alpha,'weak_type',False))))


def first_qualifying_boundary(history,phis,active):
    phis=np.asarray(phis);ticks=np.asarray(history.tick)
    if phis.shape!=ticks.shape or np.asarray(active).shape!=ticks.shape:raise ValueError('boundary history shape mismatch')
    valid=np.asarray(active,bool)&(ticks%4==0)&(ticks>0)&(ticks<4000)&np.isfinite(phis)&(abs(phis)>.302)&(abs(phis)<=.40)
    valid&=~np.asarray(history.physical.failed)&~np.asarray(history.fault)&~np.asarray(history.domain_exit)
    candidates=np.flatnonzero(valid)
    return int(candidates[0]) if len(candidates) else None


def _flat_single(logs):
    from .direct_command_training import flatten_logs
    flat=flatten_logs(logs)
    data={k:np.asarray(v)[:,0].reshape((-1,)+np.shape(v)[3:]) for k,v in flat.items()}
    active=np.asarray(data['active_tick'],bool)
    return {k:v[active] for k,v in data.items()}


def _concatenate(chunks):
    if not chunks:raise ValueError('no actual recovery transitions')
    if any(set(d)!=set(chunks[0]) for d in chunks):raise ValueError('inconsistent recovery log fields')
    return {k:np.concatenate([d[k] for d in chunks],axis=0) for k in chunks[0]}


def evaluate_from_state(evaluator,state,path,v_user,params,alpha):
    """Inject a complete clone into the old compiled executor; reset is not run."""
    cfg=evaluator.config;dt=cfg['timing']['lower_dt_s'];horizon=cfg['training_future_phase_C']['episode_s']
    if dt!=.005 or cfg['timing']['upper_dt_s']!=.02 or horizon!=20. or evaluator.env.terminal_contract!='terminal_contract_v2':raise ValueError('recovery timing/terminal contract mismatch')
    initial=clone_for_alpha(state,alpha);start_tick=int(initial.tick)
    paths=jax.tree.map(lambda x:x[None],path);speeds=j.asarray([v_user],dtype=initial.alpha.dtype)
    states=jax.tree.map(lambda x:x[None],initial)
    _,chunk=evaluator._compiled_group(paths,speeds,initial.alpha,params,horizon)
    chunks=[]
    for _ in range(math.ceil((round(horizon/dt)-start_tick)/200)):
        evaluator.check_budget();states,logs=chunk(states,paths,speeds,params);jax.block_until_ready(states)
        host=jax.device_get(logs);d=_flat_single(host)
        if d['time'].size:chunks.append(d)
        evaluator.account(len(d['time']))
        if np.any(states.fault):raise ValueError('recovery engineering fault; no qualification')
        if np.all(np.asarray(states.physical.failed)|np.asarray(states.domain_exit)|(np.asarray(states.tick)>=round(horizon/dt))):break
    data=_concatenate(chunks);data['goal_progress']=np.full(len(data['time']),float(path.goal))
    _validate_trace(data,start_tick,dt)
    return jax.tree.map(lambda x:x[0],states),data


def capture_natural_snapshots(evaluator,zero_params,output,identity):
    """Run B0 once per fixed high turn; retain first qualifying full boundary."""
    from .geometric_path import build_path
    out=Path(output)
    if out.exists():raise FileExistsError('immutable recovery capture exists')
    out.mkdir(parents=True);cfg=evaluator.config
    if cfg['training_future_phase_C']['episode_s']!=20. or cfg['timing']['lower_dt_s']!=.005 or cfg['timing']['upper_dt_s']!=.02:raise ValueError('capture timing mismatch')
    _identity(identity)
    pose=np.asarray(evaluator.env.physics.helpers.pose(evaluator.snapshot.data));result={};charged=0
    for route in ROUTES:
        path=build_path(route,pose,cfg);paths=jax.tree.map(lambda x:x[None],path)
        speeds=j.asarray([2.6],j.float32);alpha=j.array(0.,j.float32)
        reset,chunk=evaluator._compiled_group(paths,speeds,alpha,zero_params,20.,capture_states=True)
        states=reset(paths,speeds,alpha);prefix=[];found=False
        if bool(j.any(states.fault|states.domain_exit|states.physical.failed)):raise ValueError('invalid B0 initial state')
        for _ in range(20):
            evaluator.check_budget();states,(logs,history)=chunk(states,paths,speeds,zero_params);jax.block_until_ready(states)
            host=jax.device_get(logs);data=_flat_single(host);evaluator.account(len(data['time']));charged+=len(data['time'])
            if charged>8000:raise ValueError('capture exceeds declared two-prefix budget')
            if data['time'].size:prefix.append(data)
            if np.any(states.fault):raise ValueError('B0 capture engineering fault')
            single=jax.tree.map(lambda x:x[:,0],history)
            phis=jax.vmap(lambda s:evaluator.env.physics.observe(s.physical.data)[0][0])(single)
            active=np.any(np.asarray(host['active_tick'])[:,0],axis=1)
            index=first_qualifying_boundary(single,np.asarray(phis),active)
            if index is not None:
                boundary=jax.tree.map(lambda x:x[index],single);d=_concatenate(prefix)
                d={k:v[:int(boundary.tick)] for k,v in d.items()};d['goal_progress']=np.full(len(d['time']),float(path.goal))
                result[route]=save_snapshot(out/route,boundary,path,d,identity,route=route,v_user=2.6,horizon=20.,boundary_phi=float(phis[index]),config_key=config_key(cfg))
                found=True;break
            if np.any(states.fault):raise ValueError('B0 capture engineering fault')
            if np.all(np.asarray(states.physical.failed)|np.asarray(states.domain_exit)|(np.asarray(states.tick)>=4000)):break
        if not found:result[route]=dict(status='NOT_VERIFIED',reason='no qualifying natural upper boundary')
    receipt=dict(schema=SCHEMA,identity=identity,cases=result,training_steps=0,active_prefix_lower_ticks=charged,completed=True)
    receipt['receipt_sha256']=None;atomic_json(out/'capture.json',receipt)
    return result


def recovery_oracle(tail,config,*,start_tick,initial_phi,route,alpha,prefix):
    """Predeclared roll recovery plus original horizon/goal/last-second gates."""
    dt=config['timing']['lower_dt_s'];horizon=config['training_future_phase_C']['episode_s']
    if dt!=.005 or horizon!=20. or config['timing']['upper_dt_s']!=.02 or not .302<abs(initial_phi)<=.40 or alpha not in (0,1):raise ValueError('invalid recovery oracle contract')
    n=_validate_trace(tail,start_tick,dt);_validate_trace(prefix,0,dt,start_tick)
    if not np.isclose(float(prefix['phi'][-1]),initial_phi,rtol=0,atol=1e-7):raise ValueError('initial boundary roll/prefix mismatch')
    for data in (tail,prefix):
        if np.any(data['policy_fault']) or np.any(data['lower_fault']):raise ValueError('engineering fault cannot qualify recovery')
    roll=np.asarray(tail['peak_roll']);run=0;completed=None
    for i,x in enumerate(roll[:round(1./dt)]):
        run=run+1 if x<=.28 else 0
        if run>=round(.5/dt):completed=i+1;break
    recovered=completed is not None
    later=roll[completed:] if recovered else roll
    failures=bool(np.any(tail['physical_failure'])|np.any(tail['domain_exit']))
    full={k:np.concatenate([np.asarray(prefix[k]),np.asarray(tail[k])]) for k in prefix if k in tail}
    t=np.asarray(full['time'])+dt;ev=full['actual_forward_speed']-full['v_user'];ey=full['path_cross_track'];eh=full['path_heading_error']
    b=config['stage_B_baseline'];hold=t>horizon-b['hold_s']+1e-5
    full_end=start_tick+n==round(horizon/dt)
    final_hold=bool(full_end and hold.sum()==round(b['hold_s']/dt) and np.all(abs(ey[hold])<=b['final_y_error_m']) and np.all(abs(eh[hold])<=b['final_path_heading_error_rad']) and np.all(abs(ev[hold])<=b['final_speed_error_m_s']) and not failures)
    goal=bool(np.any((full['goal_section_signed_distance']>=0)&(full['path_progress']>=full['goal_progress'])))
    corridor=bool(np.all(abs(ey)<=.40));conflict=full['chi']>0;selected=conflict if conflict.any() else np.ones(len(t),bool)
    rms=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2)))
    primary=max(rms(ev/.10),rms(ey/.10)) if not conflict.any() else rms((ey/.10 if alpha==0 else ev/.05)[selected])
    stop=np.flatnonzero(roll<=.302);initial_end=int(stop[0]) if len(stop) else n
    prior_roll=np.asarray(prefix['peak_roll']);clear=np.flatnonzero(prior_roll<=.302)
    initial_start=int(clear[-1]+1) if len(clear) else 0
    initial_segment=slice(initial_start,start_tick+initial_end)
    initial_roll=np.concatenate([prior_roll[initial_start:],roll[:initial_end]])
    return dict(schema=SCHEMA,qualified=bool(recovered and not np.any(later>.302) and not failures and corridor and final_hold and goal),
        recovered_within_1s=recovered,recovery_completed_s=completed*dt if recovered else None,
        recovery_hold_s=.5,observed_lower_ticks=n,start_tick=start_tick,original_horizon_s=horizon,
        full_original_end=full_end,physical_failure=bool(np.any(tail['physical_failure'])),domain_exit=bool(np.any(tail['domain_exit'])),
        corridor_pass=corridor,goal=goal,final_hold=final_hold,primary_band_pass=primary<=1.,primary_normalized_rmse=primary,
        initial_phi_rad=float(initial_phi),takeover_peak_roll_rad=float(max(abs(initial_phi),roll.max())),
        initial_overlimit_peak_rad=float(max(abs(initial_phi),initial_roll.max() if initial_roll.size else 0.)),initial_overlimit_duration_s=(start_tick-initial_start+initial_end)*dt,
        initial_overlimit_start_absolute_s=initial_start*dt,full_episode_overlimit_s=float(np.sum(full['peak_roll']>.302))*dt,
        total_takeover_overlimit_s=float(np.sum(roll>.302))*dt,later_peak_roll_rad=float(later.max()) if len(later) else None,
        later_overlimit_s=float(np.sum(later>.302))*dt,speed_loss_integral_m=float(np.maximum(-ev,0).sum()*dt),
        speed_gain_integral_m=float(np.maximum(ev,0).sum()*dt),speed_error_squared_integral=float(np.sum(ev**2)*dt),
        path_error_squared_integral=float(np.sum(ey**2)*dt),initial_segment_speed_loss_m=float(np.maximum(-ev[initial_segment],0).sum()*dt),
        initial_segment_path_cost=float(np.sum(ey[initial_segment]**2)*dt))


def validate_recovery_arm(*,name,alpha,manifest,run_status,training_status,selected,final_source,source,lower_sha256):
    """Pure preflight: final completed arm role/source/selection before any tail."""
    if name not in ('control0','candidate0','control1','candidate1') or alpha!=int(name[-1]):
        raise ValueError('recovery arm independent endpoint mismatch')
    plan=manifest.get('plan',{});cfg=manifest.get('config',{})
    role=name[:-1]
    if plan.get('experiment_arm')!=role:raise ValueError('recovery arm role mismatch')
    reward=cfg.get('path_reward',{})
    if (role=='candidate' and reward.get('reward_version')!='path_safety_preference_v1') or (role=='control' and 'reward_version' in reward):
        raise ValueError('recovery arm reward version/role mismatch')
    if manifest.get('terminal_contract')!='terminal_contract_v2' or plan.get('terminal_contract')!='terminal_contract_v2':
        raise ValueError('recovery arm terminal contract mismatch')
    if not source or config_key(manifest.get('source'))!=config_key(source):
        raise ValueError('recovery arm complete source mismatch')
    target=plan.get('updates')
    if (run_status.get('state')!='completed' or training_status.get('state')!='completed'
        or type(target) is not int or target<=0 or training_status.get('declared_updates')!=target
        or training_status.get('completed_updates')!=target):
        raise ValueError('recovery arm training not completed at declared endpoint')
    if config_key(selected)!=config_key(final_source):raise ValueError('recovery selected/final receipt mismatch')
    if (selected.get('selection_schema')!='path_safety_task_v1' or selected.get('alpha')!=alpha
        or selected.get('lower_actor_sha256')!=lower_sha256 or selected.get('config_key')!=config_key(cfg)):
        raise ValueError('recovery selected endpoint/config/lower identity mismatch')
    return selected


def evaluate_pairs(evaluator,snapshots,template,identity,arms,output):
    """Explicit B0/control/candidate endpoint tails, with actor+alpha reuse."""
    names={'B0','control0','candidate0','control1','candidate1'}
    if set(arms)!=names:raise ValueError('complete five-arm recovery comparison required')
    for name,arm in arms.items():
        if arm.get('alpha') not in (0,1):raise ValueError('missing selected arm identity')
        if name!='B0' and (arm.get('state')!='completed' or arm.get('selected') is not True):raise ValueError('final selected completed arms required')
    out=Path(output)
    if out.exists():raise FileExistsError('immutable paired recovery output exists')
    out.mkdir(parents=True);results={};used=0;report_cases={}
    for route in ROUTES:
        source=Path(snapshots)/route
        if not source.exists():results[route]=dict(status='NOT_VERIFIED',reason='no qualifying B0 snapshot');continue
        state,path,meta,prefix=load_snapshot(source,template,identity);cache={};results[route]={}
        report_cases[route]=dict(path=path,prefix=prefix,traces={},configs={},alphas={},actor_sha256={},
            snapshot_sha256=meta['receipt_sha256'],takeover_s=meta['absolute_takeover_s'])
        for name,arm in arms.items():
            key=(id(arm['params']),arm['alpha'])
            runner=arm.get('evaluator',evaluator)
            from .path_command_fork import physical_config_identity
            if physical_config_identity(runner.config)!=identity['physical_config']:raise ValueError('paired evaluator physical identity mismatch')
            if not np.isclose(float(runner.env.physics.observe(state.physical.data)[0][0]),meta['boundary_phi_rad'],rtol=0,atol=1e-7):raise ValueError('snapshot physical roll identity mismatch')
            if key in cache:
                original,tail=cache[key]
                reuse=original['arm']
                tail={k:v.copy() for k,v in tail.items()}
                from .path_command_reporting import reconstruct_costs,reward_series
                rebuilt=reconstruct_costs(tail,arm['alpha'],runner.config)
                for group,prefix_name in [('raw','raw_cost_'),('effective','effective_cost_'),('capped','component_capped_'),('diagnostic_subcosts','diagnostic_subcosts_')]:
                    for field,values in rebuilt[group].items():tail[prefix_name+field]=values
                tail['scored_tick_reward'],_=reward_series(tail,arm['alpha'],runner.config)
                tail['tick_reward']=-runner.config['path_reward']['scale']*.005*sum(rebuilt['effective'].values())
            else:
                _,tail=evaluate_from_state(runner,state,path,meta['v_user'],arm['params'],arm['alpha'])
                used+=len(tail['time']);reuse=None
            if used>40000:raise ValueError('recovery tails exceed declared 40000 lower ticks')
            summary=recovery_oracle(tail,evaluator.config,start_tick=int(state.tick),initial_phi=meta['boundary_phi_rad'],route=route,alpha=arm['alpha'],prefix=prefix)
            np.savez_compressed(out/f'{route}_{name}.npz',**tail)
            record=dict(arm=name,actor_sha256=arm['actor_sha256'],snapshot_sha256=meta['receipt_sha256'],metrics=summary)
            if reuse is not None:record['physical_reused_from']=reuse
            else:cache[key]=(record,tail)
            record['reward_config_key']=config_key(runner.config.get('path_reward'))
            results[route][name]=record
            report_cases[route]['traces'][name]=tail
            report_cases[route]['configs'][name]=runner.config
            report_cases[route]['alphas'][name]=arm['alpha']
            report_cases[route]['actor_sha256'][name]=arm['actor_sha256']
    artifacts=generate_recovery_report(out,report_cases)
    receipt=dict(schema=SCHEMA,cases=results,active_tail_lower_ticks=used,artifacts=artifacts,training_steps=0,completed=True)
    atomic_json(out/'recovery.json',receipt)
    with (out/'INDEX.md').open('a') as index:
        for route,row in results.items():
            if row.get('status')=='NOT_VERIFIED':index.write(f"\n{route}: NOT_VERIFIED — {row['reason']}.\n")
    return receipt


def generate_recovery_report(output,cases):
    """Read-only plotting from actual prefix/tails; never extrapolates failures.

Each case supplies the original path and prefix, five actual tail traces, each
arm's frozen config/alpha/Actor SHA, takeover time and snapshot receipt SHA.
"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path_command_reporting import reward_series
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    index=['# Complete-state paired recovery','',
        'XY uses the immutable original path in world metres. Each trace stops at its actual end.',
        'No external perturbation is synthesized; takeover is the captured B0 upper boundary.',
        'Rewards use each arm\'s own alpha and frozen reward configuration; cross-weight returns are not physical rankings.','']
    artifacts={};names=('B0','control0','candidate0','control1','candidate1')
    for route,case in cases.items():
        if set(case['traces'])!=set(names):raise ValueError('complete five-arm recovery report required')
        path=case['path'];prefix=case['prefix'];takeover=case['takeover_s'];full={};rewards={};components={}
        for name in names:
            cfg=case['configs'][name];dt=cfg['timing']['lower_dt_s'];tail=case['traces'][name]
            _validate_trace(prefix,0,dt);_validate_trace(tail,len(prefix['time']),dt)
            full[name]={k:np.concatenate([np.asarray(prefix[k]),np.asarray(tail[k])]) for k in prefix if k in tail}
            if 'actual_xy' not in full[name] or full[name]['actual_xy'].shape!=(len(full[name]['time']),2) or not np.isfinite(full[name]['actual_xy']).all():raise ValueError('actual XY required for recovery delivery')
            rewards[name],components[name]=reward_series(full[name],case['alphas'][name],cfg)
        pathfile=out/f'{route}_path.npz'
        coordinates={k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','goal','turn_end')}
        if pathfile.exists():
            with np.load(pathfile,allow_pickle=False) as previous:
                if set(previous.files)!=set(coordinates) or any(not np.array_equal(previous[k],coordinates[k]) for k in coordinates):raise ValueError('immutable report original path differs')
        else:np.savez_compressed(pathfile,**coordinates)
        title=route+' | original 20 s horizon | local300 | natural complete-state takeover'
        colors=dict(zip(names,('black','tab:blue','tab:cyan','tab:red','tab:orange')))
        def label(name):return f"{name} a{case['alphas'][name]}"
        def finish_axis(ax,ylabel,xlabel='Absolute original time (s)'):
            ax.axvline(takeover,color='grey',ls=':',label='takeover')
            ax.set_ylabel(ylabel);ax.set_xlabel(xlabel);ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
        f,ax=plt.subplots(figsize=(9,7));progress=max(float(d['path_progress'].max()) for d in full.values())
        visible=coordinates['s']<=progress+2.
        ax.plot(*coordinates['xy'][visible].T,'k--',label='fixed original reference')
        for name in names:
            d=full[name];xy=d['actual_xy'];color=colors[name]
            ax.plot(*xy.T,color=color,label=label(name),lw=1.)
            ax.plot(*xy[len(prefix['time'])-1],marker='s',color=color,ms=4)
            failed=bool(np.any(d['physical_failure'])|np.any(d['domain_exit']))
            ax.plot(*xy[-1],marker='x' if failed else 'o',color=color,ms=6)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax.set_title(title+'\nSquare: takeover; cross: true failure; circle: observed end')
        f.tight_layout();f.savefig(out/f'{route}_xy.png',dpi=130);plt.close(f)
        f,axes=plt.subplots(3,2,figsize=(14,12))
        for name in names:
            d=full[name];t=d['time']+.005;kw=dict(color=colors[name],label=label(name),lw=.8)
            values=(d['peak_roll'],d['actual_forward_speed']-d['v_user'],d['path_cross_track'],d['path_heading_error'],rewards[name],np.cumsum(rewards[name]))
            for ax,v in zip(axes.flat,values):
                ax.plot(t,v,**kw)
                failed=bool(np.any(d['physical_failure'])|np.any(d['domain_exit']))
                ax.plot(t[-1],v[-1],marker='x' if failed else 'o',color=colors[name],ms=4)
        axes[0,0].axhline(.28,color='green',ls='--',lw=.8);axes[0,0].axhline(.302,color='red',ls='--',lw=.8)
        axes[1,0].axhline(.40,color='red',ls='--',lw=.8);axes[1,0].axhline(-.40,color='red',ls='--',lw=.8)
        for ax,y in zip(axes.flat,('Peak roll (rad)','Speed error (m/s)','Signed path error (m)','Heading error (rad)','Signed reward per actual lower tick','Cumulative signed reward')):finish_axis(ax,y)
        f.suptitle(title);f.tight_layout();f.savefig(out/f'{route}_diagnostics.png',dpi=130);plt.close(f)
        # Components are signed rewards, including the single terminal tail.
        f,axes=plt.subplots(len(names),2,figsize=(15,19))
        for row,name in enumerate(names):
            t=full[name]['time']+.005
            for component,value in components[name].items():
                axes[row,0].plot(t,value,label=component,lw=.65)
                axes[row,1].plot(t,np.cumsum(value),label=component,lw=.65)
            for col in (0,1):
                finish_axis(axes[row,col],'Signed per-tick components' if col==0 else 'Cumulative signed components')
                axes[row,col].set_title(label(name));axes[row,col].legend(fontsize=5,ncol=4)
        f.suptitle(title);f.tight_layout();f.savefig(out/f'{route}_reward_components.png',dpi=130);plt.close(f)
        artifacts[route]=dict(xy=f'{route}_xy.png',diagnostics=f'{route}_diagnostics.png',components=f'{route}_reward_components.png',path=pathfile.name)
        index.extend([f'## {route}','',f"![Actual / original XY]({route}_xy.png)",'',
            f"[Roll/speed/path/reward]({route}_diagnostics.png) · [Signed cost components]({route}_reward_components.png) · [Original path NPZ]({pathfile.name})",'',
            f"Snapshot `{case['snapshot_sha256']}`; takeover {takeover:.3f} s; dt .005 s. Actual prefixes remain in the snapshot leaves archive; tail NPZ files contain active transitions only.",'',
            '| Arm | Upper alpha | Selected Actor SHA | Observed end (s) |', '|---|---:|---|---:|'])
        for name in names:index.append(f"| {name} | {case['alphas'][name]} | {case['actor_sha256'][name]} | {float(full[name]['time'][-1])+.005:.3f} |")
        index.append('')
    (out/'INDEX.md').write_text('\n'.join(index)+'\n')
    return artifacts
