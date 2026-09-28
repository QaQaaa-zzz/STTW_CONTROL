"""Independent NumPy audit of frozen V3 review traces and Actor sensitivity.

This module performs no physics rollout and never changes a checkpoint. Call
``audit_run(output, spec)`` only after review traces and pilot checkpoint exist.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np


COMPONENTS = ('speed', 'steer', 'heading', 'roll', 'roll_rate', 'overspeed',
              'low_speed', 'correction', 'command_change')
CASES = ('main', 'random')
METHODS = ('B0', 'pi_alpha0', 'pi_alpha1')


def _h(z):
    a = np.abs(z)
    return np.where(a <= 1., z*z, 2.*a-1.)


def _ratio(speed, controller):
    """Independent scalar transcription of ECBC's m4/m2, including floor."""
    c = controller
    v = np.maximum(c['minimum_speed'], speed)
    m2 = -(c['mass']*v*v*c['cg_height'] -
           c['mass']*c['cg_forward']*c['trail']*c['gravity'])*np.cos(c['caster'])/c['wheelbase']
    m2 = np.where(m2 < 0., -1., 1.)*np.maximum(np.abs(m2), .1)
    m4 = -c['mass']*c['gravity']*c['cg_height']
    return m4/m2


def _require(trace, name, shape):
    if name not in trace:
        raise ValueError(f'missing saved trace field {name}')
    value = np.asarray(trace[name], dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f'{name} shape {value.shape}, expected {shape}')
    return value


def reconstruct_trace(trace, spec, first_previous_final, controller):
    """Rebuild all costs, raw gate timing, reference yaw and scored rewards.

    ``trace`` is one method's active 5 ms NPZ mapping. Returns arrays and
    maximum absolute errors against saved values. No JAX reward code is used.
    """
    time = np.asarray(trace['time'], dtype=np.float64)
    n = len(time); dt = spec['plant']['control_dt_s']; r = spec['reward']
    if n == 0 or not np.allclose(time, np.arange(n)*dt, atol=dt*1e-3, rtol=0):
        raise ValueError('active trace must be a contiguous 5 ms prefix')
    raw = _require(trace, 'limited_command', (n, 2))
    rates = _require(trace, 'raw_rates', (n, 2))
    final = _require(trace, 'final_command', (n, 2))
    offsets = _require(trace, 'offsets', (n, 2))
    speed = _require(trace, 'actual_forward_speed', (n,))
    steer = _require(trace, 'actual_delta', (n,))
    heading = _require(trace, 'e_psi_unwrapped', (n,))
    roll = _require(trace, 'phi', (n,))
    roll_rate = _require(trace, 'phi_dot', (n,))
    alpha = _require(trace, 'alpha', (n,))
    if not np.all(np.isin(alpha, (0., 1.))):
        raise ValueError('alpha outside declared endpoints')
    previous = np.concatenate((np.asarray(first_previous_final, dtype=np.float64)[None], final[:-1]), axis=0)
    if previous.shape != (n, 2):
        raise ValueError('prepared actuator previous command must contain two values')
    phi_raw = raw[:, 1]/_ratio(raw[:, 0], controller)
    chi = np.clip((np.abs(phi_raw)-r['conflict_phi_start_rad'])/
                  (r['conflict_phi_full_rad']-r['conflict_phi_start_rad']), 0., 1.)
    eligible = ((np.abs(raw[:, 1]) <= r['recovery_raw_steer_abs_max_rad']) &
                (np.abs(phi_raw) <= r['recovery_raw_phi_abs_max_rad']) &
                (np.abs(rates[:, 0]) <= r['recovery_raw_speed_rate_abs_max_m_s2']) &
                (np.abs(rates[:, 1]) <= r['recovery_raw_steer_rate_abs_max_rad_s']))
    clock = np.empty(n); gate = np.empty(n); old = 0.
    for i in range(n):
        clock[i] = old
        gate[i] = np.clip((old-r['recovery_settle_s'])/r['recovery_ramp_s'], 0., 1.) if eligible[i] else 0.
        old = min(r['recovery_settle_s']+r['recovery_ramp_s'], old+dt) if eligible[i] else 0.
    ev = speed-raw[:, 0]; ed = steer-raw[:, 1]
    gap = r['priority_high']-r['priority_low']
    wv = r['priority_high']-gap*chi*(1.-alpha)
    wd = (1.-gate)*(r['priority_high']-gap*chi*alpha)+gate*r['recovery_steer_weight']
    parts = {
        'speed': wv*_h(ev/r['speed_error_scale_m_s']),
        'steer': wd*_h(ed/r['steer_error_scale_rad']),
        'heading': r['heading_weight']*gate*_h(np.maximum(np.abs(heading)-r['heading_deadband_rad'],0.)/r['heading_error_scale_rad']),
        'roll': r['roll_weight']*_h(np.maximum(np.abs(roll)-r['roll_soft_start_rad'],0.)/r['roll_error_scale_rad']),
        'roll_rate': r['roll_rate_weight']*_h(np.maximum(np.abs(roll_rate)-r['roll_rate_soft_start_rad_s'],0.)/r['roll_rate_error_scale_rad_s']),
        'overspeed': r['overspeed_weight']*_h(np.maximum(ev-r['overspeed_free_band_m_s'],0.)/r['overspeed_scale_m_s']),
        'low_speed': r['low_speed_weight']*_h(np.maximum(r['low_speed_start_m_s']-speed,0.)/r['low_speed_scale_m_s']),
        'correction': r['reference_correction_weight']*np.sum((offsets/np.asarray(r['correction_normalizers']))**2,axis=1),
        'command_change': r['final_command_change_weight']*np.sum(((final-previous)/np.asarray(r['final_command_normalizers']))**2,axis=1),
    }
    raw_cost = sum(parts.values())
    effective_cost = np.minimum(raw_cost,r['cost_rate_cap'])
    factor = np.minimum(1.,r['cost_rate_cap']/np.maximum(raw_cost,1e-12))
    effective = {name:value*factor for name,value in parts.items()}
    scored_effective = {name:value.copy() for name,value in effective.items()}
    cap_fraction = (raw_cost > r['cost_rate_cap']).astype(float)
    ordinary = -r['scale']*dt*effective_cost
    scored = ordinary.copy(); failure_cost = np.zeros(n)
    failed = np.asarray(trace.get('physical_failure',np.zeros(n)),dtype=bool)
    if failed.shape != (n,):raise ValueError('physical_failure shape mismatch')
    fail_indices = np.flatnonzero(failed)
    if len(fail_indices):
        if len(fail_indices) != 1 or fail_indices[0] != n-1:
            raise ValueError('physical failure must occur once at active trace endpoint')
        index = int(fail_indices[0]); policy_index=index//spec['plant']['control_ticks_per_action']
        N=round(spec['commands']['episode_seconds']/spec['plant']['policy_dt_s'])-policy_index
        gamma=spec['ppo']['gamma']
        geometric = -math.expm1(N*math.log(gamma))/(1.-gamma)
        tail = -r['failure_extra_penalty']-r['scale']*spec['plant']['policy_dt_s']*r['cost_rate_cap']*geometric
        scored[policy_index*spec['plant']['control_ticks_per_action']:index+1]=0.
        scored[index]=tail; failure_cost[index]=-tail
        for value in scored_effective.values():
            value[policy_index*spec['plant']['control_ticks_per_action']:index+1]=0.
    # Reference yaw is integrated independently from the current raw command.
    yaw_saved = _require(trace,'reference_yaw_unwrapped',(n,))
    actual_yaw = _require(trace,'yaw_unwrapped',(n,))
    theta = raw[:,0]*np.cos(controller['caster'])*np.tan(raw[:,1])/controller['wheelbase']*dt
    yaw_rebuilt = yaw_saved[0]-theta[0]+np.cumsum(theta)
    checks = {'chi':chi,'g':gate,'settle_clock':clock,'raw_cost':raw_cost,
              'effective_cost':effective_cost,'cap_fraction':cap_fraction,
              'scored_tick_reward':scored,'failure_cost':failure_cost,
              'reference_yaw_unwrapped':yaw_rebuilt,
              'e_psi_unwrapped':yaw_rebuilt-actual_yaw}
    checks.update({f'raw_cost_{name}':value for name,value in parts.items()})
    checks.update({f'effective_cost_{name}':value for name,value in effective.items()})
    checks.update({f'scored_cost_{name}':value for name,value in scored_effective.items()})
    errors = {}; matching = {}
    for name,value in checks.items():
        recorded = _require(trace,name,(n,))
        if not np.isfinite(recorded).all() or not np.isfinite(value).all():
            raise ValueError(f'nonfinite audit field {name}')
        errors[name] = float(np.max(np.abs(value-recorded)))
        matching[name] = bool(np.allclose(value,recorded,rtol=1e-3,atol=2e-3))
    return dict(count=n,physical_failure=bool(len(fail_indices)),errors=errors,
                matching=matching,passed=all(matching.values()),
                max_abs_error=max(errors.values()),reward_sum=float(scored.sum()),
                raw_cost_integral=float(raw_cost.sum()*dt),
                effective_cost_integral=float(effective_cost.sum()*dt),
                cap_fraction=float(cap_fraction.mean()),
                ordinary_stable_cap_warning=bool(np.mean(cap_fraction[(chi==0)&(gate==0)])>.05)
                if np.any((chi==0)&(gate==0)) else False)


def _prepared_previous(path):
    with path.open('rb') as file:
        bank = pickle.load(file)
    value = np.asarray(bank.actuator.previous)
    if value.shape != (8,2):raise ValueError('prepared bank must have eight actuator states')
    return value[3]


def audit_sensitivity(output, spec, checkpoint, checkpoint_update=None):
    """Evaluate saved active observations under both alphas, when available."""
    import jax
    import jax.numpy as jp
    import torch
    from tensordict import TensorDict
    from .direct_command_policy import DirectCommandActor, export_actor
    from .direct_command_ppo import make_algorithm

    paired_rows=[];source_cases=[];source_methods=[]
    for case_index,case in enumerate(CASES):
        path=output/'review'/case/'actual_observations.npz'
        if not path.is_file():continue
        with np.load(path,allow_pickle=False) as data:
            obs=np.asarray(data['observations'],dtype=np.float32)
        if obs.ndim!=3 or obs.shape[1:]!=(3,345):raise ValueError(f'{case}: observation shape mismatch')
        for method_index,method in ((1,'pi_alpha0'),(2,'pi_alpha1')):
            trace_path=output/'review'/case/(method+'.npz')
            if not trace_path.is_file():continue
            with np.load(trace_path,allow_pickle=False) as trace:
                if checkpoint_update is not None and int(trace['checkpoint_update'])!=checkpoint_update:
                    continue
                ticks=len(trace['time'])
            policy_steps=math.ceil(ticks/spec['plant']['control_ticks_per_action'])
            selected=obs[:policy_steps,method_index,:]
            finite=np.isfinite(selected).all(axis=1)
            selected=selected[finite]
            if len(selected):
                paired_rows.append(selected)
                source_cases.append(np.full(len(selected),case_index,dtype=np.int32))
                source_methods.append(np.full(len(selected),method_index,dtype=np.int32))
    if not paired_rows:
        summary=dict(available=False,reason='no finite saved policy observations with matching active traces',
                     checkpoint=str(checkpoint),observation_pairs=0)
        (output/'review'/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
    paired=np.concatenate(paired_rows,axis=0)
    alternate=np.repeat(paired[:,None,:],2,axis=1)
    alternate[:,0,336]=0.;alternate[:,1,336]=1.
    flat=alternate.reshape(-1,345)
    td=TensorDict({'policy':torch.zeros((4,345)),
                   'critic':torch.zeros((4,346))},batch_size=[4])
    algo=make_algorithm(td,1,spec,'cpu')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('schema')!='sttw_direct_command_v3_345_346_2':raise ValueError('checkpoint schema mismatch')
    if saved.get('config')!=spec:raise ValueError('checkpoint config mismatch')
    algo.policy.load_state_dict(saved['policy']);algo.policy.eval()
    with torch.no_grad():
        mu_torch=np.concatenate([algo.policy.act_inference(TensorDict(
            {'policy':torch.from_numpy(chunk)},batch_size=[len(chunk)])).numpy()
            for chunk in np.array_split(flat,max(1,math.ceil(len(flat)/16384)))],axis=0)
    params=export_actor(algo.policy)
    actor=DirectCommandActor()
    mu_flax=np.concatenate([np.asarray(actor.apply(params,jp.asarray(chunk)))
            for chunk in np.array_split(flat,max(1,math.ceil(len(flat)/16384)))],axis=0)
    parity=np.max(np.abs(mu_torch-mu_flax))
    means=mu_torch.reshape(len(paired),2,2)
    delta=means[:,1]-means[:,0]
    np.savez_compressed(output/'review'/'sensitivity.npz',
        source_case=np.concatenate(source_cases),
        source_method=np.concatenate(source_methods),
        paired_observation=paired,mu_alpha0=means[:,0],mu_alpha1=means[:,1],
        delta_alpha1_minus_alpha0=delta)
    summary=dict(available=True,checkpoint=str(checkpoint),checkpoint_update=int(saved['update']),
        observation_pairs=int(len(paired)),torch_flax_mu_max_abs=float(parity),
        torch_flax_mu_parity_passed=bool(np.isfinite(parity) and parity<=1e-4),
        latent_mean_delta_mean=np.mean(delta,axis=0).tolist(),
        latent_mean_delta_abs_mean=np.mean(np.abs(delta),axis=0).tolist(),
        latent_mean_delta_abs_max=np.max(np.abs(delta),axis=0).tolist(),
        note='Counterfactual alpha replacement holds each saved physical observation fixed; it does not establish closed-loop preference or task success.')
    (output/'review'/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def _resolve_checkpoint(output, selected):
    path=Path(selected['checkpoint'])
    candidates=[path] if path.is_absolute() else [path,output/path,output/'pilot'/'checkpoints'/path.name]
    for candidate in candidates:
        if candidate.is_file():return candidate.resolve()
    raise FileNotFoundError(f'last completed pilot checkpoint missing: {path}')


def audit_run(output, spec):
    """Audit available review prefixes, explicitly recording missing evidence."""
    output=Path(output);review=output/'review';review.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((output/'manifest.json').read_text())
    controller=manifest['controller']
    previous=_prepared_previous(output/'prepared_bank.pkl')
    selected=json.loads((output/'pilot'/'last_completed.json').read_text())
    checkpoint=_resolve_checkpoint(output,selected)
    report=dict(checkpoint=str(checkpoint),checkpoint_update=int(selected['update']),
                scope='saved review prefixes only; no new physics',traces={},limitations=[])
    for case in CASES:
        report['traces'][case]={}
        for method in METHODS:
            path=review/case/(method+'.npz')
            if not path.is_file():
                report['traces'][case][method]=dict(state='missing',reason='review trace file absent',path=str(path))
                continue
            with np.load(path,allow_pickle=False) as data:
                trace={key:data[key] for key in data.files}
            if int(trace['checkpoint_update'])!=selected['update']:
                report['traces'][case][method]=dict(state='invalid',reason='checkpoint identity mismatch',
                    trace_update=int(trace['checkpoint_update']))
                continue
            try:
                result=reconstruct_trace(trace,spec,previous,controller)
                partial=bool(trace['partial'])
                result['partial']=partial
                result['state']='partial' if partial else ('physical_failure' if result['physical_failure'] else 'complete')
                report['traces'][case][method]=result
            except (ValueError,KeyError,TypeError) as exc:
                report['traces'][case][method]=dict(state='invalid',reason=str(exc))
    try:
        report['sensitivity']=audit_sensitivity(output,spec,checkpoint,selected['update'])
    except Exception as exc:
        report['sensitivity']=dict(available=False,reason=str(exc),error_type=type(exc).__name__)
        (review/'sensitivity_summary.json').write_text(json.dumps(report['sensitivity'],indent=2)+'\n')
    report['limitations'].append('Initial absolute reference yaw is anchored to the first saved reference yaw; all subsequent yaw increments are rebuilt from raw commands.')
    report['limitations'].append('Observation alpha substitution is a fixed-state Actor test, not a paired physical rollout.')
    rows=[row for case in report['traces'].values() for row in case.values()]
    checked=[row for row in rows if 'passed' in row]
    report['coverage']=dict(expected=len(CASES)*len(METHODS),checked=len(checked),
        complete=sum(row['state']=='complete' for row in rows),
        physical_failure=sum(row['state']=='physical_failure' for row in rows),
        partial=sum(row['state']=='partial' for row in rows),
        missing=sum(row['state']=='missing' for row in rows),
        invalid=sum(row['state']=='invalid' for row in rows))
    report['max_abs_trace_error']=max((row['max_abs_error'] for row in checked),default=None)
    report['all_traces_passed']=len(checked)==len(rows) and all(row['passed'] for row in checked)
    (review/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
