"""Static-reference strong comparator, not a dynamic safety certificate."""
import json
from pathlib import Path


def target(nominal, alpha, config, q_of_speed, budget_rad, grid_points=65):
    """User0 slows at fixed steer; user1 compresses steer at fixed speed.

    Native delta/q(v) is the controller's equilibrium lean reference. Finite
    grid feasibility applies only to this static pair, before real filtering.
    """
    import jax.numpy as j
    from jax import vmap
    c=config['residual_interface'];eps=1e-6
    nominal=j.asarray(nominal);v,delta=nominal[0],nominal[1]
    low=j.maximum(c['speed_reference_bounds_m_s'][0],v-c['speed_negative_scale_m_s']*(1-eps))
    low=j.minimum(low,v);candidates=j.linspace(low,v,grid_points)
    ratio=vmap(q_of_speed)(candidates)
    feasible=j.isfinite(ratio)&(j.abs(ratio)>1e-8)&(j.abs(delta/ratio)<=budget_rad)
    speed=j.max(j.where(feasible,candidates,low))
    q=q_of_speed(v);steering=j.clip(delta,-budget_rad*j.abs(q),budget_rad*j.abs(q))
    desired=j.stack([(1-alpha)*speed+alpha*v,(1-alpha)*delta+alpha*steering])
    correction=desired-nominal
    scale=j.array([j.where(correction[0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),c['steer_scale_rad']])
    latent=j.arctanh(j.clip(correction/scale,-1+eps,1-eps))
    realized=nominal+j.tanh(latent)*scale
    return latent,dict(desired=desired,realized_target=realized,grid_has_feasible=j.any(feasible),
        latent_permission_limited=j.any(j.abs(correction/scale)>1-eps),
        static_lean_before=delta/q,static_lean_target=realized[1]/q_of_speed(realized[0]))


def prepare(zero_source, root, base):
    zero_source=Path(zero_source).resolve();root=Path(root).resolve()
    zp=json.loads((zero_source/'plan.json').read_text());zs=json.loads((zero_source/'evaluation/status.json').read_text())
    if zs['phase']!='completed' or zs['completed_cases']!=8:raise ValueError('completed native zero8 required')
    if base.digest(zero_source/'plan.json')!=json.loads((zero_source/'seal.json').read_text())['plan_sha256']:raise ValueError('zero declaration changed')
    for f,sha in {**zp['input_sha256'],**zp['native_source_sha256']}.items():
        if base.digest(f)!=sha:raise ValueError('zero scientific input changed '+f)
    changes={}
    for f,sha in zp['runner_sha256'].items():
        if base.digest(f)==sha:continue
        candidates=[zero_source/'source'/Path(f).name,zero_source/'engineering_revisions/after_persistence_fix'/Path(f).name]
        original=next((p for p in candidates if p.exists() and base.digest(p)==sha),None)
        if original is None:raise ValueError('actual old numerical runner snapshot missing')
        changes[f]=dict(old_sha256=sha,new_sha256=base.digest(f),old_source=str(original),reason='Explicit new rule branch; literal-zero default and archived native controller preserved')
    records=json.loads((zero_source/'analysis/review.json').read_text())['records']
    if len(records)!=24:raise ValueError('all24 native controls required')
    for r in records:
        if base.digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('cached raw changed')
    base.prepare(Path(zp['source']),root);p=json.loads((root/'plan.json').read_text())
    own=Path(__file__).resolve();(root/'source'/own.name).write_bytes(own.read_bytes())
    base.save(root/'cached_records.json',records)
    p['input_sha256'][str(root/'cached_records.json')]=base.digest(root/'cached_records.json')
    for name in ('plan.json','seal.json','analysis/review.json','evaluation/status.json'):
        f=zero_source/name;p['input_sha256'][str(f)]=base.digest(f)
    p['runner_sha256'][str(own)]=base.digest(own)
    p.update(experiment_id='native_equilibrium_rule_v1',target_policy='equilibrium',cached_cases=24,
        cached_zero_source=str(zero_source),baseline_source_changes=changes,
        rule=dict(source=str(own),budget_rad=p['config']['path_reward']['working_roll_start_rad'],grid_points=65,
            budget_origin='Original V3 working-roll cost start; not a physical failure threshold',
            alpha_semantics='0: slow at fixed nominal steer; 1: compress steer at fixed nominal speed; only endpoints tested',
            target_semantics='Finite-grid static pair; no dynamic assurance, original filters/permissions can lag or restrict it'),
        question='Can a simple static-reference rule explain learned preference and working-roll benefits?',
        initialization='Same native complete prepared bank/reset; no NN target inference or training; ordinary static rule',
        stop='Exactly8 new nominal rule traces, unchanged native10s/physical/domain/fault; no retry/grid/budget tuning or adoption')
    for c in p['cases']:c['policy']='equilibrium'
    base.save(root/'plan.json',p);base.save(root/'seal.json',dict(plan_sha256=base.digest(root/'plan.json')))
