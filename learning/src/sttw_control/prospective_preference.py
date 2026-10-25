"""Declared paired timing/alpha cases; no physics, training or policy selection."""
import math
import numpy as np

RECIPES={
    'task_only':('task_only',0,1.),
    'task_floor':('task_only',0,1.5),
    'task_early':('task_only',6,1.),
    'task_constant':('task_only',7,1.),
    'parent_early':('parent',6,1.),
    'parent_constant':('parent',7,1.),
}


def validate_common_origin_cases(rows,source_cases):
    """Check declared recipes and new user contexts against the real source task."""
    if not rows or len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('nonempty unique declared case IDs required')
    source=source_cases[0]
    for r in rows:
        if any(r[k]!=source[k] for k in ['route','requested_speed_m_s','horizon_s']):
            raise ValueError('source task geometry, request and horizon must be retained')
        if not np.allclose(r['world_direction'],source['world_direction'],rtol=0,atol=1e-8):
            raise ValueError('original world force direction required')
        role=r['role'];expected=RECIPES.get(role,('parent',0,1.) if role=='zero_middle' else None)
        if expected is None or tuple(r[k] for k in ['actor_role','profile','reference_speed_floor_m_s'])!=expected:
            raise ValueError('declared Actor/profile/reference permissions differ')
        if r['source_batch_slot']!=0 or r['source_state_alpha']!=0 or r['stage']!='reentry':
            raise ValueError('one complete task0 source and shared handback required')
        alpha=r['alpha']
        if role=='zero_middle':
            valid=alpha is None and r['state_alpha']==0 and r['zero_mode']
        else:
            valid=(alpha is not None and math.isfinite(alpha) and 0<=alpha<=1
                   and r['state_alpha']==alpha and not r['zero_mode'])
        if not valid:raise ValueError('user-alpha context or actual zero-mode identity differs')


def prospective_cases(route,*,speed,world_direction,alphas,onsets,roles=None,
                      forces=(-32.,32.),duration_s=.1,horizon_s=10.):
    """One shared complete origin per route; user-alpha is assigned at task0.

    The zero controller has one actual alpha0 execution per event, not five
    purportedly independent executions. Its fixed trajectory can be rescored.
    """
    alphas=tuple(float(a) for a in alphas);onsets=tuple(onsets)
    roles=tuple(RECIPES) if roles is None else tuple(roles);forces=tuple(float(f) for f in forces)
    if (not alphas or len(set(alphas))!=len(alphas) or
            not all(math.isfinite(a) and 0<=a<=1 for a in alphas)):
        raise ValueError('unique finite user-alpha values within[0,1] required')
    if not roles or len(set(roles))!=len(roles) or not set(roles)<=set(RECIPES):
        raise ValueError('unique declared reference recipes required')
    if not math.isfinite(speed) or speed<=0:raise ValueError('positive request speed required')
    d=np.asarray(world_direction,float)
    if d.shape!=(3,) or not np.isfinite(d).all() or not np.isclose(np.linalg.norm(d),1.):
        raise ValueError('unit world force direction required')
    if not (math.isfinite(duration_s) and math.isfinite(horizon_s) and 0<duration_s<=horizon_s):
        raise ValueError('positive finite pulse and horizon required')
    if not forces or len(set(forces))!=len(forces) or not all(math.isfinite(f) and f!=0 for f in forces):
        raise ValueError('unique finite nonzero force queries required')
    limit=round(horizon_s/.005);duration=round(duration_s/.005)
    if not math.isclose(limit*.005,horizon_s,abs_tol=1e-9) or not math.isclose(duration*.005,duration_s,abs_tol=1e-9):
        raise ValueError('times must align to original5ms clock')
    if not onsets or len(set(onsets))!=len(onsets) or any(not isinstance(t,int) or t<0 or t%4 or t+duration>limit for t in onsets):
        raise ValueError('unique complete upper-boundary onsets within horizon required')
    events=[('zero',0.,onsets[0])]+[(f't{t}_F{f:+g}',f,t) for t in onsets for f in forces]
    rows=[]
    for role in roles+('zero_middle',):
        actor,profile,floor=RECIPES[role] if role!='zero_middle' else ('parent',0,1.)
        for alpha in alphas if role!='zero_middle' else (None,):
            for event,force,onset in events:
                rows.append(dict(case_id=f'{role}/alpha{alpha}/{event}',role=role,actor_role=actor,
                    alpha=alpha,state_alpha=0. if alpha is None else alpha,source_state_alpha=0.,source_batch_slot=0,
                    profile=profile,reference_speed_floor_m_s=floor,zero_mode=role=='zero_middle',
                    event_id=event,force_N=force,pulse_onset_task_tick=onset,pulse_duration_s=duration_s,
                    world_direction=d.tolist(),route=route,requested_speed_m_s=speed,horizon_s=horizon_s,stage='reentry'))
    return rows
