"""Offline privileged-time factorial, not a deployable safety switch."""
import numpy as np
import jax
import jax.numpy as j

MODES=('AA','AB','BA','BB')


def actor_b_at_tick(mode,tick,handoff_tick):
    return (mode==3)|((mode==1)&(tick>=handoff_tick))|((mode==2)&(tick<handoff_tick))


def factorial_inventory(world_direction):
    direction=np.asarray(world_direction,float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction),1.):
        raise ValueError('finite unit world force direction required')
    cases=[]
    for alpha,early in [(.25,388),(.5,380)]:
        events=[(0.,early),(-32.,early),(32.,early),(-32.,580),(32.,580)]
        for force,onset in events:
            event='zero' if not force else f't{onset}_F{force:+g}'
            for index,mode in enumerate(MODES):
                cases.append(dict(case_id=f'a{alpha:g}/{event}/{mode}',mode=mode,mode_index=index,
                    event_id=event,alpha=alpha,state_alpha=alpha,context='right',route='right90_R1p1',
                    requested_speed_m_s=1.5,force_N=force,pulse_onset_task_tick=onset,
                    handoff_tick=onset,pulse_duration_s=.1,world_direction=direction.tolist(),
                    profile=0,source_batch_slot=0,
                    cache_candidate=mode in ['AA','BB'] and (force==0 or (force==-32 and onset==early))))
    return cases


def make_factorial_step(env,actor,*,horizon_s):
    from .anticipatory_reference import make_reference_step
    advance=make_reference_step(env,actor,horizon_s=horizon_s,allow_reentry=True)
    lower_bound=j.float32(env.config['residual_interface']['speed_reference_bounds_m_s'][0])
    def step(carry,parameters,path,speed,pulse,mode,handoff_tick):
        before=carry.state.tick
        use_b=actor_b_at_tick(mode,before,handoff_tick)
        # One Boolean selects every leaf of one whole frozen network. This is
        # not weight interpolation and never splices controller/physical state.
        selected=jax.tree.map(lambda a,b:j.where(use_b,b,a),parameters['A'],parameters['B'])
        end,logs=advance(carry,selected,path,speed,pulse,j.int32(0),lower_bound,j.bool_(False))
        logs=dict(logs,desired_B=logs['active_tick']&use_b,
            selected_B=logs['active_tick']&logs['actor_action_valid']&use_b,
            decision_task_tick=j.repeat(before,4))
        return end,logs
    return step


def factorial_contrasts(rows):
    if not rows or len({x['case_id'] for x in rows})!=len(rows):raise ValueError('unique nonempty case rows required')
    groups={}
    for r in rows:
        group=groups.setdefault((r['alpha'],r['event_id']),{})
        if r['mode'] in group or r['mode'] not in MODES:raise ValueError('one case per factorial cell required')
        group[r['mode']]=r
    if any(set(g)!=set(MODES) for g in groups.values()):raise ValueError('complete2x2 required')
    result={};safe=lambda r:r['category'] in ['safe_task','safe_task_failed']
    for target,base in [('AB','AA'),('BA','AA'),('BB','BA'),('BB','AB')]:
        counts=dict(pairs=len(groups),known=0,unknown=0,physical_gains=0,physical_losses=0,task_gains=0,task_losses=0,details=[])
        for key,g in groups.items():
            a,b=g[target],g[base];unknown='unknown' in (a['category'],b['category'])
            p=None if unknown else int(safe(a))-int(safe(b));t=None if unknown else int(a['category']=='safe_task')-int(b['category']=='safe_task')
            counts['unknown' if unknown else 'known']+=1
            if not unknown:
                counts['physical_gains']+=int(p==1);counts['physical_losses']+=int(p==-1)
                counts['task_gains']+=int(t==1);counts['task_losses']+=int(t==-1)
            counts['details'].append(dict(alpha=key[0],event_id=key[1],target_case=a['case_id'],baseline_case=b['case_id'],
                physical_change=p,task_change=t,scored_cost_change=a['scored_cost']-b['scored_cost']))
        result[target+'_vs_'+base]=counts
    return result
