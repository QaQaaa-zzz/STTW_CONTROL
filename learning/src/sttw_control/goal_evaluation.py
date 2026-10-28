"""Fixed ordinary-goal endpoint evaluation; no optimizer or candidate selection."""
import math
import numpy as np
import jax.numpy as j
from flax import struct
from .goal_live_collection import original_goal_context,goal_latch_from_block,_SampledAction
from .reference_reentry import actor_permission


@struct.dataclass
class GoalEvaluationState:
    state: object
    gate: object
    clean_blocks: object
    arrived: object


def evaluation_cases():
    """128 distinct physical executions, 192 user-alpha score views, zero deduped."""
    cases=[]
    for suite,radius,speeds,onsets in [('transfer',1.25,dict(left=1.7,right=1.6),(360,640)),
                                      ('training_geometry',1.1,dict(left=1.8,right=1.5),(500,))]:
        events=[(0.,onsets[0])]+[(force,t) for force in (-32.,32.) for t in onsets]
        for context in ('left','right'):
            for force,onset in events:
                for role,name in enumerate(('fixed_endpoint64','parent_task64','zero','zero_plus_profile7_speed_support')):
                    for alpha in ((0.,.5,1.) if role<2 else (0.,)):
                        key=f'{suite}/{context}/F{force:+g}/t{onset}/{name}/a{alpha:g}'
                        cases.append(dict(case_id=key,suite=suite,context=context,role=role,role_name=name,
                            controller_alpha=alpha,user_alphas=[alpha] if role<2 else [0.,.5,1.],
                            path_radius_m=radius,speed=speeds[context],force_N=force,
                            pulse_onset_task_tick=onset,pulse_duration_s=.1))
    return cases


def make_goal_evaluation_step(env,evaluate,*,horizon_s=10.,radius_m=1.,distance_scale_m=10.):
    """Deterministic mean through the same reference/reentry physical kernel.

    Roles0/1: learned/parent mean, profile7. Role2: explicit zero_mode and no
    support. Role3: zero NN mean plus existing profile7, same eligibility. No
    independent-support override. Pulse is supplied to plant only, not evaluate.
    """
    from .anticipatory_reference import make_reference_step
    limit=round(horizon_s/float(env.cc.dt))
    if not all(math.isfinite(x) and x>0 for x in (horizon_s,radius_m,distance_scale_m)):
        raise ValueError('positive explicit goal evaluation contract required')
    advance=make_reference_step(env,_SampledAction(),horizon_s=horizon_s,allow_reentry=True)
    lower=j.float32(env.config['residual_interface']['speed_reference_bounds_m_s'][0])
    def step(carry,params,path,speed,pulse,role,goal):
        s=carry.state;zero=role==2;neural=role<2
        stopped=s.physical.failed|s.fault|(s.tick>=limit)
        eligible=actor_permission(s.domain_exit,s.follower.invalid,carry.clean_blocks,zero,stopped)
        obs,critic,obs_fault=env.observation(s,path,speed,horizon_s)
        _,pose,_=env.physics.observe(s.physical.data)
        context=original_goal_context(pose,goal,tick=s.tick,dt=env.cc.dt,horizon_s=horizon_s,
            arrived=carry.arrived,radius_m=radius_m,distance_scale_m=distance_scale_m)
        protected=[j.where(eligible&neural,x,j.zeros_like(x)) for x in (obs,critic,context)]
        mean=evaluate(params,*protected)[0]
        action=j.where(neural,mean,j.zeros(2))
        fault=(eligible & (obs_fault | (neural & (~j.all(j.isfinite(action))|~j.all(j.isfinite(context))))))
        fault=fault|(~stopped&(~j.all(j.isfinite(goal))|(role<0)|(role>3)))
        prepared=carry.replace(state=s.replace(fault=s.fault|fault))
        profile=j.where(zero,0,7)
        end,logs=advance(prepared,dict(latent=action,support_independent_permission=False),
            path,speed,pulse,profile,lower,zero)
        end=end.replace(arrived=goal_latch_from_block(carry.arrived,logs,goal,radius_m))
        logs=dict(logs,neural_active=logs['actor_action_valid']&neural,
            decision_pose=j.tile(pose,(4,1)),
            parent_observation=j.tile(protected[0],(4,1)),
            goal_context=j.tile(context,(4,1)),raw_mean=j.tile(j.where(neural,mean,j.zeros(2)),(4,1)))
        return end,logs
    return step


def rescore_original_components(physical,*,user_alpha,config,recorded_alpha=None):
    """Reweight only V3's three alpha-dependent channels; retain other costs.

    This allows shared zero/rule physics to be scored under original user alpha.
    No trajectory, goal, reference or controller_alpha is replaced. Do not count
    these score aliases as independent physical executions.
    """
    from .path_command_reward import configured_cost_keys
    c=config['path_reward'];keys=configured_cost_keys(config)
    if c.get('reward_version')!='path_persistent_preference_v3' or not 0<=user_alpha<=1:
        raise ValueError('explicit V3 and legal user alpha required')
    active=np.asarray(physical['active_tick'],bool)
    source=np.asarray(physical['effective_costs'],float)
    if source.shape!=(len(active),len(keys)):raise ValueError('configured effective channels required')
    def huber(x):
        x=np.abs(x);return np.where(x<=1,x*x,2*x-1)
    chi=np.asarray(physical['chi'],float)
    ev=np.asarray(physical['actual_forward_speed'],float)-np.asarray(physical['requested_speed'],float)
    ey=np.asarray(physical['path_cross_track'],float);eh=np.asarray(physical['path_heading_error'],float)
    v=huber(ev/c['speed_scale_m_s'])
    path=10*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad']))
    p0=c['primary_weight']*chi*huber(np.maximum(np.abs(ey)-c['primary_path_band_m'],0)/c['primary_path_scale_m'])
    p1=c['primary_weight']*chi*huber(np.maximum(np.abs(ev)-c['primary_speed_band_m_s'],0)/c['primary_speed_scale_m_s'])
    cap=c['component_caps']
    def channels(a):
        return dict(speed=(1-a)*np.minimum((10-9*chi)*v,cap['speed'])+a*np.minimum(10*v,cap['speed']),
            path=(1-(1-c['path_secondary_ratio'])*a)*np.minimum(path,cap['path']),
            primary=(1-a)*np.minimum(p0,cap['primary'])+a*np.minimum(p1,cap['primary']))
    if recorded_alpha is not None:
        for name,value in channels(recorded_alpha).items():
            if not np.allclose(source[active,keys.index(name)],value[active],rtol=3e-5,atol=2e-5):
                raise ValueError('original recorded alpha components disagree: '+name)
    out=source.copy()
    for name,value in channels(user_alpha).items():out[:,keys.index(name)]=value
    valid=active & np.asarray(physical['task_reward_valid'],bool)
    if not np.isfinite(out[valid]).all() or np.any(out[valid]<0):raise ValueError('invalid factual costs')
    return out


def pulse_timing_diagnostic(physical,case,*,dt):
    """Compare per-lower-tick world impulses on the logged task clock.

    A contact-ending tick is still physically executed. Missing future ticks
    are unobserved, not evidence of an incorrectly applied pulse. This checks
    the lower-tick integral only, not within-tick wrench/lever-arm dynamics.
    """
    active=np.asarray(physical['active_tick'],bool)
    ticks=np.asarray(physical['physical_tick'])[active]
    impulses=np.asarray(physical['applied_force_impulse'],float)[active]
    if impulses.shape!=(len(ticks),3) or not math.isfinite(dt) or dt<=0:
        raise ValueError('explicit lower-tick impulse and positive dt required')
    if not np.isfinite(impulses).all() or not np.isfinite(ticks).all():
        return dict(pulse_timing_status='unknown',pulse_observation='unknown',
            pulse_max_tick_impulse_error_Ns=None,first_applied_task_tick=None,last_applied_task_tick=None)
    start=case['pulse_onset_task_tick']*dt
    end=start+case.get('pulse_duration_s',.1)
    direction=np.asarray(case.get('world_direction',[0.,0.,0.]),float)
    if case['force_N']!=0 and 'world_direction' not in case:
        raise ValueError('explicit original world force direction required')
    duration=np.maximum(0.,np.minimum((ticks+1)*dt,end)-np.maximum(ticks*dt,start))
    expected=duration[:,None]*case['force_N']*direction
    # Apply the existing 1% exposure tolerance to a full-tick impulse scale,
    # including zero-expected boundary ticks. Relative tolerance on expected=0
    # alone spuriously flags float32 absolute-clock overlap spill at an edge.
    tolerance=1e-5+.01*np.abs(case['force_N']*direction)*dt
    matches=(np.abs(impulses-expected)<=tolerance).all()
    applied=np.linalg.norm(impulses,axis=1)>float(np.linalg.norm(tolerance))
    full=abs(float(duration.sum())-case.get('pulse_duration_s',.1))<dt*1e-3
    observation=('zero_force' if case['force_N']==0 else 'full_window' if full else
        'partial_window' if duration.sum()>0 else 'before_pulse')
    return dict(pulse_timing_status='matched_observed_ticks' if matches else 'mismatch',
        pulse_observation=observation,
        pulse_max_tick_impulse_error_Ns=float(np.max(np.abs(impulses-expected))) if len(ticks) else None,
        first_applied_task_tick=int(ticks[applied][0]) if applied.any() else None,
        last_applied_task_tick=int(ticks[applied][-1]) if applied.any() else None)


def score_endpoint_case(physical,case,goal_xy,*,config,horizon_s=10.,goal_radius_m=1.):
    """First original-goal hit and whole-window S, with explicit raw-cost aliases."""
    from .goal_task_credit import resolve_goal_task_credit
    active=np.asarray(physical['active_tick'],bool)
    fault=np.logical_or.reduce([physical[k] for k in ('nonfinite','lower_fault','policy_fault')])
    contact=np.asarray(physical['contact_failure'],bool)
    dt=float(config['timing']['lower_dt_s']);scale=float(config['path_reward']['scale'])
    pulse_diagnostic=pulse_timing_diagnostic(physical,case,dt=dt)
    # These are legal reference decisions, not per-lower-tick repeated actions.
    action=np.zeros_like(active);action[::4]=np.asarray(physical['actor_action_valid'],bool)[::4]
    views=[]
    for alpha in case['user_alphas']:
        components=rescore_original_components(physical,user_alpha=alpha,config=config,recorded_alpha=case['controller_alpha'])
        tick_cost=scale*dt*components.sum(axis=1)
        credit=resolve_goal_task_credit(actual_xy=physical['actual_xy'],goal_xy=goal_xy,radius_m=goal_radius_m,
            running_tick_cost=tick_cost,task_cost_valid=physical['task_reward_valid'],active_tick=active,
            actor_action_valid=action,contact_failure=contact,engineering_fault=fault,
            horizon_reached=physical['horizon_reached'],physical_tick=physical['physical_tick'],
            window_start_tick=0,window_ticks=round(horizon_s/dt),collection_policy_id=case['case_id'],learner_policy_id=case['case_id'])
        stop=credit['goal_terminal_index']
        terminal=credit['physical']['terminal_index']
        observed=active & ((np.arange(len(active))<=terminal) if terminal is not None else True)
        rows=observed & ~contact & ~fault & np.asarray(physical['task_reward_valid'],bool)
        prefix=rows & (np.arange(len(rows))<=stop) if stop is not None else np.zeros_like(rows)
        hit=credit['goal_entry_index'];finite_observed=observed & np.isfinite(physical['actual_xy']).all(axis=1)
        distance=np.linalg.norm(np.asarray(physical['actual_xy'])[finite_observed]-np.asarray(goal_xy),axis=1)
        prefatal=finite_observed & ~contact & ~fault
        upright_distance=np.linalg.norm(np.asarray(physical['actual_xy'])[prefatal]-np.asarray(goal_xy),axis=1)
        speed_error=np.asarray(physical['actual_forward_speed'])-np.asarray(physical['requested_speed'])
        rmse=lambda x:float(np.sqrt(np.mean(np.asarray(x)[rows]**2))) if rows.any() else None
        costs_known=credit['cost_window_complete'] and stop is not None
        pre_cost=float(tick_cost[prefix].sum(dtype=np.float64)) if costs_known else None
        # Legacy scored return replaces an entire contact-ending upper block.
        # Its fixed remaining-horizon penalty is alpha independent; preserve
        # exactly those factual recorded blocks while reweighting other ticks.
        scored=np.where(active,-tick_cost,0.)
        if len(active)%4:raise ValueError('complete allocated upper blocks required')
        ending=np.repeat(contact.reshape(-1,4).any(axis=1),4)
        scored[ending]=np.asarray(physical['scored_tick_reward'])[ending]
        impulse=np.asarray(physical['applied_force_impulse'])[active].sum(axis=0,dtype=np.float64)
        if case['force_N']!=0 and 'world_direction' not in case:raise ValueError('explicit original world force direction required')
        expected=case['force_N']*case.get('pulse_duration_s',.1)*np.asarray(case.get('world_direction',[0.,0.,0.]))
        exposure=('zero_force' if np.linalg.norm(impulse)<1e-8 else 'unexpected_force') if case['force_N']==0 else (
            'none' if np.linalg.norm(impulse)<1e-8 else 'full' if np.allclose(impulse,expected,rtol=.01,atol=1e-5) else 'partial_or_mismatch')
        views.append(dict(case_id=case['case_id'],suite=case['suite'],context=case['context'],role=case['role_name'],
            user_alpha=alpha,controller_alpha=case['controller_alpha'],force_N=case['force_N'],onset_task_tick=case['pulse_onset_task_tick'],
            goal=credit['goal_outcome'],survival=credit['physical']['outcome'],cost_window_complete=credit['cost_window_complete'],
            goal_time_s=(hit+1)*dt if hit is not None else None,minimum_goal_distance_m=float(distance.min()) if len(distance) else None,
            minimum_prefatal_goal_distance_m=float(upright_distance.min()) if len(upright_distance) else None,
            measured_precontact_duration_s=float(rows.sum()*dt),speed_rmse_m_s=rmse(speed_error),
            path_rmse_m=rmse(physical['path_cross_track']),heading_rmse_rad=rmse(physical['path_heading_error']),
            physical_ticks=credit['physical']['physical_ticks'],pre_goal_cost=pre_cost,
            full_precontact_running_cost=float(tick_cost[rows].sum(dtype=np.float64)),legacy_scored_return=float(scored.sum(dtype=np.float64)),
            pre_goal_component_cost=(scale*dt*components[prefix].sum(axis=0)).tolist() if costs_known else None,
            impulse_Ns=impulse.tolist(),expected_impulse_Ns=expected.tolist(),pulse_exposure=exposure,
            contact_time_s=(credit['physical']['terminal_index']+1)*dt if credit['physical']['outcome']=='contact' else None,
            **pulse_diagnostic))
    return views
