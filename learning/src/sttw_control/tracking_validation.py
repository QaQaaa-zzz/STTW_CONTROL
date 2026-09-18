"""Fixed-horizon paired geometric validation using the training return state.

No horizon extension, post-failure zero padding, or yaw-rate-as-path criterion.
"""
import jax
import jax.numpy as jp
import numpy as np
from .tracking_reward import tolerances, timed_tolerances, directional_tolerances


def make_tracking_validator(env, actor, scale, config):
    c=env.config; tc=c.tracking; dt=c.controller.dt
    bank=getattr(c,"reference_paths",None)
    scenario_names=getattr(config,'development_scenarios',None)
    cases=(scenario_names if scenario_names is not None
           else (config.validation_events or ({'start':0.,'duration':dt},)))
    if scenario_names is None:
        for case in cases:
            if any(not np.isclose(case[k]/dt,round(case[k]/dt),rtol=0,atol=1e-7)
                   for k in ('start','duration')):
                raise ValueError('validation events must align to control ticks')
            present_case=any(case.get(k,0.)!=0 for k in ('steer_rate','force','rear_torque'))
            if present_case and case['start']+case['duration']+tc.return_seconds>c.horizon_seconds+1e-7:
                raise ValueError('declared return window does not fit the fixed episode')
    choices=c.priority.validation_alphas
    keys=jp.stack([jax.random.PRNGKey(seed) for alpha in choices
                   for seed in config.validation_seeds for case in cases])
    alphas=jp.asarray([alpha for alpha in choices for seed in config.validation_seeds for case in cases])
    schedules=None
    if scenario_names is not None:
        from dataclasses import replace
        from .timed_reference import schedule
        schedule_rows=[];event_list=[]
        for alpha in choices:
            for seed in config.validation_seeds:
                for name in scenario_names:
                    ref=replace(c.timed_reference,training_mix=True,fixed_scenario=name)
                    rows=schedule(jax.random.fold_in(jax.random.PRNGKey(seed),51),ref,c.speed_reference)
                    schedule_rows.append(rows);event_list.append(env._event_for(rows,jp.zeros(6)))
        schedules=jp.stack(schedule_rows)
        event_rows=np.asarray(jp.stack(event_list),dtype=np.float32)
    else:
        event_rows=np.asarray([[round(case['start']/dt),round((case['start']+case['duration'])/dt),
                           case.get('steer_rate',0.),case.get('force',0.),
                           float(case.get('waveform','constant')=='half_sine'),case.get('rear_torque',0.)]
                          for alpha in choices for seed in config.validation_seeds for case in cases],dtype=np.float32)
    path_ids=jp.zeros(len(keys),dtype=jp.int32)
    if bank is not None:
        count=len(bank.cases)
        path_ids=jp.repeat(jp.arange(count,dtype=jp.int32),len(keys))
        keys=jp.tile(keys,(count,1));alphas=jp.tile(alphas,count)
        event_rows=np.tile(event_rows,(count,1))
    # Assemble the paired panel explicitly before tracing; no scatter updates
    # to closed-over event constants and no concatenation of aliased states.
    nominal_rows=event_rows.copy();nominal_rows[:,[2,3,5]]=0.
    all_events=jp.asarray(np.concatenate((event_rows,nominal_rows),axis=0))
    all_keys=jp.concatenate((keys,keys),axis=0)
    all_alphas=jp.concatenate((alphas,alphas),axis=0)
    all_schedules=(jp.concatenate((schedules,schedules),axis=0) if schedules is not None else None)
    present=jp.asarray(np.any(event_rows[:,[2,3,5]]!=0.,axis=1))
    n=len(keys); reset=jax.vmap(env.reset); step=jax.vmap(env.step)

    @jax.jit
    def validate(params, zero=False, base_output_scale=jp.nan):
        initial=reset(all_keys) if bank is None else jax.vmap(lambda key,i:env.reset(key,reference_id=i))(all_keys,jp.concatenate((path_ids,path_ids)))
        state=jax.vmap(env.set_priority)(initial,all_alphas).replace(event=all_events)
        if all_schedules is not None:
            state=state.replace(command_schedule=all_schedules)
        if hasattr(state,'active_base_output_scale'):
            state=state.replace(active_base_output_scale=jp.where(
                jp.isfinite(base_output_scale),base_output_scale,state.active_base_output_scale))
        def tick(state,_):
            active=~state.done
            action=actor.apply(params['actor'],state.obs/scale)
            nxt=step(state,jp.where(zero,jp.zeros_like(action),action))
            if getattr(c,'timed_reference',None) is not None:
                from .timed_reference import errors
                timed_features=jax.vmap(errors)(nxt.pose,nxt.reference_pose,state.reference_command)
                features=nxt.geometric_features if tc.geometric else timed_features[:,:3]
            else:features=jax.vmap(env.path_features)(nxt.pose) if bank is None else jax.vmap(env.path_features)(nxt.pose,nxt.path_id,nxt.path_progress)
            speed=jp.sum(nxt.data.qvel[:,:3]*nxt.data.xmat[:,env.bundle.chassis,:,0],axis=-1)
            ev=speed-(state.reference_command[:,0] if getattr(c,'timed_reference',None) is not None else (jax.vmap(env.speed_command)(state.tick) if bank is None else jax.vmap(env.speed_command)(state.tick,state.path_id)))
            if tc.objective=='asymmetric_geometric_huber':
                under_band,over_band,by=directional_tolerances(
                    state.priority_alpha,nxt.tracking_state.elapsed,tc)
                speed_out=(jp.maximum(-ev,0.)>under_band)|(jp.maximum(ev,0.)>over_band)
            else:
                bv,by=tolerances(state.priority_alpha,tc)
                if tc.shrink_tolerances:
                    fraction=jp.clip(nxt.tracking_state.elapsed/(tc.return_seconds-tc.hold_seconds),0,1)
                    bv=bv+fraction*(tc.final_speed_tolerance-bv)
                    by=by+fraction*(tc.final_lateral_tolerance-by)
                speed_out=jp.abs(ev)>bv
            mature=active & (state.tick*dt>=tc.start_seconds)
            row=(active,features[:,0]**2,ev**2,features[:,1]**2,nxt.reward,
                 jp.where(active,jp.abs(nxt.measurement[:,0]),0.),
                 mature,mature & speed_out,mature & (jp.abs(features[:,0])>by),
                 jp.where(active[:n],jp.abs(features[:n,0]-features[n:,0]),0.),
                 jp.where(active,jp.maximum(-ev,0.),0.),jp.where(active,jp.maximum(ev,0.),0.),
                 mature & (jp.maximum(-ev,0.)>under_band) if tc.objective=='asymmetric_geometric_huber' else mature & speed_out,
                 mature & (jp.maximum(ev,0.)>over_band) if tc.objective=='asymmetric_geometric_huber' else mature & speed_out)
            if tc.timed:
                bx,bw=timed_tolerances(state.priority_alpha,tc)
                yaw_error=nxt.yaw_rate-state.reference_command[:,1]
                row=row+(timed_features[:,3]**2,yaw_error**2,
                         mature & (jp.abs(timed_features[:,3])>bx),
                         mature & (jp.abs(yaw_error)>bw))
            return nxt,row
        last,rows=jax.lax.scan(tick,state,None,length=env.horizon)
        active,lateral,speed,heading,reward,roll,mature,speed_out,path_out,extra=rows[:10]
        count=jp.maximum(jp.sum(active,axis=0),1)
        mature_count=jp.maximum(jp.sum(mature,axis=0),1)
        def rmse(error):return jp.sqrt(jp.sum(jp.where(active,error,0.),axis=0)/count)
        rs=last.tracking_state
        complete=((last.tick>=env.horizon)&~last.terminated&~rs.pending
                  &(rs.hold+1e-6>=tc.hold_seconds))
        qualified=complete&~rs.deadline_missed
        timed_metrics=({"longitudinal_rmse":rmse(rows[14])[:n],"nominal_longitudinal_rmse":rmse(rows[14])[n:],
                       "yaw_rate_rmse":rmse(rows[15])[:n],"nominal_yaw_rate_rmse":rmse(rows[15])[n:],
                       "longitudinal_tolerance_exceed_fraction":(jp.sum(rows[16],axis=0)/mature_count)[:n],
                       "nominal_longitudinal_tolerance_exceed_fraction":(jp.sum(rows[16],axis=0)/mature_count)[n:],
                       "yaw_rate_tolerance_exceed_fraction":(jp.sum(rows[17],axis=0)/mature_count)[:n],
                       "nominal_yaw_rate_tolerance_exceed_fraction":(jp.sum(rows[17],axis=0)/mature_count)[n:]} if tc.timed else {})
        return dict(**timed_metrics,return_ever_left=rs.ever_left[:n],return_credited=rs.credited[:n],
                    return_pending=rs.pending[:n],return_hold_seconds=rs.hold[:n],
                    radial_rmse=rmse(lateral)[:n],speed_rmse=rmse(speed)[:n],
                    heading_rmse=rmse(heading)[:n],nominal_radial_rmse=rmse(lateral)[n:],
                    nominal_speed_rmse=rmse(speed)[n:],nominal_heading_rmse=rmse(heading)[n:],
                    failed=last.terminated[:n],nominal_failed=last.terminated[n:],
                    episode_return=jp.sum(jp.where(active,reward,0.),axis=0)[:n],
                    nominal_episode_return=jp.sum(jp.where(active,reward,0.),axis=0)[n:],
                    terminal_tracking_hold=complete[:n],nominal_terminal_tracking_hold=complete[n:],
                    post_event_hold_complete=qualified[:n],event_present=present,
                    return_deadline_missed=rs.deadline_missed[:n],
                    nominal_return_deadline_missed=rs.deadline_missed[n:],
                    recovered_after_excursion=qualified[:n]&rs.ever_left[:n]&rs.credited[:n],
                    maintained_without_excursion=qualified[:n]&~rs.ever_left[:n],
                    roll_peak=jp.max(roll,axis=0)[:n],nominal_roll_peak=jp.max(roll,axis=0)[n:],
                    speed_tolerance_exceed_fraction=(jp.sum(speed_out,axis=0)/mature_count)[:n],
                    underspeed_peak=jp.max(rows[10],axis=0)[:n],overspeed_peak=jp.max(rows[11],axis=0)[:n],
                    underspeed_integral=dt*jp.sum(rows[10],axis=0)[:n],overspeed_integral=dt*jp.sum(rows[11],axis=0)[:n],
                    underspeed_band_seconds=dt*jp.sum(rows[12],axis=0)[:n],overspeed_band_seconds=dt*jp.sum(rows[13],axis=0)[:n],
                    path_tolerance_exceed_fraction=(jp.sum(path_out,axis=0)/mature_count)[:n],
                    post_event_peak=jp.max(extra,axis=0),steps=count[:n],nominal_steps=count[n:])
    return validate
