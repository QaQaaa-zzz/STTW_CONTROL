"""Describe observed failure timing and available reference permissions; no causal claim.

World-force metadata is used only for retrospective exposure description, never
to choose a reference. Does not reconstruct unlogged actuator rates or ESO state.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def readback(root):
    root=root.resolve();source=root/'preservation_timecheck_v1/analysis'
    if json.loads((source/'status.json').read_text())['phase']!='completed':raise ValueError('completed source required')
    file=source/'results.json';rows=json.loads(file.read_text())
    out=source/'mechanism_readback'
    if out.exists() and json.loads((out/'summary.json').read_text())['source_results_sha256']!=digest(file):
        raise ValueError('do not overwrite a different source analysis')
    out.mkdir(exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());goals={}
    declaration=json.loads((root/'source_declaration.json').read_text())
    steer_limit=float(declaration['source']['actuator']['steer_limit'])
    config=json.loads((root/'common_environment_config.json').read_text())
    positive_speed_authority=float(config['residual_interface']['speed_positive_scale_m_s'])
    for context,c in plan['contexts'].items():
        if digest(c['path'])!=c['path_sha256']:raise ValueError('original path changed')
        with np.load(c['path']) as z:goals[context]=np.array([np.interp(float(z['goal']),z['s'],z['xy'][:,i]) for i in range(2)])
    cache={};observations=[]
    for row in rows:
        if not row['force_N']:continue
        key=(row['raw_sha256'],row['context'])
        if key not in cache:
            if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('changed physical trace')
            with np.load(row['raw_file']) as z:raw={k:z[k] for k in z.files}
            active=np.asarray(raw['active_tick'],bool);ticks=raw['physical_tick'][active]
            if not np.array_equal(ticks,np.arange(len(ticks))):raise ValueError('actual prefix clock required')
            seen=lambda field:np.asarray(raw[field])[active]
            first=lambda flags:float((ticks[np.flatnonzero(flags)[0]]+1)*.005) if np.any(flags) else None
            contact=seen('contact_failure').astype(bool);invalid=seen('domain_invalid').astype(bool)
            near_stop=abs(seen('actual_delta'))>=.99*steer_limit
            disabled=~seen('actor_action_valid').astype(bool);onset=row['pulse_onset_task_tick']*.005
            before=np.flatnonzero(ticks<row['pulse_onset_task_tick']);pre=int(before[-1]) if len(before) else None
            first_contact=first(contact);first_invalid=first(invalid);first_disabled=first(disabled)
            support_trigger=seen('relay_trigger').astype(bool)&(ticks>=row['pulse_onset_task_tick'])
            after_stop=near_stop&(ticks>=row['pulse_onset_task_tick'])
            trigger_time=first(support_trigger);near_stop_time=first(after_stop)
            suppressed=seen('relay_active').astype(bool)&disabled
            if suppressed.any() and np.any(seen('target_offset')[suppressed]!=0):raise ValueError('old disabled-support semantics changed')
            trigger_index=int(np.flatnonzero(support_trigger)[0]) if support_trigger.any() else None
            flags=dict(contact_s=first_contact,first_geometric_invalid_s=first_invalid,first_actor_disabled_s=first_disabled,
                geometric_invalid_before_contact=bool(first_contact is not None and first_invalid is not None and first_invalid<first_contact),
                actor_disabled_before_contact=bool(first_contact is not None and first_disabled is not None and first_disabled<first_contact),
                first_contact_delay_after_onset_s=first_contact-onset if first_contact is not None else None,
                invalid_observed=bool(invalid.any()),actor_disabled_observed=bool(disabled.any()),
                invalid_observed_s=float(invalid.sum()*.005),actor_disabled_observed_s=float(disabled.sum()*.005),
                reentry_observed_s=float(seen('reentry_actor_active').sum()*.005),
                peak_abs_roll_rad=float(abs(seen('phi')).max()),peak_abs_roll_rate_rad_s=float(abs(seen('phi_dot')).max()),
                peak_abs_actual_steering_rad=float(abs(seen('actual_delta')).max()),
                steering_near_joint_limit_observed=bool(near_stop.any()),
                steering_near_joint_limit_s=float(near_stop.sum()*.005),
                first_steering_near_joint_limit_s=first(near_stop),
                first_postonset_support_trigger_s=trigger_time,
                existing_support_active_before_onset=bool(seen('relay_active')[pre]) if pre is not None else None,
                support_trigger_preceded_near_limit=bool(trigger_time is not None and near_stop_time is not None and trigger_time<near_stop_time),
                speed_target_at_existing_support_trigger_m_s=float(seen('target_offset')[trigger_index,0]) if trigger_index is not None else None,
                support_active_but_actor_disabled_s=float(suppressed.sum()*.005),
                first_support_active_but_actor_disabled_s=first(suppressed),
                final_goal_distance_m=float(np.linalg.norm(seen('actual_xy')[-1]-goals[row['context']])))
            if invalid.any():
                k=int(np.flatnonzero(invalid)[0])
                flags['at_first_geometric_invalid']=dict(roll_rad=float(seen('phi')[k]),roll_rate_rad_s=float(seen('phi_dot')[k]),
                    actual_speed_m_s=float(seen('actual_forward_speed')[k]),steering_rad=float(seen('actual_delta')[k]),
                    seconds_until_contact=first_contact-first_invalid if first_contact is not None else None)
            if pre is not None:
                yaw=float(seen('actual_yaw')[pre]);direction=np.asarray(row['world_direction'])
                flags['before_onset']=dict(actual_speed_m_s=float(seen('actual_forward_speed')[pre]),
                    roll_rad=float(seen('phi')[pre]),roll_rate_rad_s=float(seen('phi_dot')[pre]),
                    steering_rad=float(seen('actual_delta')[pre]),yaw_rad=yaw,
                    reference=np.asarray(seen('reference')[pre]).tolist(),
                    progress_m=float(seen('path_progress')[pre]),
                    horizontal_heading_normal_force_N=float(row['force_N']*(-np.sin(yaw)*direction[0]+np.cos(yaw)*direction[1])),
                    actor_enabled=bool(seen('actor_action_valid')[pre]),geometric_invalid=bool(invalid[pre]))
            cache[key]=flags
        observations.append(dict(case_id=row['case_id'],context=row['context'],alpha=row['alpha'],role=row['role'],
            event_id=row['event_id'],force_N=row['force_N'],raw_sha256=row['raw_sha256'],physical_outcome=row['physical_outcome'],
            arrived_1m=row['goals']['1.0']['goal_arrived'],minimum_goal_distance_m=row['goals']['1.0']['minimum_upright_goal_distance_m'],**cache[key]))
    summary={}
    for method in ['full','context_choice','geometry_alpha','alpha_only']:
        rr=[r for r in observations if r['role']==method];contact=[r for r in rr if r['physical_outcome']=='contact'];miss=[r for r in rr if r['physical_outcome']=='survived' and not r['arrived_1m']]
        summary[method]=dict(n=len(rr),contact_n=len(contact),contact_after_geometric_invalid=sum(r['geometric_invalid_before_contact'] for r in contact),
            contact_after_actor_disabled=sum(r['actor_disabled_before_contact'] for r in contact),
            contact_delay_range_s=[min(r['first_contact_delay_after_onset_s'] for r in contact),max(r['first_contact_delay_after_onset_s'] for r in contact)] if contact else None,
            surviving_nonarrival_n=len(miss),surviving_nonarrival_with_geometric_invalid=sum(r['invalid_observed'] for r in miss),
            surviving_nonarrival_with_actor_disabled=sum(r['actor_disabled_observed'] for r in miss),
            contact_with_steering_near_limit=sum(r['steering_near_joint_limit_observed'] for r in contact),
            contact_with_existing_support_trigger_before_near_limit=sum(r['support_trigger_preceded_near_limit'] for r in contact),
            contact_with_support_already_active_before_onset=sum(r['existing_support_active_before_onset'] for r in contact),
            contact_with_suppressed_existing_support=sum(r['support_active_but_actor_disabled_s']>0 for r in contact),
            contact_with_maximum_speed_target_at_trigger=sum(r['speed_target_at_existing_support_trigger_m_s'] is not None and abs(r['speed_target_at_existing_support_trigger_m_s']-positive_speed_authority)<1e-5 for r in contact),
            survived_with_steering_near_limit=sum(r['steering_near_joint_limit_observed'] for r in rr if r['physical_outcome']=='survived'),
            surviving_nonarrival_cases=[dict(case_id=r['case_id'],minimum_goal_distance_m=r['minimum_goal_distance_m'],invalid_s=r['invalid_observed_s'],disabled_s=r['actor_disabled_observed_s'],reentry_s=r['reentry_observed_s']) for r in miss])
    groups={}
    for r in observations:groups.setdefault((r['context'],r['alpha'],r['event_id']),{})[r['role']]=r
    all_contact=[k for k,g in groups.items() if all(r['physical_outcome']=='contact' for r in g.values())]
    atomic_json(out/'rows.json',observations)
    result=dict(methods=summary,all_four_choices_contact_events=all_contact,logical_rows=len(observations),unique_raw_traces=len(cache),
        source_results_sha256=digest(file),script_sha256=digest(__file__),new_physics_steps=0,optimizer_updates=0,
        source_declaration_sha256=digest(root/'source_declaration.json'),steering_joint_limit_rad=steer_limit,diagnostic_near_limit_fraction=.99,
        positive_speed_offset_authority_m_s=positive_speed_authority,
        scope='Observed timing association, not causal attribution. The99percent joint-angle marker is posthoc descriptive, not a fitted detector/control threshold. Currentgeometry logs after5ms and Actor eligibility held20ms;16cleanblock reentry already exists. Before-onset force projection is retrospective horizontal heading-normal, not full3D body-lateral force or a controller input. No reconstruction of unlogged actuator command/rate saturation or ESO states. Allchoices-fail is finite candidate evidence, not physical impossibility.')
    atomic_json(out/'summary.json',result)
    render_examples(out,rows,steer_limit)
    print(json.dumps(result,ensure_ascii=False))


def render_examples(out,rows,steer_limit):
    """Four disclosed illustrations; full statistics use all384 logical rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[('v1.7_yaw-12__left',0.,'early_F-32','full'),
           ('v1.7_yaw-12__left',0.,'late_F32','full'),
           ('v2.1_yaw+12__left',.5,'early_F-32','full'),
           ('v2.1_yaw+12__left',.5,'early_F-32','alpha_only')]
    fig,axes=plt.subplots(3,4,figsize=(15,8),layout='constrained');index=[]
    for column,(context,alpha,event,method) in enumerate(specs):
        row=next(r for r in rows if r['context'].startswith(context) and r['alpha']==alpha and r['event_id']==event and r['role']==method)
        with np.load(row['raw_file']) as z:
            active=np.asarray(z['active_tick'],bool);a=lambda k:np.asarray(z[k])[active]
            t=(a('physical_tick')+1)*.005
            series=[[(a('phi'),'Actual roll','-')],
                    [(a('actual_delta'),'Actual steering','-'),(a('reference')[:,1],'Steering reference','--')],
                    [(a('actual_forward_speed'),'Actual body speed','-'),(a('reference')[:,0],'Speed reference','--'),(a('requested_speed'),'Original request',':')]]
            for i,(lines,label) in enumerate(zip(series,['Roll (rad)','Steering (rad)','Speed (m/s)'])):
                ax=axes[i,column]
                for values,name,style in lines:
                    ax.plot(t,values,label=name,ls=style,lw=1.)
                    ax.scatter(t[-1],values[-1],marker='x' if row['physical_outcome']=='contact' else 'o',s=17)
                ax.fill_between(t,0,1,where=a('domain_invalid').astype(bool),transform=ax.get_xaxis_transform(),color='#CC6677',alpha=.12,label='Geometry invalid')
                start=row['pulse_onset_task_tick']*.005;ax.axvspan(start,start+row['pulse_duration_s'],color='gray',alpha=.2,label='Pulse')
                trigger=np.flatnonzero(a('relay_trigger')&(a('physical_tick')>=row['pulse_onset_task_tick']))
                if len(trigger):ax.axvline(t[trigger[0]],color='#0072B2',ls=':',lw=.8,label='Existing support trigger')
                ax.set(xlim=(0,10),xlabel='Original task time (s)',ylabel=label);ax.grid(alpha=.2)
                if column==0:ax.legend(fontsize=6,loc='best')
            axes[1,column].axhline(steer_limit,color='gray',ls=':',lw=.8);axes[1,column].axhline(-steer_limit,color='gray',ls=':',lw=.8)
            axes[0,column].set_title(f'{context.split("__")[0]} / a{alpha:g}\n{event} / {method} / {row["physical_outcome"]}',fontsize=8)
        index.append(dict(case_id=row['case_id'],raw_sha256=row['raw_sha256']))
    fig.suptitle('Observed timing examples: existing support, steering travel, and geometric validity\n'
                 'Four selected illustrations, not an efficacy comparison; all384 logical-event statistics retained separately',fontsize=11)
    fig.savefig(out/'timing_examples.png',dpi=140,facecolor='white');plt.close(fig)
    atomic_json(out/'figure_provenance.json',dict(cases=index,script_sha256=digest(__file__),figure_sha256=digest(out/'timing_examples.png'),
        selection='Illustrate contact with/without priorgeometry invalid, plus same-event full contact and alpha-only survival without arrival; not chosen to train a rule.',
        transformations='Actual5ms samples to10s; no interpolation/smoothing or invented postcontact samples. Four exact case IDs above. Steering joint-angle lines from source declaration, not unlogged rate commands.',
        new_physics_steps=0,journal_compliance_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);readback(p.parse_args().root)
