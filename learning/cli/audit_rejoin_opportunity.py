"""Diagnose saved live-control misses; no new controller or physical rollout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, output):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    for name in ['status.json','analysis/status.json']:
        if json.loads((root/name).read_text())['phase']!='completed':raise ValueError('completed source required')
    plan=json.loads((root/'plan.json').read_text())
    cfg=json.loads((root/'common_environment_config.json').read_text())
    dt=cfg['timing']['lower_dt_s'];horizon=plan['horizon_s']
    contexts=plan['contexts'];goals={}
    for name,c in contexts.items():
        if digest(c['path'])!=c['path_sha256']:raise ValueError('path identity drift')
        with np.load(c['path'],allow_pickle=False) as z:
            goals[name]=np.array([np.interp(float(z['goal']),z['s'],z['xy'][:,i]) for i in range(2)])
    fields=cfg['network']['context_fields'];ey=[f for f in fields if f['name']=='signed_cross_track_error_m']
    if len(ey)!=1:raise ValueError('unique cross-track context required')
    # Existing assemble_observation clips scaled context to [-5,5]. This is a
    # measurement of representation clipping, not proof it causes the miss.
    ey_clip_m=5*ey[0]['scale']
    allrows=json.loads((root/'analysis/results.json').read_text())
    rows=[r for r in allrows if r['role'] in ('feedback_010','steer_only_feedback')]
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('duplicate cases')
    records=[];checked=0
    for row in rows:
        if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('raw identity drift')
        checked+=1
        if not row['force_N'] or row['goal_arrived'] or row['physical_outcome']!='survived':continue
        with np.load(row['raw_file'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        active=np.asarray(raw['active_tick'],bool);tick=raw['physical_tick'][active]
        if not np.array_equal(tick,np.arange(round(horizon/dt))):raise ValueError('full live window required')
        time=(tick+1)*dt;xy=raw['actual_xy'][active];distance=np.linalg.norm(xy-goals[row['context']],axis=1)
        closest=int(np.argmin(distance))
        if not np.isclose(distance[closest],row['minimum_upright_goal_distance_m'],atol=2e-5,rtol=0):
            raise ValueError('original goal reconstruction disagrees')
        post=time>=row['pulse_onset_task_tick']*dt+row['pulse_duration_s']
        last=time>horizon-1
        legal=raw['actor_action_valid'][active].astype(bool)
        nominal=raw['nominal_reference'][active]
        changes=np.r_[False,np.any(np.abs(np.diff(nominal,axis=0))>1e-6,axis=1)]
        offsets=raw['correction_offset'][active]
        c=cfg['residual_interface'];reference=raw['reference'][active]
        clipped=np.abs(raw['path_cross_track'][active])>ey_clip_m
        decoded=raw['feedback_path_error_m'][active]
        # Feedback is sampled before each 20ms block and held for four physics
        # ticks; path_cross_track is post-tick. Do not equate the two clocks.
        np.testing.assert_array_less(np.abs(decoded[legal]),ey_clip_m+2e-6)
        decoded_clipped=legal&np.isclose(np.abs(decoded),ey_clip_m,atol=2e-6,rtol=0)
        record=dict(case_id=row['case_id'],role=row['role'],context=row['context'],alpha=row['alpha'],
            force_N=row['force_N'],minimum_goal_distance_m=float(distance[closest]),closest_task_time_s=float(time[closest]),
            time_remaining_at_closest_s=float(horizon-time[closest]),final_goal_distance_m=float(distance[-1]),
            goal_plane_crossed=bool(np.any(raw['goal_section_signed_distance'][active]>=0)),
            post_pulse_actor_legal_fraction=float(np.mean(legal[post])),last_1s_actor_legal_fraction=float(np.mean(legal[last])),
            last_1s_nominal_change_ticks=int(np.sum(changes&last)),
            post_pulse_context_cross_track_clipped_fraction=float(np.mean(clipped[post])),
            last_1s_context_cross_track_clipped_fraction=float(np.mean(clipped[last])),
            post_pulse_legal_feedback_at_clip_fraction=float(np.mean(decoded_clipped[post&legal])) if np.any(post&legal) else None,
            maximum_absolute_cross_track_m=float(np.max(np.abs(raw['path_cross_track'][active]))),
            post_pulse_steer_offset_near_bound_fraction=float(np.mean(np.abs(offsets[post,1])>=.99*c['steer_scale_rad'])),
            post_pulse_issued_steer_near_bound_fraction=float(np.mean(np.abs(reference[post,1])>=.99*c['steer_reference_bound_rad'])),
            last_1s_steering_tracking_mae_rad=float(np.mean(np.abs(raw['actual_delta'][active][last]-reference[last,1]))),
            raw_file=row['raw_file'],raw_sha256=row['raw_sha256'])
        records.append(record)
    summary=dict(source=str(root),source_results_sha256=digest(root/'analysis/results.json'),
        source_script_sha256=digest(__file__),raw_files_authenticated=checked,misses=len(records),
        cross_track_context_clip_m=ey_clip_m,
        per_role={role:dict(misses=len(x),
            all_last_second_legal=sum(r['last_1s_actor_legal_fraction']==1 for r in x),
            nominal_updates_in_last_second=sum(r['last_1s_nominal_change_ticks']>0 for r in x),
            goal_plane_crossed=sum(r['goal_plane_crossed'] for r in x),
            closest_within_final_second=sum(r['time_remaining_at_closest_s']<=1 for r in x),
            any_post_pulse_context_clipping=sum(r['post_pulse_context_cross_track_clipped_fraction']>0 for r in x))
            for role in ('feedback_010','steer_only_feedback') if (x:=[r for r in records if r['role']==role])},
        physics_steps=0,optimizer_updates=0,
        limits='Correlated DEV misses only. Updated references do not prove the geometric target is good; '
               'clipping is not a causal explanation. No inference of physical infeasibility or new recovery gain. '
               'Task0 histories, original goal, 10s and historical labels unchanged.')
    output.mkdir()
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'misses.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    if records:
        with (output/'misses.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    return summary


def review_tails(root, audit_dir, output):
    """Describe last-two-second motion; preserve the original 10s task labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    plan=json.loads((root/'plan.json').read_text())
    cfg=json.loads((root/'common_environment_config.json').read_text());dt=cfg['timing']['lower_dt_s']
    rows=json.loads((root/'analysis/results.json').read_text())
    misses=json.loads((Path(audit_dir)/'misses.json').read_text())
    index={(r['context'],r['alpha'],r['event_id'],r['role']):r for r in rows}
    output.mkdir();summaries=[];figures=[];diagnostics={};checked={}
    keys=sorted({(r['context'],r['alpha'],next(x['event_id'] for x in rows if x['case_id']==r['case_id'])) for r in misses})
    for number,(context,alpha,event) in enumerate(keys):
        c=plan['contexts'][context]
        if digest(c['path'])!=c['path_sha256']:raise ValueError('path changed')
        with np.load(c['path'],allow_pickle=False) as z:path={k:z[k] for k in z.files}
        goal=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in range(2)])
        fig,axes=plt.subplots(1,4,figsize=(17,4.2),layout='constrained')
        axes[0].plot(*path['xy'][path['s']<=float(path['goal'])].T,':',color='.6',label='Original path')
        axes[0].add_patch(Circle(goal,.5,fill=False,color='black'));axes[0].plot(*goal,'k*')
        for role,color,label in [('feedback_010','#0072B2','Full neural middle + fixed feedback'),
                                 ('steer_only_feedback','#009E73','Neural steering only + fixed feedback')]:
            row=index[context,alpha,event,role];file=row['raw_file']
            if digest(file)!=row['raw_sha256']:raise ValueError('raw changed')
            checked[file]=row['raw_sha256']
            with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            active=raw['active_tick'].astype(bool)
            if row['physical_outcome']!='survived' or active.sum()!=2000:raise ValueError('complete paired survivor required')
            xy=raw['actual_xy'][active];t=(raw['physical_tick'][active]+1)*dt
            d=np.linalg.norm(xy-goal,axis=1);tail=t>=8.
            lag=round(.1/dt);closing=(d[:-lag]-d[lag:])/(lag*dt);tt=t[lag:]
            final_lag=round(.2/dt);final_closing=float((d[-1-final_lag]-d[-1])/(final_lag*dt))
            # This threshold labels measured motion, not task success or safety.
            motion='approaching' if final_closing>.1 else 'moving_away' if final_closing<-.1 else 'low_radial_motion'
            if not row['goal_arrived']:
                summaries.append(dict(case_id=row['case_id'],closest_goal_m=float(np.min(d)),
                    final_goal_m=float(d[-1]),distance_change_last_2s_m=float(d[-1]-d[np.flatnonzero(tail)[0]]),
                    final_0p2s_mean_closing_speed_m_s=final_closing,final_motion=motion,
                    moving_away_in_last_2s_fraction=float(np.mean(closing[tt>=8]<-.1)),
                    paired_role_arrived=index[context,alpha,event,'steer_only_feedback' if role=='feedback_010' else 'feedback_010']['goal_arrived']))
            diagnostics[row['case_id']]=dict(time=t,xy=xy,distance=d,closing_time=tt,closing_speed=closing,
                logged_projection_remaining=float(path['goal'])-raw['path_progress'][active])
            axes[0].plot(*xy.T,color=color,alpha=.25);axes[0].plot(*xy[tail].T,color=color,label=label)
            axes[0].plot(*xy[-1],'o',color=color)
            axes[1].plot(t[tail],d[tail],color=color)
            axes[2].plot(tt[tt>=8],closing[tt>=8],color=color)
            axes[3].plot(t[tail],float(path['goal'])-raw['path_progress'][active][tail],color=color)
        axes[0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Actual XY; thick = last 2s');axes[0].set_aspect('equal',adjustable='box')
        axes[1].set(title='Distance to original goal',ylabel='Distance (m)');axes[1].axhline(.5,color='k',ls=':')
        axes[2].set(title='Positive = approaching goal',ylabel='Closing speed (m/s)');axes[2].axhline(0,color='k',ls=':')
        axes[3].set(title='Logged projection to goal',ylabel='Signed progress remaining (m)');axes[3].axhline(0,color='k',ls=':')
        for ax in axes[1:]:ax.set(xlabel='Task time (s)',xlim=(8,10))
        for ax in axes:ax.grid(alpha=.25)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle(f'{context} | original alpha={alpha:g} | {event} | same complete task0, 10s DEV')
        name=f'paired_tail_{number:02d}.png';fig.savefig(output/name,dpi=120,bbox_inches='tight');plt.close(fig);figures.append(name)
    np.savez_compressed(output/'tail_series.npz',**{f'{i:03d}_{k}':v for i,series in enumerate(diagnostics.values()) for k,v in series.items()})
    (output/'series_case_ids.json').write_text(json.dumps(list(diagnostics),indent=2)+'\n')
    counts={motion:sum(r['final_motion']==motion for r in summaries) for motion in ['approaching','moving_away','low_radial_motion']}
    result=dict(misses=summaries,final_motion_counts=counts,figures=figures,authenticated_unique_raw=len(checked),
        raw_sha256=checked,physics_steps=0,optimizer_updates=0,visual_review_complete=False,
        scope='End motion threshold +/-0.1m/s on last0.2s is descriptive only. Closing speed uses trailing0.1s chord. '
              'Remaining progress is the existing monotone projection, not physical shortest distance or recoverability. '
              'No extension past10s, new rollout, causal claim or task relabeling.')
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(final_motion_counts=counts,misses=len(summaries),paired_events=len(keys),physics_steps=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tail-from',type=Path)
    args=parser.parse_args()
    result=review_tails(args.run,args.tail_from,args.output) if args.tail_from else audit(args.run,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
