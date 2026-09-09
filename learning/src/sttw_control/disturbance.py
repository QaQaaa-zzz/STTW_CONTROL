"""Paired disturbance evaluation with explicitly separate path/hold metrics."""
import numpy as np
from .media import state_series
from .path import CircleConfig, lateral_space_metrics


def recovery_metrics(trace,nominal,config,*,path_tolerance=.2,extra_tolerance=.05,heading_tolerance=.15,hold_seconds=.5):
    """No resampling: compare equal-time samples; later failure invalidates recovery."""
    t=trace['time'];n=len(t)
    if len(nominal['time'])<n:
        return {'paired_reference_available':False,'reason':'nominal trajectory ended before disturbed trajectory; no extrapolation','failed':bool(trace['terminated'][-1]),'recovered_after_excursion':False,'post_event_extra_radial_peak_m':None,'first_joint_hold_completion_after_event_end_seconds':None,'settled_joint_hold_completion_after_event_end_seconds':None}
    if not np.allclose(t,nominal['time'][:n],rtol=0,atol=1e-5):
        raise ValueError('nominal/disturbed timestamps mismatch')
    c=config['circle'];xy=trace['qpos'][:,:2];nomxy=nominal['qpos'][:n,:2]
    center=np.array([c['center_x'],c['center_y']])
    radial=np.linalg.norm(xy-center,axis=1)-c['radius']
    nomradial=np.linalg.norm(nomxy-center,axis=1)-c['radius']
    extra=radial-nomradial
    s=state_series(trace,config)
    tangent=np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0])+c['direction']*np.pi/2
    heading=np.arctan2(np.sin(s['yaw']-tangent),np.cos(s['yaw']-tangent))
    start=config['disturbance_start'];end=start+config['disturbance_duration']
    post=t>=start-1e-8
    if not np.any(post):
        return {'event_reached':False,'failed':bool(trace['terminated'][-1]),'recovered_after_excursion':False,'final_joint_band_held':False,'left_extra_radial_band':False,'first_joint_hold_completion_after_event_end_seconds':None,'settled_joint_hold_completion_after_event_end_seconds':None,'post_event_radial_peak_m':None,'post_event_extra_radial_peak_m':None,'post_event_xy_separation_peak_m':None}
    band=np.abs(extra)<=extra_tolerance
    left=np.flatnonzero(post&~band)
    valid=band&(np.abs(radial)<=path_tolerance)&(np.abs(heading)<=heading_tolerance)&(np.abs(s['speed']-config['speed_reference'])<=.2)
    needed=int(np.ceil(hold_seconds/config['controller']['dt']))
    count=0;completion=None
    # First qualifying interval must start after event end AND first exit sample.
    earliest=max(end,float(t[left[0]]) if len(left) else end)
    for i in range(1,n):
        if t[i-1]>=earliest-1e-8 and valid[i]:count+=1
        else:count=0
        if count>=needed:
            completion=float(t[i]-end);break
    failed=bool(trace['terminated'][-1])
    tail=t>=max(end,t[-1]-hold_seconds)
    invalid_indices=np.flatnonzero((t>=earliest-1e-8)&~valid)
    settling_start=max(int(np.searchsorted(t,earliest-1e-8)),int(invalid_indices[-1]) if len(invalid_indices) else 0)
    settling_completion=settling_start+needed
    settled=(float(t[settling_completion]-end) if settling_completion<n and not failed and np.all(valid[settling_start+1:]) else None)
    space=lateral_space_metrics(xy[post],CircleConfig(**c))
    space.update({'window_start_seconds':float(t[post][0]),'window_end_seconds':float(t[-1]),
                  'window':'event_start_to_observed_episode_end',
                  'recovery_censored':settled is None,'physical_failure':failed})
    return {'event_reached':True,'path_relative_space':space,'post_event_radial_peak_m':float(np.max(np.abs(radial[post]))),
            'post_event_extra_radial_peak_m':float(np.max(np.abs(extra[post]))),
            'post_event_xy_separation_peak_m':float(np.max(np.linalg.norm(xy[post]-nomxy[post],axis=1))),
            'left_extra_radial_band':bool(len(left)),
            'first_joint_hold_completion_after_event_end_seconds':completion,
            'settled_joint_hold_completion_after_event_end_seconds':settled,
            'recovered_after_excursion':bool(len(left) and completion is not None and not failed and np.all(valid[tail])),
            'final_joint_band_held':bool(not failed and np.all(valid[tail]) and t[-1]-end>=hold_seconds),
            'failed':failed,'criteria':{'absolute_radial_m':path_tolerance,'extra_radial_m':extra_tolerance,'heading_rad':heading_tolerance,'true_speed_error_m_s':.2,'hold_seconds':hold_seconds},
            'note':'Extra radial error compares same-time nominal radial errors; XY separation includes phase lag. No excursion is not recovery evidence.'}


def plot_panel(root):
    """Plot all declared scenarios with event windows and nominal subtraction."""
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);decl=json.loads((root/'declaration.json').read_text())
    cases=[(k,c) for k,c in decl['scenarios'].items() if k!='nominal']
    fig,axes=plt.subplots(len(cases),3,figsize=(14,3*len(cases)),squeeze=False,layout='constrained')
    for row,(name,c) in zip(axes,cases):
        center=np.array([c['circle']['center_x'],c['circle']['center_y']])
        for policy,color,ls in [('baseline','#24567a','-'),('residual','#b45f24','--')]:
            tr=dict(np.load(root/name/policy/'trace.npz'));nom=dict(np.load(root/'nominal'/policy/'trace.npz'))
            t=tr['time'];n=len(t)
            paired=min(n,len(nom['time']))
            extra=np.full(n,np.nan)
            extra[:paired]=np.linalg.norm(tr['qpos'][:paired,:2]-center,axis=1)-np.linalg.norm(nom['qpos'][:paired,:2]-center,axis=1)
            series=state_series(tr,c)
            for ax,y in zip(row,[extra,np.rad2deg(tr['measurement'][:,0]-tr['reference_roll']),series['speed']]):
                ax.plot(t,y,color=color,ls=ls,label=policy)
        amplitude=(f"steer {c['disturbance_steer_rate']:+.2f} rad/s" if c['disturbance_steer_rate'] else f"COM lateral force {c['disturbance_force']:+.1f} N")
        for ax,title,units in zip(row,['Extra radial error vs own nominal','Roll tracking error','True longitudinal speed'],['m','deg','m/s']):
            ax.axvspan(c['disturbance_start'],c['disturbance_start']+c['disturbance_duration'],color='gray',alpha=.25)
            ax.set(xlim=(c['disturbance_start']-1,c['horizon_seconds']),xlabel='Time (s)',ylabel=units,title=f'{amplitude}\n{title}')
            ax.grid(alpha=.2)
        for sign in (-1,1):row[0].axhline(sign*decl['panel']['extra_tolerance_m'],color='black',ls=':',lw=.8)
        row[0].legend(fontsize=8)
    fig.suptitle('Turning disturbance recovery | CPU seed '+str(decl['panel']['seed'])+' | Frozen residual policy\nShading: event interval; dotted lines: extra radial tolerance. See report for joint recovery criteria.',fontsize=12)
    dest=root/'analysis';dest.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(dest/f'disturbance_comparison.{suffix}',dpi=160)
    plt.close(fig)
