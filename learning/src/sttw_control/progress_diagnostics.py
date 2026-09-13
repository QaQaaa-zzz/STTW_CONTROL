"""Recorded circle progress; geometric schedule lag is not an arrival-time claim."""
from pathlib import Path
import json
import numpy as np


def circle_progress(trace, circle):
    t=np.asarray(trace['time'])
    xy=np.asarray(trace['pose'])[:,:2]-[circle['center_x'],circle['center_y']]
    angle=np.unwrap(np.arctan2(xy[:,1],xy[:,0]))
    progress=circle['direction']*circle['radius']*(angle-angle[0])
    reference=np.asarray(trace['motion_command'])[:,1]
    if np.any(reference<=0) or not np.allclose(reference,reference[0]):
        raise ValueError('schedule lag currently requires constant positive target speed')
    target=reference[0]*(t-t[0])
    return dict(time=t,progress_m=progress,target_progress_m=target,
                schedule_lag_s=(target-progress)/reference[0],
                radial_error_m=np.linalg.norm(xy,axis=1)-circle['radius'])


def generate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'analysis/progress';out.mkdir(parents=True,exist_ok=True)
    rows=[];links=[]
    for panel in sorted((root/'evaluation').glob('alpha_*/seed_*')):
        for case in sorted(panel.glob('*/residual')):
            dest=out/panel.parent.name/panel.name/case.parent.name;dest.mkdir(parents=True,exist_ok=True)
            fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
            for policy in ('baseline','residual'):
                path=case.parent/policy
                c=json.loads((path/'declaration.json').read_text())['config']
                if c.get('circle') is None or c.get('figure_eight') is not None:
                    plt.close(fig)
                    raise ValueError('progress diagnostic requires a circle; figure-eight projection not implemented')
                tr=dict(np.load(path/'trace.npz'));s=circle_progress(tr,c['circle'])
                np.savez_compressed(dest/f'{policy}.npz',**s)
                failed=bool(tr['terminated'][-1]);event=json.loads((path/'event.json').read_text())
                start=event['start_seconds'] if case.parent.name!='nominal' else s['time'][0]
                mask=s['time']>=start
                y=s['radial_error_m'][mask]
                width=float(max(0.,y.max())-min(0.,y.min())) if len(y) else None
                rows.append(dict(alpha=float(tr['priority_alpha'][0]),seed=panel.name,scenario=case.parent.name,policy=policy,
                    duration_s=float(s['time'][-1]-s['time'][0]),failed=failed,
                    final_progress_m=float(s['progress_m'][-1]),final_schedule_lag_s=float(s['schedule_lag_s'][-1]),
                    corridor_width_m=width,space_window_start_s=float(start),
                    scope='Development: circle projection, root-point space, signed schedule lag; no body-envelope or arrival-time guarantee'))
                for ax,key in zip(axes,('progress_m','schedule_lag_s','radial_error_m')):
                    ax.plot(s['time'],s[key],label=policy)
                    if failed:ax.plot(s['time'][-1],s[key][-1],'x')
                if policy=='baseline':axes[0].plot(s['time'],s['target_progress_m'],'k--',label='target schedule')
            for ax,title in zip(axes,('Reference-path progress [m]','Signed schedule lag [s]','Outward radial error [m]')):
                ax.set_title(title);ax.set_xlabel('Time [s]');ax.grid(alpha=.2);ax.legend()
                if case.parent.name!='nominal':ax.axvspan(event['start_seconds'],event['end_seconds'],color='grey',alpha=.18)
            fig.suptitle(f"{panel.parent.name} | {panel.name} | {case.parent.name}; x: failure; no extrapolation")
            for ext in ('png','pdf'):fig.savefig(dest/f'progress.{ext}',dpi=140)
            plt.close(fig)
            links.append(f'- {panel.parent.name}/{panel.name}/{case.parent.name}: [plot]({(dest / "progress.png").relative_to(out)})')
    (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    (out/'INDEX.md').write_text('# Paired circle progress diagnostics\n\nSigned schedule lag = (target progress - projected progress) / target speed. Positive means behind schedule. Compare equal complete horizons; failures are censored. Space covers event start to recorded end (nominal: full episode), root point only. No corridor pass/fail threshold fitted to these results.\n\n'+'\n'.join(links)+'\n')
    return rows
