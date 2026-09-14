"""Paired speed-only disturbance diagnostics, independent of path recovery."""
import numpy as np


def paired_speed(time,speed,nominal_time,nominal_speed,*,start,end,target,failed=False,tolerance=.05,hold=.5):
    t=np.asarray(time);v=np.asarray(speed);nt=np.asarray(nominal_time);nv=np.asarray(nominal_speed)
    if len(t)<2 or len(nt)<len(t) or not np.allclose(t,nt[:len(t)],rtol=0,atol=1e-7):
        raise ValueError('paired timestamps unavailable; no extrapolation')
    dt=float(t[1]-t[0])
    if not np.allclose(np.diff(t),dt,rtol=1e-5,atol=1e-8):raise ValueError('nonuniform time')
    dv=v-nv[:len(t)];window=t>=start-1e-8;post=t>=end-1e-8
    deficit=np.maximum(-dv,0.);surplus=np.maximum(dv,0.)
    eligible=bool(np.any(window&(np.abs(dv)>tolerance)))
    valid=(np.abs(dv)<=tolerance)&(np.abs(v-target)<=.2)
    needed=int(np.ceil(hold/dt));settled=None
    if eligible and not failed and post.any():
        begin=int(np.flatnonzero(post)[0]);bad=np.flatnonzero(post&~valid)
        begin=max(begin,int(bad[-1])+1 if len(bad) else begin)
        finish=begin+needed
        if finish<len(t):settled=float(t[finish]-end)
    return dict(extra_speed=dv,deficit=deficit,surplus=surplus,summary=dict(
        event_reached=bool(window.any()),speed_excursion=eligible,failed=bool(failed),
        observed_end_s=float(t[-1]),maximum_extra_drop_m_s=float(deficit[window].max()) if window.any() else None,
        maximum_extra_rise_m_s=float(surplus[window].max()) if window.any() else None,
        surplus_integral_m=float(np.trapezoid(surplus[window],t[window])) if window.any() else None,
        absolute_extra_speed_integral_m=float(np.trapezoid(np.abs(dv[window]),t[window])) if window.any() else None,
        deficit_integral_m=float(np.trapezoid(deficit[window],t[window])) if window.any() else None,
        speed_settling_after_event_s=settled,
        final_speed_band_held=bool(not failed and post.any() and t[-1]-end>=hold and np.all(valid[t>=t[-1]-hold])),
        scope='Speed-only: extra error <=0.05 m/s and true target error <=0.2 m/s; 0.5s hold; no path/safety guarantee'))


def generate(root):
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .media import state_series
    root=Path(root);out=root/'analysis/speed_recovery';out.mkdir(parents=True,exist_ok=True)
    rows=[];links=[]
    for panel in sorted((root/'evaluation').glob('alpha_*/seed_*')):
        for case in sorted(panel.glob('*/residual')):
            if case.parent.name=='nominal':continue
            dest=out/panel.parent.name/panel.name/case.parent.name;dest.mkdir(parents=True,exist_ok=True)
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for policy in ('baseline','residual'):
                p=case.parent/policy;tr=dict(np.load(p/'trace.npz'));nom=dict(np.load(panel/'nominal'/policy/'trace.npz'))
                c=json.loads((p/'declaration.json').read_text())['config'];v=state_series(tr,c)['speed'];nv=state_series(nom,c)['speed']
                x=paired_speed(tr['time'],v,nom['time'],nv,start=c['disturbance_start'],end=c['disturbance_start']+c['disturbance_duration'],target=c['speed_reference'],failed=bool(tr['terminated'][-1]))
                rows.append(dict(alpha=float(tr['priority_alpha'][0]),seed=panel.name,scenario=case.parent.name,policy=policy,**x['summary']))
                np.savez_compressed(dest/f'{policy}.npz',time=tr['time'],speed=v,nominal_speed=nv[:len(v)],extra_speed=x['extra_speed'],deficit=x['deficit'],surplus=x['surplus'])
                axes[0].plot(tr['time'],x['extra_speed'],label=policy)
                dt=np.diff(tr['time']);d=x['deficit'];mask=tr['time']>=c['disturbance_start']-1e-8
                cumulative=np.r_[0,np.cumsum((d[1:]+d[:-1])*dt/2*(mask[1:]&mask[:-1]))]
                axes[1].plot(tr['time'],cumulative,label=policy+' deficit')
                rise=x['surplus'];positive=np.r_[0,np.cumsum((rise[1:]+rise[:-1])*dt/2*(mask[1:]&mask[:-1]))]
                axes[1].plot(tr['time'],positive,ls='--',label=policy+' surplus')
                if tr['terminated'][-1]:axes[0].plot(tr['time'][-1],x['extra_speed'][-1],'x')
            for ax,title in zip(axes,('Speed minus own nominal [m/s]','Speed deficit / surplus integrals [m]')):
                ax.set_title(title);ax.set_xlabel('Time [s]');ax.grid(alpha=.2);ax.legend();ax.axvspan(c['disturbance_start'],c['disturbance_start']+c['disturbance_duration'],color='grey',alpha=.2)
            axes[0].axhline(.05,color='grey',ls=':');axes[0].axhline(-.05,color='grey',ls=':')
            fig.suptitle(f'{panel.parent.name} | {panel.name} | {case.parent.name}; paired own-nominal diagnostic')
            for ext in ('png','pdf'):fig.savefig(dest/f'comparison.{ext}',dpi=140)
            plt.close(fig);links.append(f'- [{panel.parent.name}/{panel.name}/{case.parent.name}]({(dest/"comparison.png").relative_to(out)})')
    (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    (out/'INDEX.md').write_text('# Paired speed disturbance response\n\nNo excursion gives no recovery time; null does not automatically mean failure. Deficit is lost speed integral, not projected path progress.\n\n'+'\n'.join(links)+'\n')
    return rows
