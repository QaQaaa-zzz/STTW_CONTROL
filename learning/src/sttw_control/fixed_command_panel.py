"""Frozen six historical command cases; postprocessing does not redefine reward."""
from pathlib import Path
import json
import numpy as np

def load_protocol():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/fixed_command_six.json').read_text())

def schedules(protocol):
    result=[]
    for name,commands in protocol['cases'].items():
        rows=np.zeros((16,3));rows[:,0]=99;rows[:len(commands)]=commands
        result.append((name,rows))
    return result

def report(root,spec,protocol):
    from .direct_command_reporting import _window,_step_csv,_plot_case,_plot_rewards,_plot_components
    out=Path(root)/'review';metrics={};lines=['# Fixed six command cases','',
        'B0 = same frozen lower with zero upper correction. Independent upper alpha0/1. Fixed seed77001,10s observation; original16s failure scoring unchanged. Raw-command windows, no reference resetting. Best selection recorded in ../manifest.json.','',
        '[Training diagnostics](training_diagnostics.png)','',
        '|Case|Method|speed RMSE [1,6) m/s|steer RMSE [1,6) rad|heading RMSE [9.5,10) rad|final speed/steer/heading hold|peak roll rad|failure|',
        '|---|---|---:|---:|---:|---|---:|---|']
    for case in protocol['cases']:
        traces={m:dict(np.load(out/case/f'{m}.npz')) for m in protocol['methods'] if (out/case/f'{m}.npz').exists()}
        if not traces:continue
        metrics[case]={}
        for m,d in traces.items():
            t=d['time'];failed=bool(np.any(d['physical_failure']));mask=(t>=9.5-1e-6)&(t<10-1e-6)
            complete=len(t)==2000 and not failed
            ev=d['actual_forward_speed']-d['limited_command'][:,0];ed=d['actual_delta']-d['limited_command'][:,1]
            tol=protocol['final_tolerances'];hold=[bool(complete and mask.sum()==100 and np.all(np.abs(x[mask])<=lim)) for x,lim in [(ev,tol['speed_m_s']),(ed,tol['steer_rad']),(d['e_psi_unwrapped'],tol['heading_rad'])]]
            cmd=_window(d,1,6,.005);tail=_window(d,9.5,10,.005)
            v=dict(command_window=cmd,final_window=tail,final_hold=hold,complete=complete,physical_failure=failed,observed_ticks=len(t),peak_roll=float(np.max(np.abs(d['phi']))),diagnostic_roll_over_0_3_s=float(np.sum(np.abs(d['phi'])>.3)*.005),working_roll_violation_s=float(np.sum(np.abs(d['phi'])>spec['limits']['working_roll_rad'])*.005))
            metrics[case][m]=v;_step_csv(out/case/f'{m}_steps.csv',d)
            fmt=lambda x:'missing' if x is None else f'{x:.4f}'
            lines.append(f"|{case}|{m}|{fmt(cmd['speed_rmse_m_s'] if cmd else None)}|{fmt(cmd['steer_rmse_rad'] if cmd else None)}|{fmt(tail['heading_rmse_rad'] if tail else None)}|{hold}|{v['peak_roll']:.4f}|{failed}|")
        b1=dict(np.load(out/case/'B0_alpha1.npz'))
        plot_command_references(out,case,traces,spec,protocol)
        _plot_case(out,case,traces,spec);_plot_rewards(out,case,traces,spec,b1)
        for m,d in traces.items():_plot_components(out,case,m,d,spec)
        lines.extend(['',f'[{case} physical comparison]({case}_comparison.png) · [original vs network commands]({case}_command_references.png) · [per-step and cumulative rewards]({case}_rewards.png)',''])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    lines+=['','No independent holdout or broad success-rate claim. Missing windows after failure are not zero error. Historical speed+steer holds do not establish heading recovery.']
    (out/'REPORT.md').write_text('\n'.join(lines))


def plot_command_references(out,case,traces,spec,protocol):
    """Raw issued command, network request, and actual reference sent to lower."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    baseline=traces.get('B0',next(iter(traces.values())))
    for j,ax in enumerate(axes):
        ax.plot(baseline['time'],baseline['target'][:,j],color='.65',ls='--',lw=1,label='original target before command slew')
        ax.plot(baseline['time'],baseline['limited_command'][:,j],color='black',lw=1.8,label='original issued command (reward reference)')
        for row in protocol['cases'][case][1:]:ax.axvline(row[0],color='.5',lw=.7,alpha=.5)
        ax.axvspan(9.5,10,color='green',alpha=.07,label='fixed final hold window')
    with (out/f'{case}_command_references.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['method','time_s','raw_speed','raw_steer','network_requested_speed','network_requested_steer','sent_speed','sent_steer','physical_failure'])
        for method,color in [('pi_alpha0','tab:blue'),('pi_alpha1','tab:orange')]:
            if method not in traces:continue
            d=traces[method];a=np.tanh(d['latent_z']);cfg=spec['action'];delta=np.stack([a[:,0]*np.where(a[:,0]>=0,cfg['speed_positive_scale_m_s'],cfg['speed_negative_scale_m_s']),a[:,1]*cfg['steer_scale_rad']],axis=-1)
            request=d['limited_command']+delta
            np.testing.assert_allclose(d['governed'],d['limited_command']+d['offsets'],atol=1e-6)
            alpha=0 if method=='pi_alpha0' else 1;update=int(d.get('checkpoint_update',-1))
            for j,ax in enumerate(axes):
                ax.plot(d['time'],request[:,j],color=color,ls=':',lw=.8,alpha=.55,label=f'alpha{alpha} network request before guards')
                ax.plot(d['time'],d['governed'][:,j],color=color,lw=1.5,label=f'alpha{alpha} reference sent to lower (update{update})')
                if np.any(d['physical_failure']):ax.axvline(d['time'][-1],color=color,ls='--');ax.plot(d['time'][-1],d['governed'][-1,j],marker='x',color=color)
            for i,t in enumerate(d['time']):writer.writerow([method,float(t),*d['limited_command'][i],*request[i],*d['governed'][i],bool(d['physical_failure'][i])])
    for ax,label in zip(axes,['Speed reference (m/s)','Steer reference (rad)']):ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best')
    axes[-1].set_xlabel('Time (s)');fig.suptitle(f'{case} | original vs upper-network commands | lower_alpha=1, seed77001');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'{case}_command_references.{ext}',dpi=140)
    plt.close(fig)
