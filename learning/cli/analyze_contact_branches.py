"""Clearance-state branch outcomes and explicit center-path feasibility choices."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--center-error-limit-m',type=float,default=.4);a=p.parse_args()
c=json.loads((a.run/'contract.json').read_text());s=c['settings'];forks=json.loads((a.run/'contact_forks.json').read_text());status=json.loads((a.run/'contact_branch_status.json').read_text())
with np.load(a.run/'trajectory.npz') as d:
    pose=d['pose'];measurement=d['measurement'];alive=d['alive'].astype(bool);trace=d['lower_trace'].reshape(len(pose),s['num_envs'],11)
rows=[]
for fork in forks:
    group=s['contact_branching']['groups'][fork['group']];source=fork['source_lane'];branches=fork['branch_lanes'];start=fork['policy_step']*4;ids=[source]+branches
    shared=alive[start:,ids].all(1);results=[]
    for lane,alpha in zip(ids,[fork['entry_alpha']]+fork['target_alphas']):
        valid=alive[start:,lane];cross=trace[start:,lane,3];speed=trace[start:,lane,2]-group['speed_m_s'];roll=measurement[start:,lane,0]
        peak=float(abs(cross[valid]).max()) if valid.any() else None
        full=bool(valid.all());feasible=full and peak is not None and peak<=a.center_error_limit_m
        results.append(dict(lane=lane,alpha=alpha,unreset_source=lane==source,full_remaining_window_alive=full,observed_after_fork_s=float(valid.sum()*c['lower_dt']),path_rmse_m=float(np.sqrt(np.mean(cross[valid]**2))) if valid.any() else None,path_peak_abs_m=peak,roll_peak_abs_rad=float(abs(roll[valid]).max()) if valid.any() else None,speed_rmse_m_s=float(np.sqrt(np.mean(speed[valid]**2))) if valid.any() else None,feasible_center_path=feasible,shared_window_path_rmse_m=float(np.sqrt(np.mean(cross[shared]**2))) if shared.any() else None,shared_window_speed_rmse_m_s=float(np.sqrt(np.mean(speed[shared]**2))) if shared.any() else None))
    same=branches[fork['target_alphas'].index(fork['entry_alpha'])];mask=alive[start:,source]&alive[start:,same]
    restoration=dict(shared_s=float(mask.sum()*c['lower_dt']),xy_rms_m=float(np.sqrt(np.mean(np.sum((pose[start:,same,:2][mask]-pose[start:,source,:2][mask])**2,axis=1)))) if mask.any() else None,roll_difference_peak_rad=float(abs(measurement[start:,same,0][mask]-measurement[start:,source,0][mask]).max()) if mask.any() else None)
    candidates=[r for r in results[1:] if r['feasible_center_path']]
    selected=min(candidates,key=lambda r:r['speed_rmse_m_s']) if candidates else None
    # Empty sets stay empty; a diagnostic fallback is not a feasible selection.
    fallback=min(results[1:],key=lambda r:(not r['full_remaining_window_alive'],max(0,(r['path_peak_abs_m'] if r['path_peak_abs_m'] is not None else float('inf'))-a.center_error_limit_m),r['speed_rmse_m_s'] if r['speed_rmse_m_s'] is not None else float('inf')))
    features=dict(speed_m_s=float(trace[start-1,source,2]),roll_rad=float(measurement[start-1,source,0]),roll_rate_rad_s=float(measurement[start-1,source,1]),entry_alpha=fork['entry_alpha'],center_error_limit_m=a.center_error_limit_m,front_vertical_load_N=fork['contact_features_oracle']['front_vertical_load_N'])
    rows.append(dict(group=fork['group'],seed=group['seed'],speed_m_s=group['speed_m_s'],entry_alpha=fork['entry_alpha'],fork_time_s=fork['time_s'],event=status['event'],ramp_s=status['ramp_s'],shared_remaining_s=float(shared.sum()*c['lower_dt']),restoration_control=restoration,base_features=features,contact_features_oracle=fork['contact_features_oracle'],outcomes=results,feasible_alphas=[r['alpha'] for r in candidates],selected_alpha=selected['alpha'] if selected else None,empty_feasible_set=not bool(candidates),diagnostic_fallback_alpha=fallback['alpha']))
report=dict(role='DEV same-clearance-state branches; center-error and survival constraint, not arrival or vehicle swept-envelope certification; selected alpha is retrospective label, not deployed predictor',selection='among surviving branches with peak absolute center-path error within declared limit, minimize requested-speed RMSE; preserve empty feasible sets',center_error_limit_m=a.center_error_limit_m,branch_status=status,groups=rows)
(a.run/'contact_branch_outcomes.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(groups=len(rows),empty_feasible_sets=sum(r['empty_feasible_set'] for r in rows))))

# Full paired trajectories, one condition per page; no aggregation or smoothing.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figures=a.run/'branch_figures';figures.mkdir(exist_ok=True)
route=np.load(Path(c['bundle'])/'route.npz')['xy'];palette={0:'#0072B2',.25:'#D55E00',1:'#009E73'};styles={0:'-',.25:'--',1:'-.'};time=(np.arange(len(pose))+1)*c['lower_dt'];manifest=[]
for fork,row in zip(forks,rows):
    start=fork['policy_step']*4;source=fork['source_lane'];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');xy,roll,ev,ey=axes.flat
    xy.plot(*route.T,':',color='.5',label='Original reference')
    extent=[]
    for lane,alpha in zip([source]+fork['branch_lanes'],[fork['entry_alpha']]+fork['target_alphas']):
        valid=alive[:,lane].copy()
        if lane!=source:valid[:start]=False
        label='Unreset entry continuation' if lane==source else f'Recovery alpha={alpha:g}'
        color='.25' if lane==source else palette[alpha];style=':' if lane==source else styles[alpha]
        points=np.where(valid[:,None],pose[:,lane,:2],np.nan);xy.plot(*points.T,color=color,ls=style,label=label,lw=1.2)
        indices=np.flatnonzero(valid)
        if len(indices):
            xy.plot(*pose[indices[-1],lane,:2],marker='x',color=color);extent.append(pose[valid,lane,:2])
        roll.plot(time,np.where(valid,np.rad2deg(measurement[:,lane,0]),np.nan),color=color,ls=style)
        ev.plot(time,np.where(valid,trace[:,lane,2]-row['speed_m_s'],np.nan),color=color,ls=style)
        ey.plot(time,np.where(valid,trace[:,lane,3],np.nan),color=color,ls=style)
    points=np.concatenate(extent);xy.set_xlim(points[:,0].min()-.3,points[:,0].max()+.3);xy.set_ylim(points[:,1].min()-.3,points[:,1].max()+.3)
    xy.plot(*pose[start-1,source,:2],marker='o',mfc='none',mec='black',label='Clearance fork')
    xy.set(xlabel='World x (m)',ylabel='World y (m)');xy.set_aspect('equal',adjustable='box');xy.legend(fontsize=8)
    roll.set(ylabel='Roll (deg)');ev.set(ylabel='Speed error (m/s)');ey.set(ylabel='Signed path error (m)')
    ey.axhline(a.center_error_limit_m,color='.5',ls=':',lw=.8);ey.axhline(-a.center_error_limit_m,color='.5',ls=':',lw=.8)
    for ax in [roll,ev,ey]:ax.axvline(fork['time_s'],color='.5',ls=':');ax.set_xlabel('Time (s)')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(f"Frozen H75 + original Local300 | DEV | seed{row['seed']} | {row['speed_m_s']} m/s | entry alpha={row['entry_alpha']:g}\n{status['event']}, ramp={status['ramp_s']}s; crosses: last observed endpoints; branch curves start at fork")
    filename=f"group_{row['group']:03d}.png";fig.savefig(figures/filename,dpi=140);plt.close(fig)
    manifest.append(dict(group=row['group'],figure=filename,source='trajectory.npz and contact_forks.json; original route from contract bundle',description='Actual/reference XY, roll, speed error, signed path error; all three recovery alphas and unreset source; no smoothing; gaps and true observed endpoints retained'))
(figures/'index.json').write_text(json.dumps(manifest,indent=2)+'\n')
