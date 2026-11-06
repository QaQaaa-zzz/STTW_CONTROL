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
