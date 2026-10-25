"""Read-only phase/permission/saturation audit; temporal association, not causation."""
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import digest,atomic_json


def main():
    root=Path('runs/causal_recovery_relay_v1').resolve();source=root/'analysis/results.json'
    rows=json.loads(source.read_text());cfg=json.loads((root/'common_environment_config.json').read_text())
    interface=cfg['residual_interface'];out=[];pairing={}
    for row in rows:
        assert digest(row['raw_file'])==row['raw_sha256']
        with np.load(row['raw_file'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        active=raw['active_tick'].astype(bool);t=raw['physical_tick']*.005
        gate=raw.get('relay_active',np.zeros_like(active)).astype(bool)&active
        release=np.flatnonzero(~gate&np.r_[False,gate[:-1]]&active)
        last=None if not len(release) else int(release[-1]);trigger=row['first_trigger_tick']
        windows={'pre_event':active&(t<row['pulse_onset_task_tick']*.005),
            'after_event':active&(t>=row['pulse_onset_task_tick']*.005),'last_second':active&(t>=9)}
        if last is not None:
            windows.update(first_second_after_release=active&(t>=last*.005)&(t<last*.005+1),
                later_after_release=active&(t>=last*.005+1),post_release=active&(t>=last*.005))
        metrics={}
        for name,mask in windows.items():
            if not np.any(mask):continue
            ey=raw['path_cross_track'][mask];heading=raw['path_heading_error'][mask]
            costs=raw['effective_costs'][mask].sum(0,dtype=np.float64)*.005*cfg['path_reward']['scale']
            ref=raw['reference'][mask];legal=raw['actor_action_valid'][mask].astype(bool)
            factor=1-(1-cfg['path_reward']['path_secondary_ratio'])*row['alpha']
            path_cap=factor*cfg['path_reward']['component_caps']['path']
            metrics[name]=dict(observed_s=float(mask.sum()*.005),path_rmse_m=float(np.sqrt(np.mean(ey**2))),
                heading_rmse_rad=float(np.sqrt(np.mean(heading**2))),actor_legal_fraction=float(np.mean(legal)),
                current_invalid_fraction=float(np.mean(raw['domain_invalid'][mask])),
                speed_reference_at_bound_fraction=float(np.mean((ref[:,0]<=interface['speed_reference_bounds_m_s'][0]+1e-5)|(ref[:,0]>=interface['speed_reference_bounds_m_s'][1]-1e-5))),
                steering_reference_at_bound_fraction=float(np.mean(abs(ref[:,1])>=interface['steer_reference_bound_rad']-1e-5)),
                path_cost_cap_fraction=float(np.mean(raw['raw_costs'][mask,1]>path_cap)),
                mean_turn_gate_chi=float(np.mean(raw['chi'][mask])),
                uncapped_path_cost_integral=float(raw['raw_costs'][mask,1].sum(dtype=np.float64)*.005*cfg['path_reward']['scale']),
                running_cost=float(costs.sum()),component_integrals=costs.tolist())
        after=active&(t>=row['pulse_onset_task_tick']*.005)
        invalid=np.flatnonzero(after&raw['domain_invalid'].astype(bool))
        peak=int(np.flatnonzero(after)[np.argmax(abs(raw['path_cross_track'][after]))]) if np.any(after) else None
        summary=dict(case_id=row['case_id'],mode=row['mode'],alpha=row['alpha'],event_id=row['event_id'],force_N=row['force_N'],
            category=row['category'],goal_crossed=row['goal_crossed_while_task_valid'],final_hold=row['final_hold'],
            trigger_s=None if trigger is None else trigger*.005,last_release_s=None if last is None else last*.005,
            first_after_event_invalid_s=None if not len(invalid) else int(invalid[0])*.005,
            peak_after_event_path_error_s=None if peak is None else peak*.005,
            peak_after_event_path_error_m=None if peak is None else float(abs(raw['path_cross_track'][peak])),windows=metrics)
        out.append(summary);pairing.setdefault((row['alpha'],row['event_id']),{})[row['mode']]=row
    differences=[]
    for (alpha,event),group in pairing.items():
        x,y=group['causal_B'],group['causal_support']
        differences.append(dict(alpha=alpha,event_id=event,force_N=x['force_N'],B_minus_rule_cost=x['scored_cost']-y['scored_cost'],
            B_minus_rule_path_rmse_m=x['task_path_rmse_m']-y['task_path_rmse_m'],B_minus_rule_speed_rmse_m_s=x['task_speed_rmse_m_s']-y['task_speed_rmse_m_s'],
            same_physical_outcome=x['physical_outcome']==y['physical_outcome'],same_task=x['task_recovered']==y['task_recovered']))
    atomic_json(root/'analysis/post_intervention_audit.json',dict(source_sha256=digest(source),physics_steps=0,optimizer_updates=0,cases=out,paired_continuous_differences=differences,
        interpretation='Facts/temporal associations on selected DEV; post-release geometry/fallback may be causes or consequences. No new thresholds or counterfactual control.'))
    for x in out:
        if x['mode'] in ['causal_B','causal_support'] and x['force_N']:
            w=x['windows'];print(x['mode'],x['alpha'],x['event_id'],'goal',x['goal_crossed'],'last_release',x['last_release_s'],'postlegal',round(w['post_release']['actor_legal_fraction'],3),'later_cost_fraction',round(w.get('later_after_release',{}).get('running_cost',0)/sum(x['windows'][n]['running_cost'] for n in ['pre_event','after_event']),3))


if __name__=='__main__':main()
