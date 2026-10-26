"""Read-only stage-mechanism recovery-gap audit, not a deployable controller."""
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    if set(a) != set(b):
        raise ValueError('paired full event identities required')
    d = [a[k]['minimum_upright_goal_distance_m']-b[k]['minimum_upright_goal_distance_m'] for k in a]
    return dict(n=len(a), arrival_gain=sum(a[k]['minimum_upright_goal_distance_m']<=1<b[k]['minimum_upright_goal_distance_m'] for k in a),
        arrival_loss=sum(b[k]['minimum_upright_goal_distance_m']<=1<a[k]['minimum_upright_goal_distance_m'] for k in a),
        survival_gain=sum(a[k]['physical_outcome']=='survived' and b[k]['physical_outcome']!='survived' for k in a),
        survival_loss=sum(b[k]['physical_outcome']=='survived' and a[k]['physical_outcome']!='survived' for k in a),
        median_minimum_distance_change_m=statistics.median(d), improved_at_least_1m=sum(v<=-1 for v in d), worsened_at_least_1m=sum(v>=1 for v in d),
        baseline_over4m_to_candidate_within4m=sum(b[k]['minimum_upright_goal_distance_m']>4>=a[k]['minimum_upright_goal_distance_m'] for k in a),
        baseline_within4m_to_candidate_over4m=sum(a[k]['minimum_upright_goal_distance_m']>4>=b[k]['minimum_upright_goal_distance_m'] for k in a))


def main(root):
    out=root/'runs/same_state_reference_pause_v1/relaxed_goal_stage_review_v1'
    rows={}
    sources={}
    for name in ['entry_history_confirmation_v1','steering_stage_mechanism_v1']:
        base=root/'runs'/name
        if json.loads((base/'analysis/status.json').read_text())['phase']!='completed':
            raise ValueError('completed original audit required')
        p=base/'analysis/results.json';sources[name]=dict(file=str(p),sha256=digest(p))
        for r in json.loads(p.read_text()):
            key=(r['context'],r['alpha'],r['event_id'],r['role'])
            if key in rows and rows[key]['raw_sha256']!=r['raw_sha256']:
                raise ValueError('overlapping cached trajectory changed')
            rows[key]=r
    roles=['goal_zero_feedback','goal_full_feedback','goal_steer_feedback','neural_before_only','neural_after_only']
    banks={role:{k[:3]:r for k,r in rows.items() if r['role']==role and r['force_N']} for role in roles}
    if any(len(b)!=144 for b in banks.values()):raise ValueError('complete144 per role required')
    for k in banks[roles[0]]:
        for field in ['force_N','pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s','feedback_gain_rad','causal_support']:
            if any(b[k][field]!=banks[roles[0]][k][field] for b in banks.values()):raise ValueError('unmatched event or reference permission')
    pairs={role:compare(banks['neural_after_only'],banks[role]) for role in roles if role!='neural_after_only'}
    plan=json.loads((root/'runs/entry_history_confirmation_v1/plan.json').read_text())
    checks=[]
    # Re-read every after/full disturbed raw, rather than only new successes.
    for role in ['goal_full_feedback','neural_after_only']:
        for k,r in banks[role].items():
            rawfile=Path(r['raw_file'])
            if digest(rawfile)!=r['raw_sha256']:raise ValueError('raw identity changed')
            with np.load(plan['contexts'][r['context']]['path']) as path:
                goal=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,j]) for j in range(2)])
            with np.load(rawfile) as z:
                raw={field:z[field] for field in ['active_tick','physical_tick','actual_xy','contact_failure','nonfinite','lower_fault','policy_fault']}
            m=goal_arrival_metrics(raw,goal_xy=goal,radius_m=1.,dt=.005,horizon_s=10.)
            if abs(m['minimum_upright_goal_distance_m']-r['minimum_upright_goal_distance_m'])>1e-8:raise ValueError('original goal distance mismatch')
            active=raw['active_tick'].astype(bool)
            xy=raw['actual_xy'][active];distance=np.linalg.norm(xy-goal,axis=1)
            tick=raw['physical_tick'][active]
            tail=distance[tick>=1800]  # Fixed last 1s of original 10s, no extension.
            checks.append(dict(case_id=r['case_id'],role=role,alpha=r['alpha'],physical_outcome=r['physical_outcome'],raw_sha256=r['raw_sha256'],
                minimum_goal_distance_m=m['minimum_upright_goal_distance_m'],arrived_1m=m['goal_arrived'],first_arrival_s=m['first_arrival_s'],
                full_10s_observed=len(tick)==2000,final_distance_m=float(distance[-1]),
                last1s_radial_progress_m=float(tail[0]-tail[-1]) if len(tail)==200 else None,
                warning='Positive last1s progress is descriptive; not proof arrival would occur with extra time. Contact endings do not supply a counterfactual 10s tail.'))
    result=dict(sources=sources,paired_after_only_vs=pairs,raw_rechecked_cases=len(checks),details=checks,
        scope='Exploratory shared-ancestor DEV; privileged exact-onset switch. Same raw duplicates deduplicated by context/alpha/event/role and SHA. No new physics/training, alpha reassignment, extra observation time or controller adopted. after vs full also changes neural speed channel; after vs full-time steer isolates this timing policy, not instantaneous reset of filtered references.',new_physics_steps=0,optimizer_updates=0)
    (out/'stage_gap.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'paired':pairs,'raw_rechecked':len(checks)},ensure_ascii=False))


if __name__=='__main__':main(Path(__file__).resolve().parents[2])
