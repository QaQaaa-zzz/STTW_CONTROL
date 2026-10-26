"""Read-only activation and paired pre-intervention diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root,output):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    for name in ['dispatch_status.json','status.json','analysis/status.json']:
        if json.loads((root/name).read_text())['phase']!='completed':raise ValueError('completed run required')
    plan=json.loads((root/'plan.json').read_text());dt=plan['lower_dt_s']
    rows=json.loads((root/'analysis/results.json').read_text())
    checks=json.loads((root/'analysis/raw_audit.json').read_text());byid={x['case_id']:x for x in checks}
    index={(r['context'],r['alpha'],r['event_id'],r['role']):r for r in rows}
    keys=['context','alpha','event_id','force_N','pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s']
    records=[];checked={}
    for row in rows:
        if not row.get('goal_preview'):continue
        old=index[row['context'],row['alpha'],row['event_id'],row['source_role']]
        if any(row[k]!=old[k] for k in keys):raise ValueError('unmatched original control')
        files=[]
        for r in [row,old]:
            if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('raw changed')
            checked[r['raw_file']]=r['raw_sha256']
            with np.load(r['raw_file'],allow_pickle=False) as z:files.append({k:z[k] for k in z.files})
        new_raw,old_raw=files;effect=byid[row['case_id']]['goal_preview']
        first=effect['first_changed_target_s'];stop=round(first/dt) if first is not None else len(new_raw['active_tick'])
        common=np.asarray(new_raw['active_tick'],bool)&np.asarray(old_raw['active_tick'],bool)&(np.arange(len(new_raw['active_tick']))<stop)
        # Descriptive only. Existing accepted replay differences do not become
        # a new equality gate, and do not justify rerunning historical cases.
        delta={k:float(np.max(np.abs(new_raw[k][common]-old_raw[k][common]))) if np.any(common) else None
               for k in ['actual_xy','actual_forward_speed','phi','phi_dot','actual_delta','nominal_reference']}
        changed=row['goal_arrived']!=old['goal_arrived'] or row['physical_outcome']!=old['physical_outcome']
        records.append(dict(case_id=row['case_id'],source_case_id=old['case_id'],role=row['role'],
            disturbed=bool(row['force_N']),activation=effect,
            activated_after_pulse=(None if first is None or not row['force_N'] else first>=row['pulse_onset_task_tick']*dt+row['pulse_duration_s']),
            paired_outcome_changed=changed,arrival_new=row['goal_arrived'],arrival_old=old['goal_arrived'],
            distance_new=row['minimum_upright_goal_distance_m'],distance_old=old['minimum_upright_goal_distance_m'],
            cost_delta=row['scored_cost']-old['scored_cost'],prefix_common_ticks=int(common.sum()),prefix_max_abs_differences=delta))
    if len(records)!=plan['new_cases']:raise ValueError('complete declared new cases required')
    groups={}
    for role in sorted({r['role'] for r in records}):
        groups[role]={}
        for label,flag in [('disturbed',True),('nominal',False)]:
            group=[r for r in records if r['role']==role and r['disturbed']==flag]
            activated=[r for r in group if r['activation']['changed_target_ticks']>0]
            groups[role][label]=dict(total=len(group),activated=len(activated),
                changed_target_total_s=sum(r['activation']['changed_target_s'] for r in group),
                earliest_activation_s=min((r['activation']['first_changed_target_s'] for r in activated),default=None),
                changed_outcome_without_activation=sum(r['paired_outcome_changed'] and r not in activated for r in group),
                physical_exposure_precedes_all_activated=all(r['activated_after_pulse'] for r in activated) if flag else None)
    result=dict(groups=groups,records=records,authenticated_unique_raw=len(checked),
        source_results_sha256=digest(root/'analysis/results.json'),source_plan_sha256=digest(root/'plan.json'),
        script_sha256=digest(__file__),physics_steps=0,optimizer_updates=0,
        scope='Paired correlated DEV diagnostic. Target selection is after pulse, not evidence of improved intrinsic fall tolerance. '
              'Pre-target differences are reported, not a new exact-replay gate. No tests rerun, fitting or final-test claim.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();d=audit(a.run,a.output)
    print(json.dumps(d['groups'],ensure_ascii=False,indent=2))
