"""Audit fixed existing ESO threshold timing; never fit or actuate a gate."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sttw_control.causal_reference_baseline import THRESHOLD


def first_legal_latch(raw):
    """Existing sensor/permission only: no force, onset or outcome required."""
    active=raw['active_tick'].astype(bool);tick=raw['physical_tick']
    decision=active&(tick%4==0)&raw['actor_action_valid'].astype(bool)
    valid=decision&raw['probe_eso_history_valid'].astype(bool)
    crossings=np.flatnonzero(valid&(np.abs(raw['probe_eso_signal'])>=THRESHOLD))
    return (int(tick[crossings[0]]) if len(crossings) else None,
            int(valid.sum()),int(decision.sum()),len(crossings))


def review(root):
    root=Path(root).resolve()
    for file in ['status.json','analysis/status.json']:
        if json.loads((root/file).read_text())['phase']!='completed':
            raise ValueError('complete raw audit required')
    plan=json.loads((root/'plan.json').read_text())
    if plan['experiment']!='passive_sensor_probe_v1' or THRESHOLD!=.015:
        raise ValueError('fixed passive probe and original threshold required')
    rows=json.loads((root/'analysis/results.json').read_text())
    if len(rows)!=60 or any(r['alpha']!=.5 or r['role']!='goal_zero_feedback' for r in rows):
        raise ValueError('complete original60 zero-neural cases required')
    records=[]
    for row in rows:
        with np.load(row['raw_file'],allow_pickle=False) as z:
            raw={k:z[k] for k in ['active_tick','actor_action_valid','physical_tick',
                'probe_eso_latest','probe_eso_previous','probe_eso_signal',
                'probe_eso_history_valid','gate_detected','gate_override','applied_force_impulse']}
        active=raw['active_tick'].astype(bool);tick=raw['physical_tick']
        np.testing.assert_allclose(raw['probe_eso_signal'][active],
            -(raw['probe_eso_latest'][active]-raw['probe_eso_previous'][active]),atol=1e-7)
        if raw['gate_detected'][active].any() or raw['gate_override'][active].any():
            raise ValueError('passive source must not actuate ESO gate')
        first_cross,valid_decisions,actor_decisions,crossings=first_legal_latch(raw)
        force=np.flatnonzero(active&np.any(raw['applied_force_impulse']!=0,axis=-1))
        first_force=int(tick[force[0]]) if len(force) else None
        dt=plan['lower_dt_s']
        delay=(first_cross-first_force)*dt if first_force is not None and first_cross is not None else None
        records.append(dict(case_id=row['case_id'],context=row['context'],force_N=row['force_N'],
            alpha=row['alpha'],first_measured_force_tick=first_force,first_fixed_threshold_cross_tick=first_cross,
            first_latch_delay_from_force_s=delay,valid_decisions=valid_decisions,
            actor_decisions=actor_decisions,threshold_crossing_decisions=crossings,
            original_planned_onset_tick=row['pulse_onset_task_tick'],physical_outcome=row['physical_outcome'],
            goal_arrived=row['goal_arrived'],raw_sha256=row['raw_sha256']))
    nominal=[r for r in records if not r['force_N']];forced=[r for r in records if r['force_N']]
    summary=dict(nominal_cases=len(nominal),unforced_first_latch=sum(r['first_fixed_threshold_cross_tick'] is not None for r in nominal),
        disturbed_cases=len(forced),first_latch_before_force=sum(r['first_latch_delay_from_force_s'] is not None and r['first_latch_delay_from_force_s']<0 for r in forced),
        first_latch_at_or_after_force=sum(r['first_latch_delay_from_force_s'] is not None and r['first_latch_delay_from_force_s']>=0 for r in forced),
        never_latched=sum(r['first_fixed_threshold_cross_tick'] is None for r in forced))
    result=dict(summary=summary,records=records,threshold=THRESHOLD,physics_steps=0,optimizer_updates=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Original fixed normalized ESO threshold0.015 (history scale0.3rad), evaluated once as a first-event latch using current/past legal actor observations. Original actor permission retained. No threshold fitting/reset, new control/gate actuation or simulated counterfactual improvement. Upcoming force and onset only retrospective labels. Raw-source outcomes describe the unchanged baseline, not a switched controller. Seen correlated DEV, only alpha0.5, no independent repeat or deployment validation.')
    out=root/'analysis/passive_timing_review.json'
    if out.exists():raise ValueError('preserve existing timing review')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    review(p.parse_args().run)
