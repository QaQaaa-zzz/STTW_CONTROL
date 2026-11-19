"""Screen a chapter-two approach speed on the frozen wide ditch and DEV roots."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_playground_task_reversal import write_rows
from run_playground_wide_gap_oracle import run, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    plan = json.loads((out/'plan.json').read_text())
    data = json.loads((args.domain/'group_0/screen.json').read_text())
    roots = json.loads((args.domain/'plan.json').read_text())['roots']
    h = data['ground_handoff']
    variants, labels, route_ids = [], [], []
    for seed in plan['roots']:
        y, yaw, roll = roots[str(seed)]
        for idx in plan['templates']:
            for speed in plan['approach_speeds_m_s']:
                variants.append(dict(h['handoff_variants'][idx], root_variant=seed,
                                     start_y_m=y, start_yaw_rad=yaw, start_roll_rad=roll,
                                     approach_speed_m_s=speed))
                labels.append(dict(root=seed, template=idx, approach_request_m_s=speed))
                route_ids.append(idx)
    with np.load(data['oracle_branch_reference']) as routes:
        np.savez_compressed(out/'routes.npz', s=routes['s'],
                            **{k:routes[k][route_ids] for k in ('xy','heading','curvature')})
    cfg = {k:v for k,v in data.items() if k not in ('initial_snapshot_path','oracle_branch_reference',
                                                    'oracle_initial_speed_m_s','oracle_initial_alpha')}
    cfg.update(ground_handoff=dict(h,reference_path=str(out/'routes.npz'),handoff_variants=variants),
               evaluation_lanes=[[i,'h75',i] for i in range(len(labels))],snapshot_loaded_far=False)
    rows = []
    for skill in plan['skills']:
        dst = out/skill
        results = run(cfg, dst, skill, 10, out)
        audits = json.loads((dst/'jump_validation.json').read_text())['cases']
        seen = {}
        for lane,(label,r,audit) in enumerate(zip(labels,results,audits,strict=True)):
            flight = any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in audit['events'])
            land = min((t for t in (audit['both_far_after_flight_s'],audit['solid_bank_landing_s']) if t is not None),default=None)
            hand = r['handoff_s']
            accepted = flight and land is not None and hand is not None and land<=hand+1e-5
            with np.load(dst/r['case_dir']/'trajectory.npz') as z:
                t=z['trace'][:,0]
                entry=r['rb_entry_s']
                k=np.argmin(abs(t-entry)) if entry is not None else None
                vx=None if k is None else float(z['qvel'][k,0])
                distinct=False
                if accepted:
                    key=(label['root'],label['approach_request_m_s'])
                    fingerprint=z['qpos'][:np.flatnonzero(z['trace'][:,-1]==2)[0]+1].tobytes()
                    histories=seen.setdefault(key,set())
                    distinct=fingerprint not in histories
                    histories.add(fingerprint)
                rows.append(dict(**label,lane=lane,skill=skill,entry_s=entry,entry_world_vx_m_s=vx,
                                 flight=flight,far_touch=audit['far_contact'],loaded_s=land,handoff_s=hand,
                                 qualified=accepted,distinct_qualified_prefix=distinct,goal=r['goal'],
                                 termination=r['termination'],end_s=float(t[-1])))
        write_rows(out/'cases.csv',rows)
    save(out/'terminal.json',dict(episodes=len(rows),optimizer_updates=0))


if __name__=='__main__':
    main()
