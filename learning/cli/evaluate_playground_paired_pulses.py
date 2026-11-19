"""Apply the same measured lateral pulse to each frozen jump skill."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_playground_task_reversal import write_rows
from run_playground_wide_gap_oracle import run, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve()
    plan=json.loads((out/'plan.json').read_text())
    lane=plan['lane']
    rows=[]
    for label,start in plan['windows'].items():
        for skill in plan['skills']:
            base=Path(plan['input'])/('055_'+skill)
            cfg=json.loads((base/'batch.json').read_text())
            pulse=dict(lane=lane,start_tick=start,stop_tick=start+plan['ticks'],force_n=plan['force_n'])
            case=out/f'{label}_{skill}'
            results=run(dict(cfg,world_y_force_pulse=pulse),case,skill,10,out)
            result=results[lane]
            audit=json.loads((case/'jump_validation.json').read_text())['cases'][lane]
            receipts=[json.loads(line) for line in (case/'applied_pulse.jsonl').read_text().splitlines()]
            assert [(r['tick'],r['lane'],r['world_y_force_n']) for r in receipts]==[(start,lane,plan['force_n']),(pulse['stop_tick'],lane,0.)]
            path=Path(result['case_dir'])/'trajectory.npz'
            with np.load(case/path) as z,np.load(base/'batch'/path) as original:
                t=z['trace'][:,0]
                k=np.argmin(abs(t-start*.005))
                delta=float(np.max(abs(z['qpos'][:k+1]-original['qpos'][:k+1])))
                active=(t>start*.005+1e-6)&(t<=pulse['stop_tick']*.005+1e-6)
                normal=z['contact_substep'][active,:,:3].sum(axis=-1)
            loads=[v for v in (audit['solid_bank_landing_s'],audit['both_far_after_flight_s']) if v is not None]
            hand=result['handoff_s']
            rows.append(dict(window=label,skill=skill,lane=lane,force_n=plan['force_n'],
                impulse_ns=plan['force_n']*plan['ticks']*.005,start_s=start*.005,end_s=pulse['stop_tick']*.005,
                max_qpos_diff_from_nominal_through_pulse_onset=delta,loaded_substeps_during_pulse=int((normal>0).sum()),
                recorded_substeps=normal.size,flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in audit['events']),
                far_touch=audit['far_contact'],loaded_s=min(loads,default=None),handoff_s=hand,
                qualified=bool(loads and hand is not None and min(loads)<=hand+1e-5),goal=result['goal'],
                termination=result['termination'],end_time_s=float(t[-1])))
            write_rows(out/'paired_pulses.csv',rows)
    save(out/'terminal.json',dict(scored=len(rows)))


if __name__=='__main__':
    main()
