"""Check the fixed-alpha handback prerequisite on existing successful prefixes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_playground_task_reversal import write_rows
from run_playground_wide_gap_oracle import run, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    target=parser.parse_args().run.resolve()
    plan=json.loads((target/'plan.json').read_text())
    source=Path(plan['input'])
    config=json.loads((source/'screen.json').read_text())
    rows=[]
    for alpha in plan['alphas']:
        folder=source/'screen'
        if alpha:
            alphas=list(config['oracle_initial_alpha'])
            for lane in plan['lanes']:
                alphas[lane]=alpha
            folder=target/f'alpha{alpha:g}'
            run(dict(config,oracle_initial_alpha=alphas),folder,'source',10,target)
        results=json.loads((folder/'status.json').read_text())['results']
        audits=json.loads((folder/'jump_validation.json').read_text())['cases']
        for lane in [*plan['lanes'],plan['control']]:
            result,audit=results[lane],audits[lane]
            path=Path(result['case_dir'])/'trajectory.npz'
            with np.load(folder/path) as z,np.load(source/'screen'/path) as initial:
                for field in ('qpos','qvel'):
                    np.testing.assert_array_equal(z[field][0],initial[field][0])
                control_error=float(np.max(np.abs(z['qpos']-initial['qpos']))) if lane==plan['control'] and z['qpos'].shape==initial['qpos'].shape else None
                trace=z['trace']
                entry=result['rb_entry_s']
                speed=None
                if entry is not None:
                    k=np.argmin(np.abs(trace[:,0]-entry))
                    speed=float(trace[k,7]*np.cos(trace[k,6])+trace[k,8]*np.sin(trace[k,6]))
            load=min((t for t in (audit['solid_bank_landing_s'],audit['both_far_after_flight_s']) if t is not None),default=None)
            hand=result['handoff_s']
            rows.append(dict(batch_alpha=alpha,approach_alpha=0 if lane==plan['control'] else alpha,
                unchanged_control=lane==plan['control'],root=result['variant']['root_variant'],lane=lane,
                entry_s=entry,entry_forward_m_s=speed,flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in audit['events']),
                loaded_s=load,handoff_s=hand,qualified=bool(load is not None and hand is not None and load<=hand+1e-5),
                goal=result['goal'],termination=result['termination'],control_qpos_max_diff=control_error))
        write_rows(target/'alpha_handbacks.csv',rows)
    save(target/'terminal.json',dict(scored=len(rows),experimental_cases=sum(not r['unchanged_control'] for r in rows)))


if __name__=='__main__':
    main()
