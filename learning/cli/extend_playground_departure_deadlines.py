"""Re-evaluate only censored suffixes, preserving the measured first ten seconds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_playground_wide_gap_oracle import run, save
from run_playground_task_reversal import crossings_by_section, write_rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    out=parser.parse_args().output.resolve()
    plan=json.loads((out/'plan.json').read_text())
    rows=[]
    for group,folder in plan['groups'].items():
        folder=Path(folder)
        contract=json.loads((folder/'plan.json').read_text())['contracts']
        with (folder/'oracle_task_reversal.csv').open() as f:
            original=list(csv.DictReader(f))
        cases={(r['lane'],r['departure']):r for r in original}
        for (lane,dep),r in cases.items():
            trajectory=Path(r['trajectory'])
            term=r['termination']
            end=float(r['end_s'])
            hold=float(r['hold_s'])
            hand=float(r['handoff_s'])
            wall=r['forbidden_contact']=='True'
            if term=='timeout':
                cfg=json.loads((folder/f"lane{lane}_{dep if dep!='straight' else 'right'}.json").read_text())
                if dep=='straight':
                    ref=json.loads((folder/'repeat.json').read_text())['ground_handoff']['reference_path']
                    cfg.update(oracle_branch_reference=ref,ground_handoff=dict(cfg['ground_handoff'],reference_path=ref))
                case=out/f'{group}_{lane}_{dep}'
                if (case/'status.json').exists() and (case.parent/(case.name+'.exit.json')).exists():
                    result=json.loads((case/'status.json').read_text())['results'][int(lane)]
                else:
                    result=run(cfg,case,'source',12,out)[int(lane)]
                next_path=case/result['case_dir']/'trajectory.npz'
                with np.load(trajectory) as a,np.load(next_path) as b:
                    start=np.argmin(abs(a['trace'][:,0]-b['trace'][0,0]))
                    length=len(a['trace'])-start
                    for field in ('qpos','qvel','trace'):
                        np.testing.assert_array_equal(a[field][start:],b[field][:length])
                    # The original final row records a state but applies no next command.
                    valid=a['applied_target_valid'][start:].astype(bool)
                    np.testing.assert_array_equal(a['applied_targets'][start:][valid],b['applied_targets'][:length][valid])
                    end=float(b['trace'][-1,0])
                    wall=bool(b['world_flags'][:,2].any())
                trajectory=next_path
                term=result['termination']
                hold=result['goal_hold_s']
            with np.load(trajectory) as z:
                crossings={label:y for label,_,_,y in crossings_by_section(z['trace'],hand,contract['x_sections_m'])}
            for old in original:
                if (old['lane'],old['departure'])!=(lane,dep):continue
                y=crossings[old['bin']]
                lo,hi=contract[old['task']+'_y_interval_m']
                gate=y is not None and lo<=y<=hi
                rows.append(dict(group=group,lane=lane,departure=dep,bin=old['bin'],task=old['task'],
                                 checkpoint_12s=gate,wall_contact=wall,termination_10s=r['termination'],
                                 termination_12s=term,hold_12s=hold,end_12s=end,success_10s=old['success'],
                                 success_12s=bool(gate and not wall and term=='goal'),trajectory=str(trajectory)))
            write_rows(out/'deadlines.csv',rows)
    save(out/'terminal.json',dict(contract_scores=len(rows),deadline_s=12))


if __name__=='__main__':
    main()
