"""Replay selected known handbacks on an existing layout with the original batch ordering."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_playground_task_reversal import write_rows
from run_playground_wide_gap_oracle import run, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--report', type=Path, help='Read completed trajectories and write their comparison here')
    args = parser.parse_args()
    target = args.run.resolve()
    plan = json.loads((target/'plan.json').read_text())
    original = Path(plan['input'])
    if args.report is None:
        neutral = json.loads((original/'neutral.json').read_text())
        cfg = json.loads((original/'screen.json').read_text())
        run(dict(neutral, world_map=plan['map']), target/'neutral', 'source', .005, target)
        cfg.update(world_map=plan['map'], initial_snapshot_path=str(target/'neutral/source_initial_snapshot.pkl'))
        run(cfg, target/'screen', 'source', 10, target)
    destination = (args.report or target)
    destination.mkdir(parents=True, exist_ok=True)
    results = json.loads((target/'screen/status.json').read_text())['results']
    audits = json.loads((target/'screen/jump_validation.json').read_text())['cases']
    baseline = {r['lane']:r for r in csv.DictReader((original/'prefixes.csv').open())}
    rows = []
    fig, axes = plt.subplots(len(plan['lanes']), 1, figsize=(10, 2.2*len(plan['lanes'])),
                             squeeze=False, layout='constrained')
    for panel, lane in zip(axes[:,0], plan['lanes']):
        result, audit, before = results[lane], audits[lane], baseline[str(lane)]
        assert before['qualified']=='True' and audit['case']==result['case_dir']
        path = Path(result['case_dir'])/'trajectory.npz'
        with np.load(original/'screen'/path) as first, np.load(target/'screen'/path) as second:
            for field in ('qpos','qvel','controller_state','rb_state'):
                np.testing.assert_array_equal(first[field][0], second[field][0])
            end = second['trace'][-1]
            reference = first['published_reference_xy']
            np.testing.assert_array_equal(reference,second['published_reference_xy'])
            panel.plot(*reference.T, ':', color='0.45', label='Reference', lw=.8)
            for record, label, color in ((first,'Original map','#0072b2'),(second,'Layout 19033','#d55e00')):
                xy = record['qpos'][:,:2]
                panel.plot(*xy.T, color=color, label=label)
                panel.plot(*xy[-1], 'x', color=color)
            panel.set(aspect='equal', xlim=(1,18), ylim=(-1.4,1.4), xlabel='World x (m)',
                      ylabel='World y (m)', title=f"Root {before['root']} | approach lane {lane}")
        loads = [t for t in (audit['solid_bank_landing_s'],audit['both_far_after_flight_s']) if t is not None]
        hand = result['handoff_s']
        rows.append(dict(root=before['root'], lane=lane, template=before['template'],
                         trigger_offset_m=before['trigger_offset_m'], original_handoff_s=before['handoff_s'],
                         flight=any(e['bank_to_bank_flight'] or e['solid_bank_rear_launch'] for e in audit['events']),
                         far_contact=audit['far_contact'], loaded_s=min(loads,default=None), handoff_s=hand,
                         qualified=bool(loads and hand is not None and min(loads)<=hand+1e-5),
                         goal=result['goal'], termination=result['termination'], end_s=float(end[0])))
    axes[0,0].legend(fontsize=8, ncol=3)
    fig.savefig(destination/'matched_layout_xy.png', dpi=145)
    write_rows(destination/'matched_layout.csv', rows)
    save(destination/'terminal.json', dict(scored=len(rows), qualified=sum(r['qualified'] for r in rows)))


if __name__=='__main__':
    main()
