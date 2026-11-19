"""Compare recorded nominal and disturbed physical outcomes on the fixed jump task."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_playground_task_reversal import crossings_by_section, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('contracts', type=Path, help='The frozen task-reversal plan')
    parser.add_argument('output', type=Path)
    parser.add_argument('--windows', nargs='+', help='Limit the comparison to completed exposure windows')
    args = parser.parse_args()
    plan = json.loads((args.run/'plan.json').read_text())
    contracts = json.loads(args.contracts.read_text())['contracts']
    timings = {'nominal':None, **{k:v*.005 for k,v in plan['windows'].items()}}
    if args.windows:
        timings = {key:timings[key] for key in args.windows}
    lane = plan['lane']
    rows, scores = [], []
    fig, axes = plt.subplots(len(timings), 1, figsize=(9, 2.2*len(timings)), layout='constrained', squeeze=False)
    for ax, (window, onset) in zip(axes[:,0], timings.items()):
        for skill, color in zip(plan['skills'], ('#0072b2','#d55e00'), strict=True):
            folder = Path(plan['input'])/f'055_{skill}'/'batch' if onset is None else args.run/f'{window}_{skill}'
            result = json.loads((folder/'status.json').read_text())['results'][lane]
            audit = json.loads((folder/'jump_validation.json').read_text())['cases'][lane]
            with np.load(folder/result['case_dir']/'trajectory.npz') as z:
                trace, ref = z['trace'], z['published_reference_xy']
                wall = bool(z['world_flags'][:,2].any())
                if skill == plan['skills'][0]:
                    ax.plot(*ref.T, ':', color='0.5', lw=.8, label='Reference')
                ax.plot(trace[:,1], trace[:,2], color=color, label=skill)
                ax.plot(*trace[-1,1:3], 'x', color=color)
                if onset is not None:
                    k = np.argmin(abs(trace[:,0]-onset))
                    ax.plot(*trace[k,1:3], 'o', markerfacecolor='none', color=color)
            land = min((t for t in (audit['solid_bank_landing_s'],audit['both_far_after_flight_s'])
                        if t is not None), default=None)
            hand = result['handoff_s']
            returned = land is not None and hand is not None and land<=hand+1e-5
            flight = any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in audit['events'])
            rows.append(dict(window=window, skill=skill, flight=flight, far_touch=audit['far_contact'],
                             loaded_s=land, handoff_s=hand, qualified=returned, original_goal=result['goal'],
                             full_task=bool(flight and returned and not wall and result['goal']),
                             hold_s=result['goal_hold_s'], termination=result['termination'],
                             end_s=float(trace[-1,0])))
            if returned:
                for name, _, at, y in crossings_by_section(trace,hand,contracts['x_sections_m']):
                    for side in ('left','right'):
                        lo,hi = contracts[side+'_y_interval_m']
                        scores.append(dict(window=window,skill=skill,bin=name,task=side,cross_s=at,cross_y_m=y,
                                           nominal_departure_pass=bool(y is not None and lo<=y<=hi and not wall and result['goal'])))
        ax.axvspan(10.3,10.85,color='0.8',alpha=.4)
        ax.set(aspect='equal',xlim=(1,18),ylim=(-1.5,1.5),xlabel='World x (m)',ylabel='World y (m)',title=window.replace('_',' '))
    axes[0,0].legend(fontsize=8,ncol=3)
    fig.suptitle('Root 3 | matched jump skills | open circle: pulse onset | cross: endpoint')
    args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/'paired_pulses_xy.png',dpi=145)
    write_rows(args.output/'skill_outcomes.csv',rows)
    score_file = args.output/'executed_contracts.csv'
    if scores:
        write_rows(score_file,scores)
    else:
        score_file.write_text('window,skill,bin,task,cross_s,cross_y_m,nominal_departure_pass\n')
    paired = []
    for window in timings:
        source, rb = [r for r in rows if r['window']==window]
        for stage in ('flight','qualified','full_task'):
            a,b = source[stage],rb[stage]
            paired.append(dict(window=window,stage=stage,pairs=1,source=int(a),full_rb=int(b),gained=int(b and not a),
                               retained=int(a and b),lost=int(a and not b),neither=int(not a and not b)))
    write_rows(args.output/'paired_changes.csv',paired)


if __name__=='__main__':
    main()
