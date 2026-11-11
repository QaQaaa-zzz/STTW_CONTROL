"""Describe frozen candidate ranking on saved TEST branches; no new execution."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence', type=Path, required=True)
a = p.parse_args()
models = {name: json.loads((a.evidence / file).read_text()) for name, file in [('Three-head', 'event.json'), ('Shared-head', 'shared_head.json')]}
collection = Path(models['Three-head']['collection'])
protocol = json.loads((collection / 'protocol.json').read_text())
cfg = json.loads((collection / 'config.json').read_text())
traj = np.load(collection / 'evaluation/trajectories.npz')
roads = np.load(cfg['task_reference_path'])['xy']
roots = sorted({c['root'] for c in protocol['cases']})
fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
rows = []
for root, ax in zip(roots, axes.flat):
    lanes = [i for i, c in enumerate(protocol['cases']) if c['root'] == root]
    initial = traj['initial_root_xyzw'][lanes]
    row = dict(root=root, candidate_count=len(lanes), successful_candidates=sum(models['Three-head']['cases'][i]['success'] for i in lanes), initial_position_component_range_m=np.ptp(initial[:, :3], axis=0).tolist(), initial_velocity_component_range_m_s=np.ptp(initial[:, 7:10], axis=0).tolist(), selections={})
    for name, color, style in [('Three-head', '#0072B2', '-'), ('Shared-head', '#D55E00', '--')]:
        cases = models[name]['cases']
        assert all((cases[i]['root'], cases[i]['candidate']) == (protocol['cases'][i]['root'], protocol['cases'][i]['candidate']) for i in lanes)
        lane = min(lanes, key=lambda i: (-cases[i]['predicted_completion'], protocol['cases'][i]['candidate_id']))
        row['selections'][name] = dict(cases[lane], lane=lane)
        actual = traj['root_xyzw'][:, lane, :2][traj['valid'][:, lane].astype(bool)]
        road = roads[protocol['cases'][lane]['candidate_id']]
        ax.plot(road[:, 0], road[:, 1], ':', color=color, alpha=.55)
        ax.plot(actual[:, 0], actual[:, 1], style, color=color, label=f"{name}: {'goal' if cases[lane]['success'] else 'failure'}")
        ax.scatter(*actual[-1], color=color, marker='o' if cases[lane]['success'] else 'x', s=35)
    ax.set(title=f'Root {root}: {row["successful_candidates"]}/60 candidates succeed', xlabel='World x (m)', ylabel='World y (m)')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    rows.append(row)
fig.suptitle('Saved TEST branches selected by frozen scores\nDotted: original route; endpoints: actual goal/failure; not new online execution', fontsize=10)
fig.savefig(a.evidence / 'ranking_xy.png', dpi=220)
plt.close(fig)
result = dict(scope='Post-hoc frozen ranking diagnostic on previously inspected TEST roots; candidate-specific publication states differ. Not an online comparison or exact common-state counterfactual.', new_physics_steps=0, new_training_updates=0, models={k:v['checkpoint_sha256'] for k,v in models.items()}, roots=rows)
(a.evidence / 'ranking.json').write_text(json.dumps(result, indent=2)+'\n')
lines = ['# Frozen candidate-ranking diagnostic', '', '![All selected actual/reference XY trajectories](ranking_xy.png)', '', 'This analysis ranks the 60 saved candidates per root by their existing frozen autoregressive completion scores. Ties use candidate ID. Neither fitting nor new physical execution is performed. All four previously inspected TEST roots are shown.', '', '|Root|Successful candidates|Three-head selection / result|Shared-head selection / result|', '|---|---:|---|---|']
for r in rows:
    t,s=r['selections']['Three-head'],r['selections']['Shared-head']
    lines.append(f"|{r['root']}|{r['successful_candidates']}/60|{t['candidate']} / {t['success']}|{s['candidate']} / {s['success']}|")
lines += ['', 'The three-head top-ranked branches complete 4/4; the shared-head top-ranked branches complete 0/4. This connects frozen prediction differences to a finite action ranking, but does not show superiority over the fixed low-speed detour (also 4/4), parameter-count-controlled superiority, or new online improvement.', '', 'Each score was evaluated at its own saved publication state. Publication time is 0.18 s for all branches, but states are not exactly identical across candidates; componentwise initial position and velocity ranges are recorded in `ranking.json`. Consequently, these are observed branch outcomes associated with frozen rankings, not exact common-state counterfactuals. There are only four root groups, not 240 independent planning tests.', '', 'The 36-trajectory online result remains event/direct/rule = 5/12, 5/12, 6/12. These offline counts must not replace it. Follow-up causal comparisons should score all candidates at one observed state and physically execute each model selection under the same protocol.', '', 'Figure code: `learning/cli/analyze_event_candidate_ranking.py`. Scientific-visualization skill guidance and tooling attribution: [paper tooling notes](../../chapter3_paper/README.md#author-review-and-tooling-notes).']
(a.evidence / 'RANKING.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'selected_success': {k:sum(r['selections'][k]['success'] for r in rows) for k in models}, 'roots':len(rows)}))
