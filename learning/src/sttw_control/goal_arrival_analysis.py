"""Reinterpret saved physical trajectories under an explicit spatial goal rule."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .survival_endpoint_task_metrics import goal_arrival_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(sources, output, *, radius_m=.3):
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    radii = sorted(set([.1, .2, .3, .5, radius_m]))
    records = []; identities = []; examples = []
    for name, table, plan_file in sources:
        table = Path(table); plan_file = Path(plan_file)
        plan = json.loads(plan_file.read_text())
        contexts = plan.get('contexts', {'default': dict(path=plan.get('path_file'),
                                                       path_sha256=plan.get('path_sha256'))})
        paths = {}
        for key, item in contexts.items():
            if digest(item['path']) != item['path_sha256']:
                raise ValueError('original path identity changed')
            with np.load(item['path'], allow_pickle=False) as z:
                paths[key] = {k: z[k] for k in z.files}
        rows = list(csv.DictReader(table.open()))
        if len({r['case_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate source trajectory')
        identities.append(dict(dataset=name, rows=len(rows), table=str(table.resolve()),
            table_sha256=digest(table), plan=str(plan_file.resolve()), plan_sha256=digest(plan_file)))
        for row in rows:
            file = Path(row['raw_file'])
            if digest(file) != row['raw_sha256']:
                raise ValueError('raw trajectory changed')
            with np.load(file, allow_pickle=False) as z:
                raw = {k: z[k] for k in z.files}
            path = paths[row.get('context', 'default')]
            goal = np.array([np.interp(float(path['goal']), path['s'], path['xy'][:, k]) for k in [0, 1]])
            base = dict(dataset=name, case_id=row['case_id'], role=row['role'],
                context=row.get('context', 'right'), alpha=float(row['alpha']),
                force_N=float(row['force_N']), event_id=row.get('event_id', row['force_N']),
                old_strict_task=row['category'] == 'safe_task',
                physical_outcome=row['physical_outcome'], raw_sha256=row['raw_sha256'])
            for radius in radii:
                metrics = goal_arrival_metrics(raw, goal_xy=goal, radius_m=radius,
                    dt=plan['lower_dt_s'], horizon_s=plan['horizon_s'])
                records.append(dict(base, **metrics))
            if name == 'force_grid' and base['alpha'] == .5 and base['force_N'] in [-32., 32., 48.]:
                examples.append((base, raw, path, goal))
    summary = []
    for dataset in dict.fromkeys(r['dataset'] for r in records):
        for role in dict.fromkeys(r['role'] for r in records if r['dataset'] == dataset):
            for split in ['all', 'nominal', 'disturbed']:
                for radius in radii:
                    group = [r for r in records if r['dataset'] == dataset and r['role'] == role
                        and r['goal_radius_m'] == radius and
                        (split == 'all' or bool(r['force_N']) == (split == 'disturbed'))]
                    summary.append(dict(dataset=dataset, role=role, split=split, radius_m=radius,
                        total=len(group), arrived=sum(r['goal_arrived'] is True for r in group),
                        not_arrived=sum(r['goal_arrived'] is False for r in group),
                        unknown=sum(r['goal_arrived'] is None for r in group),
                        arrived_and_survived=sum(r['goal_arrived'] is True and r['physical_outcome'] == 'survived' for r in group),
                        contact_after_arrival=sum(r['contact_after_arrival'] for r in group)))
    for filename, rows in [('cases.csv', records), ('summary.csv', summary)]:
        with (output / filename).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    receipt = dict(sources=identities, working_radius_m=radius_m,
        radius_authority='30cm is an explicitly stated working assumption, not a user-specified numerical value.',
        radii_m=radii, physics_steps=0, optimizer_updates=0,
        semantics='Observed actual XY enters original target disk before contact/fault; first arrival, no terminal heading/speed/hold test. Later physical outcome reported separately.',
        limits='Retrospective reanalysis of old DEV data; no independent test or new control improvement; original labels and rewards unchanged.')
    receipt['analysis_source_sha256'] = {str(Path(__file__).resolve()): digest(__file__),
        str(Path(__file__).with_name('survival_endpoint_task_metrics.py').resolve()):
            digest(Path(__file__).with_name('survival_endpoint_task_metrics.py'))}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    render_examples(examples, output, radius_m)
    return summary


def render_examples(examples, output, radius_m):
    if not examples:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    styles = {'task_only': ('#0072B2', '--'), 'task_constant': ('#D55E00', '-')}
    for ax, force in zip(axes, [-32., 32., 48.]):
        for row, raw, path, goal in examples:
            if row['force_N'] != force:
                continue
            active = raw['active_tick'].astype(bool); xy = raw['actual_xy'][active]
            color, ls = styles[row['role']]
            ax.plot(xy[:, 0], xy[:, 1], color=color, ls=ls, label=row['role'])
            ax.scatter(*xy[-1], color=color, marker='x' if row['physical_outcome'] == 'contact' else 'o')
        original = path['xy'][path['s'] <= float(path['goal'])]
        ax.plot(original[:, 0], original[:, 1], ':', color='.5', label='Original path')
        ax.add_patch(Circle(goal, radius_m, fill=False, color='black', lw=1.5))
        ax.scatter(*goal, marker='*', color='black', label=f'Goal radius {radius_m:g}m')
        ax.set(title=f'Original alpha .5 | {force:+g}N', xlabel='World X (m)', ylabel='World Y (m)')
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Actual XY and target disk; endpoints remain original observed endpoints')
    fig.savefig(output / 'goal_disk_examples.png', dpi=150)
    plt.close(fig)
