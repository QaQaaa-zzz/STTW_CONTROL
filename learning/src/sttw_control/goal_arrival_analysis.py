"""Reinterpret saved physical trajectories under an explicit spatial goal rule."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .survival_endpoint_task_metrics import goal_arrival_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_legacy_sources(worktree, output):
    """Normalize three historical schemas; preserve route-qualified identity.

    This writes only derived inventories. Raw paths/hashes are inherited from
    original indices and still authenticated by analyze before use.
    """
    root = Path(worktree); output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    definitions = [
        ('early_prospective', 'early_support_prospective_v1', 'analysis/all_cases.csv', 434),
        ('causal_relay', 'causal_recovery_relay_v1', 'analysis/results.csv', 50),
        ('hidden_time', 'hidden_time_reference_confirmation_v1', 'comparison_retry2/all_cases.csv', 110)]
    sources = []; provenance = []
    for name, directory, table_name, count in definitions:
        run = root / 'runs' / directory
        table = run / table_name; original_plan = run / 'plan.json'
        plan = json.loads(original_plan.read_text())
        rows = list(csv.DictReader(table.open()))
        if len(rows) != count:
            raise ValueError('historical table count changed')
        inputs = {str(table.resolve()): digest(table), str(original_plan.resolve()): digest(original_plan)}
        contexts = {}; indices = {}; outcomes = {}
        if name == 'early_prospective':
            route_runs = plan['route_run_paths']
            expected = plan['route_plan_sha256']
        elif name == 'hidden_time':
            route_runs = {item['route'].split('90_')[0]: item['run'] for item in plan['routes']}
            expected = {item['route'].split('90_')[0]: item['plan_sha256'] for item in plan['routes']}
        else:
            route_runs = {}; expected = {}
            if plan['lower_dt_s'] != .005 or plan['horizon_s'] != 10:
                raise ValueError('unexpected historical clock/window')
            contexts['right'] = dict(path=plan['path_file'], path_sha256=plan['path_sha256'])
        for route, route_run in route_runs.items():
            route_run = Path(route_run); file = route_run / 'plan.json'
            if digest(file) != expected[route]:
                raise ValueError('historical route plan changed')
            inputs[str(file.resolve())] = digest(file)
            rp = json.loads(file.read_text())
            if rp['lower_dt_s'] != .005 or rp['horizon_s'] != 10:
                raise ValueError('unexpected historical clock/window')
            contexts[route] = dict(path=rp['path_file'], path_sha256=rp['path_sha256'])
            if name == 'hidden_time':
                for filename in ['case_index.json', 'results.json']:
                    file = route_run / 'timing' / filename
                    inputs[str(file.resolve())] = digest(file)
                indices[route] = {r['case_id']: r for r in json.loads((route_run / 'timing/case_index.json').read_text())}
                outcomes[route] = {r['case_id']: r for r in json.loads((route_run / 'timing/results.json').read_text())}
        normalized = []
        for row in rows:
            route = row.get('route_group', row.get('context', row['route']))
            role = row.get('role', row.get('mode'))
            result = dict(row, context=route, role=role, case_id=route + '/' + row['case_id'])
            if name == 'hidden_time':
                item = indices[route][row['case_id']]
                actual = outcomes[route][row['case_id']]
                for key in ['role', 'alpha', 'force_N', 'event_id']:
                    left, right = actual[key], row[key]
                    if key in ['alpha', 'force_N']:
                        left = None if left in [None, ''] else float(left)
                        right = None if right in [None, ''] else float(right)
                    if left != right:
                        raise ValueError('historical metadata join mismatch')
                result.update(raw_file=str(Path(route_runs[route]) / item['file']),
                    raw_sha256=item['sha256'], physical_outcome=actual['physical_outcome'])
            normalized.append(result)
        table_out = output / (name + '.csv'); plan_out = output / (name + '_plan.json')
        with table_out.open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(normalized[0])); w.writeheader(); w.writerows(normalized)
        plan_out.write_text(json.dumps(dict(contexts=contexts, lower_dt_s=.005, horizon_s=10.,
            original_inputs_sha256=inputs, raw_records=count), indent=2) + '\n')
        sources.append((name, table_out, plan_out))
        provenance.append(dict(dataset=name, original_inputs_sha256=inputs, normalized_records=count))
    (output / 'source_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return sources


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
                context=row.get('context', 'right'),
                alpha=None if row['alpha'] in ['', None] else float(row['alpha']),
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
    if any(name == 'early_prospective' for name, _, _ in sources):
        render_legacy_overview(sources, output, radius_m)
        reference_bank_opportunity(records, output, radius_m)
    return summary


def reference_bank_opportunity(records, output, radius_m):
    """Posthoc finite-bank ceilings only; no access-to-future deployment claim."""
    roles = ['task_only', 'task_floor', 'task_early', 'task_constant']
    rows = [r for r in records if r['dataset'] == 'early_prospective' and
            float(r['goal_radius_m']) == radius_m and r['role'] in roles]
    if any(r['goal_arrived'] is None or r['goal_arrived'] == '' for r in rows):
        raise ValueError('known outcomes required for a finite-bank ceiling')
    # CSV-loaded callers and in-memory callers share explicit Boolean semantics.
    arrived = lambda r: r['goal_arrived'] is True or r['goal_arrived'] == 'True'
    groups = {}
    for row in rows:
        key = (row['context'], row['alpha'], row['event_id'])
        group = groups.setdefault(key, {})
        if row['role'] in group:
            raise ValueError('duplicate candidate in opportunity group')
        group[row['role']] = row
    if len(groups) != 70 or any(set(g) != set(roles) for g in groups.values()):
        raise ValueError('complete70 groups of four task-reference candidates required')
    ceilings = {}
    for split in ['nominal', 'disturbed']:
        subset = [g for g in groups.values() if
            bool(float(next(iter(g.values()))['force_N'])) == (split == 'disturbed')]
        ceilings[split] = dict(groups=len(subset), any_candidate_arrival=sum(
            any(arrived(r) for r in g.values()) for g in subset))
    fixed = []
    for route, alpha in dict.fromkeys((k[0], k[1]) for k in groups):
        scores = []
        for role in roles:
            rr = [r for r in rows if r['context'] == route and r['alpha'] == alpha and r['role'] == role]
            scores.append(dict(role=role,
                disturbed_arrival=sum(arrived(r) for r in rr if float(r['force_N'])),
                nominal_arrival=sum(arrived(r) for r in rr if not float(r['force_N'])),
                disturbed_survival=sum(r['physical_outcome'] == 'survived' for r in rr if float(r['force_N']))))
        best = max(scores, key=lambda s: (s['disturbed_arrival'], s['nominal_arrival'], s['disturbed_survival']))
        fixed.append(dict(route=route, alpha=alpha, **best))
    result = dict(radius_m=radius_m, per_event_ceilings=ceilings, fixed_route_alpha=fixed,
        scope='Choices use THIS panel outcomes: diagnostic ceilings, not a deployable selector or independent test. No zero-failure constraint imposed.',
        physics_steps=0, optimizer_updates=0)
    # Retrospective, time-grouped diagnostic: choose a complete reference rule
    # on the other two onsets, then evaluate both signs of the omitted onset.
    # This is deliberately separate from the full-panel hindsight ceiling.
    cross = []
    if all('_F' in str(r['event_id']) for r in rows if float(r['force_N'])):
        for route, alpha in dict.fromkeys((k[0], k[1]) for k in groups):
            cells = [r for r in rows if r['context'] == route and r['alpha'] == alpha]
            onsets = sorted({r['event_id'].split('_F')[0] for r in cells if float(r['force_N'])})
            if len(onsets) != 3:
                raise ValueError('three original onset groups required')
            for held in onsets:
                candidates = []
                for role in roles:
                    train = [r for r in cells if r['role'] == role and float(r['force_N']) and
                             r['event_id'].split('_F')[0] != held]
                    zero = [r for r in cells if r['role'] == role and not float(r['force_N'])]
                    score = (sum(arrived(r) for r in train), sum(arrived(r) for r in zero),
                             sum(r['physical_outcome'] == 'survived' for r in train))
                    candidates.append((score, role))
                # max keeps first role on exact ties; no held-out result used.
                selected = max(candidates, key=lambda item: item[0])[1]
                test = [r for r in cells if r['role'] == selected and float(r['force_N']) and
                        r['event_id'].split('_F')[0] == held]
                if len(test) != 2:
                    raise ValueError('both held-out force signs required')
                cross.append(dict(route=route, alpha=alpha, held_onset=held, selected_role=selected,
                    test_count=2, arrived=sum(arrived(r) for r in test),
                    survived=sum(r['physical_outcome'] == 'survived' for r in test)))
        result['leave_one_onset_out'] = dict(decisions=cross,
            test_count=sum(c['test_count'] for c in cross), arrived=sum(c['arrived'] for c in cross),
            survived=sum(c['survived'] for c in cross),
            scope='Retrospective cross-fitting on old DEV; no new route/initial-state independence. Both signs of each onset are held together; rule chosen before the evaluated event.')
    (Path(output) / 'reference_bank_opportunity.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def render_legacy_overview(sources, output, radius_m):
    """All seven events/two routes at alpha .5, without outcome selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    _, table, plan_file = next(s for s in sources if s[0] == 'early_prospective')
    plan = json.loads(Path(plan_file).read_text())
    rows = [r for r in csv.DictReader(Path(table).open()) if r['alpha'] and float(r['alpha']) == .5]
    routes = ['left', 'right']
    events = {route: list(dict.fromkeys(r['event_id'] for r in rows if r['context'] == route)) for route in routes}
    if any(len(v) != 7 for v in events.values()) or len(rows) != 84:
        raise ValueError('all six methods/two routes/seven events at alpha .5 required')
    fig, axes = plt.subplots(2, 7, figsize=(24, 9), layout='constrained')
    roles = list(dict.fromkeys(r['role'] for r in rows))
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
    for ri, route in enumerate(routes):
        with np.load(plan['contexts'][route]['path'], allow_pickle=False) as z:
            path = {k: z[k] for k in z.files}
        goal = [np.interp(float(path['goal']), path['s'], path['xy'][:, k]) for k in [0, 1]]
        for ei, event in enumerate(events[route]):
            ax = axes[ri, ei]
            reference = path['xy'][path['s'] <= float(path['goal'])]
            ax.plot(reference[:, 0], reference[:, 1], ':', color='.5')
            for role, color in zip(roles, colors):
                row = next(r for r in rows if r['context'] == route and r['event_id'] == event and r['role'] == role)
                if digest(row['raw_file']) != row['raw_sha256']:
                    raise ValueError('raw XY changed')
                with np.load(row['raw_file'], allow_pickle=False) as z:
                    xy = z['actual_xy'][z['active_tick'].astype(bool)]
                ax.plot(xy[:, 0], xy[:, 1], color=color, ls='--' if roles.index(role) % 2 else '-', label=role)
                ax.scatter(*xy[-1], color=color, marker='x' if row['physical_outcome'] == 'contact' else 'o', s=10)
            ax.add_patch(Circle(goal, radius_m, fill=False, color='black'))
            ax.scatter(*goal, color='black', marker='*', s=25)
            ax.set(title=route + ' ' + event, xlabel='World X (m)', ylabel='World Y (m)')
            ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6)
    fig.suptitle('Original alpha .5 | all14 route/event groups,84 saved trajectories | original target disk')
    fig.savefig(Path(output) / 'legacy_all_events_alpha05_xy.png', dpi=120)
    plt.close(fig)


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
