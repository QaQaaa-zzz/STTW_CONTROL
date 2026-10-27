"""Read-only enumeration of the sealed legacy preference panel; no simulation."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save_table(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    plan = json.loads((root / 'plan.json').read_text())
    for path, sha in plan['sources_sha256'].items():
        assert digest(path) == sha, path
    paths = list(plan['sources_sha256'])
    source = table(paths[0])
    legacy = defaultdict(list)
    for row in table(paths[2]):
        if row['dataset'] == 'early_prospective':
            legacy[row['case_id']].append(row)
    assert len(source) == len(legacy) == plan['expected_records']
    rows = []
    old_checks = 0
    for row in source:
        sha = digest(row['raw_file'])
        assert sha == row['raw_sha256'], row['key']
        old = legacy[row['key']]
        assert sorted(float(v['goal_radius_m']) for v in old) == [.1, .2, .3, .5]
        distances = {float(v['minimum_upright_goal_distance_m']) for v in old}
        assert len(distances) == 1
        distance = distances.pop()
        for v in old:
            assert v['raw_sha256'] == sha and v['context'] == row['route']
            assert v['physical_outcome'] == row['physical_outcome']
            assert (distance <= float(v['goal_radius_m'])) == (v['goal_arrived'] == 'True')
            assert v['arrival_outcome'] != 'unknown'
            old_checks += 1
        # These source outcomes are all known. Do not generalize unknown to false.
        assert row['physical_outcome'] in ('survived', 'contact')
        survival = row['physical_outcome'] == 'survived'
        rows.append(dict(key=row['key'], role=row['role'], route=row['route'],
                         alpha=None if row['role'] == 'zero_middle' else float(row['alpha']),
                         event_id=row['event_id'], onset=int(row['onset_tick']),
                         force=float(row['force_N']), survival=survival, goal=distance <= 1,
                         joint=survival and distance <= 1, distance=distance, raw_sha256=sha))
    roles = plan['learned_roles']
    metrics = ('survival', 'goal', 'joint')
    checks = []
    for role, route in itertools.product(roles, sorted({r['route'] for r in rows})):
        group = [r for r in rows if r['role'] == role and r['route'] == route and r['force']]
        for onset in sorted({r['onset'] for r in group}):
            lookup = {(r['alpha'], r['force']): r for r in group if r['onset'] == onset}
            assert len(lookup) == 10
            for a, b in itertools.combinations(plan['execution_alphas'], 2):
                out = dict(role=role, route=route, onset=onset, alpha_a=a, alpha_b=b)
                for metric in metrics:
                    differences = [int(lookup[a, f][metric]) - int(lookup[b, f][metric]) for f in (-32., 32.)]
                    out[metric] = differences[0] * differences[1] == -1
                checks.append(out)
    zero = {(r['route'], r['event_id']): r for r in rows if r['role'] == 'zero_middle'}
    pairs = []
    for r in rows:
        if r['role'] == 'zero_middle':
            continue
        z = zero[r['route'], r['event_id']]
        pairs.append(dict(key=r['key'], zero_key=z['key'], role=r['role'],
                          **{m: int(r[m]) - int(z[m]) for m in metrics}))
    assert len(checks) == 360 and len(pairs) == 420 and len(zero) == 14
    summary = {role: dict(n=sum(r['role'] == role for r in rows),
                          **{m: sum(r[m] for r in rows if r['role'] == role) for m in metrics})
               for role in roles + ['zero_middle']}
    reversal = {m: dict(pairs=sum(c[m] for c in checks), checks=len(checks),
                       role_contexts=len({(c['role'], c['route'], c['onset']) for c in checks if c[m]}),
                       route_onsets=len({(c['route'], c['onset']) for c in checks if c[m]})) for m in metrics}
    comparison = {role: {m: dict(gains=sum(p[m] == 1 for p in pairs if p['role'] == role),
                                losses=sum(p[m] == -1 for p in pairs if p['role'] == role))
                         for m in metrics} for role in roles}
    review = dict(summary=summary, reversals=reversal, zero_comparison=comparison,
                  raw_records=len(rows), distinct_raw_sha256=len({r['raw_sha256'] for r in rows}),
                  old_radius_checks=old_checks, new_physics_ticks=0, new_training_steps=0,
                  first_arrival_time_1m='not_reconstructed', scope=plan['statistics'])
    independent = json.loads((root / 'independent_review.json').read_text())
    for role, values in summary.items():
        assert all(independent['summary'][role][k] == v for k, v in values.items())
    for m, values in reversal.items():
        other = independent['reversal_summary'][m]
        assert values == dict(pairs=other['reversal_pairs'], checks=other['checks'],
                              role_contexts=other['reversal_role_contexts'], route_onsets=other['reversal_route_onsets'])
    review['independent_aggregate_agreement'] = True
    (root / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    save_table(root / 'records.csv', rows)
    save_table(root / 'directional_checks.csv', checks)
    save_table(root / 'zero_pairs.csv', pairs)
    # All contexts displayed; letters retain meaning without relying on color.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    for route in sorted({r['route'] for r in rows}):
        events = sorted({(r['onset'], r['force'], r['event_id']) for r in rows if r['route'] == route and r['force']})
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for ax, role in zip(axes.flat, roles):
            matrix, labels = [], []
            for alpha in plan['execution_alphas'] + [None]:
                cells, letters = [], []
                for _, _, event in events:
                    r = next(r for r in rows if r['route'] == route and r['event_id'] == event and
                             r['role'] == ('zero_middle' if alpha is None else role) and r['alpha'] == alpha)
                    code = 3 if r['goal'] and not r['survival'] else 2 if r['joint'] else 1 if r['survival'] else 0
                    cells.append(code)
                    letters.append(['C', 'S', 'A', 'A/C'][code])
                matrix.append(cells)
                labels.append(letters)
            ax.imshow(matrix, vmin=0, vmax=3, cmap=ListedColormap(['#df9a91', '#f0dc9a', '#9fcbb7', '#c5b3dd']))
            for i, line in enumerate(labels):
                for j, label in enumerate(line):
                    ax.text(j, i, label, ha='center', va='center')
            ax.set(title=role, yticks=range(6), yticklabels=['a=0', '.25', '.5', '.75', '1', 'zero'],
                   xticks=range(6), xticklabels=[f'{t}\n{f:+g} N' for t, f, _ in events])
        fig.suptitle(f'{route}: all signed events; onset in lower ticks (0.005 s)\nA: goal + survival; S: survival only; C: contact without goal; A/C: goal then contact\nOriginal goal radius 1 m, original 10 s window; zero row reused, not new trials')
        fig.savefig(root / f'{route}_all_contexts.png', dpi=160)
        plt.close(fig)
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
