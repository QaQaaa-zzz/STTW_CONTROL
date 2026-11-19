"""Count the opportunity to test H1 per physical map and start in each time bin."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    evidence = parser.parse_args().evidence
    known = 'wide_oracle_055'
    bins = ('near', 'mid', 'far')
    starts = {}
    # Mirrored search includes repeats of the older discovery prefixes.
    # Count distinct handback states, never the number of trigger commands.
    for row in read(evidence/'mirrored_prefixes.csv'):
        key = ('DEV approach search', known, row['root'])
        states = starts.setdefault(key, set())
        if row['qualified'] == 'True':
            states.add((row['y_h_m'], row['yaw_h_rad']))
    for row in read(evidence/'prefixes.csv'):
        if float(row['width_m']) == .45:
            states = starts.setdefault(('DEV 0.45 m', 'wide_oracle_045', row['root']), set())
            if row['eligible'] == 'True':
                states.add((row['handoff_y_m'], row['handoff_yaw_deg']))
    for row in read(evidence/'new_domain_prefixes.csv'):
        key = ('DEV new starts', known, row['root'])
        states = starts.setdefault(key, set())
        if row['qualified'] == 'True':
            states.add((row['y_h_m'], row['yaw_h_rad']))
    for row in read(evidence/'heldout_prefixes.csv'):
        key = ('Frozen transfer', row['map'], row['root'])
        starts.setdefault(key, set())
        # These failed prefixes have no valid handback to branch.
        assert row['qualified'] == 'False'

    for result in evidence.glob('matched_layout/matched_layout.csv'):
        for row in read(result):
            key = ('Known start, other layout', 'heldout_reversal_055_19033', row['root'])
            starts.setdefault(key, set())
            assert row['qualified'] == 'False'

    measured = {}
    for path in sorted(p.parent for p in evidence.rglob('reversal_summary.csv')):
        matrix = read(path/'oracle_task_reversal.csv')
        (map_id, root), = {(r['map_id'], r['root']) for r in matrix}
        for r in read(path/'reversal_summary.csv'):
            measured[map_id, root, r['bin']] = r
    categories = ('insufficient_prefixes', 'both_infeasible', 'shared_one_side',
                  'one_way_differentiation', 'both_universal', 'two_way')
    records, counts, chart = [], Counter(), []
    tick_labels = []
    for (role, map_id, root), states in starts.items():
        colors = []
        for label in bins:
            r = measured.get((map_id, root, label))
            if len(states) >= 2:
                category = r['category']
            else:
                category = 'insufficient_prefixes'
            records.append(dict(role=role, map=map_id, root=root, bin=label,
                                distinct_handbacks=len(states), category=category,
                                physical_suffixes=r['physical_suffixes'] if r else '',
                                min_T_s=r['min_T_s'] if r else '', max_T_s=r['max_T_s'] if r else ''))
            counts[role, map_id, label, category] += 1
            colors.append(categories.index(category))
        chart.append(colors)
        tick_labels.append(f"{role}  {root}" + ('  new map' if role == 'Frozen transfer' and map_id != known else ''))
    totals = []
    for role, map_id in sorted({(r['role'],r['map']) for r in records}):
        for label in bins:
            values = {key:counts[role, map_id, label, key] for key in categories}
            totals.append(dict(role=role, map=map_id, bin=label,
                               total_map_root_pairs=sum(values.values()), **values))
    write(evidence/'h1_per_root.csv', records)
    write(evidence/'h1_counts.csv', totals)
    fig, ax = plt.subplots(figsize=(7, 5.6), layout='constrained')
    palette = ListedColormap(['#dedede','#6b6b6b','#56b4e9','#e69f00','#009e73','#332288'])
    ax.imshow(chart, cmap=palette, norm=BoundaryNorm(np.arange(7)-.5, 6), aspect='auto')
    ax.set(xticks=range(3), xticklabels=['Near', 'Middle', 'Far'],
           yticks=range(len(chart)), yticklabels=tick_labels, title='Same-map task reversal by start')
    for i, cells in enumerate(chart):
        for j, cell in enumerate(cells):
            ax.text(j, i, ['<2 prefixes','Neither','Shared side','One-way','Universal','Two-way'][cell],
                    ha='center', va='center', fontsize=8)
    fig.savefig(evidence/'h1_phase_map.png', dpi=150)


if __name__ == '__main__':
    main()
