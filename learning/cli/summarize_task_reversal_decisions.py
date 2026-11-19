"""Compare admissible post-handoff departures for each frozen contract."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def summarize(directories):
    comparisons = []
    for directory in directories:
        with (directory/'oracle_task_reversal.csv').open() as stream:
            identities = {r['lane']:(r['map_id'],r['root']) for r in csv.DictReader(stream)}
        with (directory/'offline_false_accepts.csv').open() as stream:
            groups = defaultdict(list)
            for r in csv.DictReader(stream):
                groups[r['lane'],r['bin'],r['task'],r['margin']].append(r)
        for (lane, time_bin, task, margin), scores in groups.items():
            map_id, root = identities[lane]
            truth = {r['departure'] for r in scores if r['actual_pass']=='True'}
            row = dict(map=map_id, root=root, prefix=lane, bin=time_bin, task=task, margin=margin,
                       feasible=' '.join(sorted(truth)))
            for method in ('kinematic','phase2'):
                candidates = {r['departure']:r for r in scores if r['method']==method}
                assert set(candidates)=={'left','right','straight'}
                accepted = {key for key,r in candidates.items() if r['predicted_pass']=='True'}
                row[method+'_accepted'] = ' '.join(sorted(accepted))
                row[method+'_contains_success'] = bool(accepted & truth)
                row[method+'_contains_failure'] = bool(accepted - truth)
            comparisons.append(row)
    return comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    rows = summarize(args.directories)
    with args.output.open('w') as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
