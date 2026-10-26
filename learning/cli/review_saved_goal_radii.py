"""Compare saved DEV trajectories at shared goal radii; no physics or fitting."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def review(sources, output):
    summaries, provenance = [], []
    for name, filename in sources:
        path = Path(filename)
        rows = list(csv.DictReader(path.open()))
        provenance.append(dict(name=name, file=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_rows=len(rows)))
        unique = {}
        for row in rows:
            key = (row.get('dataset', name), row['case_id'])
            if key in unique:
                old = unique[key]
                for field in ['raw_sha256', 'minimum_upright_goal_distance_m', 'physical_outcome', 'role', 'force_N']:
                    if old[field] != row[field]:
                        raise ValueError(f'inconsistent repeated-radius metadata: {key}, {field}')
            unique[key] = row
            # Independently verify saved binary arrival against recorded minimum,
            # which was measured only before first contact/fault in the raw audit.
            value = row['minimum_upright_goal_distance_m']
            if value and row['arrival_outcome'] != 'unknown':
                expected = float(value) <= float(row['goal_radius_m'])
                if expected != (row['goal_arrived'] == 'True'):
                    raise ValueError(f'saved arrival/minimum mismatch: {key}')
        groups = defaultdict(list)
        for (dataset, _), row in unique.items():
            groups[dataset, row['role'], 'disturbed' if float(row['force_N']) else 'nominal'].append(row)
        for (dataset, role, split), group in sorted(groups.items()):
            counts = []
            for radius in [.1, .2, .3, .5]:
                arrived = unknown = old_strict = 0
                near_miss = far = 0
                for row in group:
                    value = row['minimum_upright_goal_distance_m']
                    distance = float(value) if value else float('inf')
                    hit = distance <= radius
                    arrived += hit
                    unknown += not hit and row['arrival_outcome'] == 'unknown'
                    old_strict += row.get('old_strict_task', row.get('performance_success')) == 'True'
                    near_miss += .3 < distance <= .5
                    far += distance > 1 and row['physical_outcome'] == 'survived'
                counts.append(arrived)
                summaries.append(dict(panel=name, dataset=dataset, role=role, split=split,
                    radius_m=radius, total=len(group), arrived=arrived, unknown=unknown,
                    survived=sum(r['physical_outcome'] == 'survived' for r in group),
                    old_strict_success=old_strict, near_miss_30_to_50cm=near_miss,
                    survived_more_than_1m_away=far))
            if counts != sorted(counts):
                raise ValueError('nonmonotone radius counts')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / 'receipt.json').write_text(json.dumps(dict(sources=provenance,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_steps=0, optimizer_updates=0, radii_m=[.1, .2, .3, .5],
        authority='User allows relaxed task criteria; 50cm is an agent working assumption for retrospective review, not an independently validated numerical requirement.',
        definition='First measured XY entry into original goal disk before contact/fault within original 10s; no final hold/speed/heading test; later physical outcome separate.',
        limits='Derived tables from previously raw-authenticated DEV audits. Raw traces not rehashed this pass. Panels may share traces: never pool counts as independent samples. Radius chosen after observing results; not a new prospective test. Old labels and original alpha costs unchanged.'), indent=2) + '\n')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, help='panel_name=CSV_path')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review([item.split('=', 1) for item in args.source], args.output)
    for row in result:
        if row['split'] == 'disturbed' and row['radius_m'] in [.3, .5]:
            print(json.dumps(row))
