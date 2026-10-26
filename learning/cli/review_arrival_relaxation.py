"""Recount completed STTW traces without final-hold task requirements.

This derives metrics only: it does not run physics, select a checkpoint, or alter
the frozen historical labels. Runs may share cached traces; do not pool counts.
"""
import argparse
import hashlib
import json
from pathlib import Path


def review(roots, output):
    if output.exists():
        raise ValueError('preserve existing derived reviews')
    result = {'physics_steps': 0, 'optimizer_updates': 0, 'runs': {},
              'definition': 'Original target XY minimum distance before contact/fault, within original 10s. No final speed, heading or hold requirement. Whole-window survival separate.',
              'limits': 'Retrospective DEV reinterpretation. Runs overlap through exact caches; never pool denominators or count threshold variants as independent repeats.'}
    for root in roots:
        source = root / 'analysis/results.json'
        for status in [root / 'status.json', root / 'analysis/status.json']:
            if json.loads(status.read_text())['phase'] != 'completed':
                raise ValueError(f'incomplete run: {root}')
        rows = json.loads(source.read_text())
        if len({row['case_id'] for row in rows}) != len(rows):
            raise ValueError('duplicate case_id')
        if any(row['arrival_outcome'] == 'unknown' or row['physical_outcome'] == 'unknown' for row in rows):
            raise ValueError('unknown outcomes require separate handling')
        roles = {}
        for role in sorted({row['role'] for row in rows}):
            group = [row for row in rows if row['role'] == role and row['force_N'] != 0]
            radii = {}
            for radius in [.1, .2, .3, .5]:
                arrived = [row for row in group if row['minimum_upright_goal_distance_m'] is not None
                           and row['minimum_upright_goal_distance_m'] <= radius]
                radii[str(radius)] = {'arrived': len(arrived),
                    'arrived_and_survived_whole_window': sum(row['physical_outcome'] == 'survived' for row in arrived),
                    'arrived_then_contact': sum(row['physical_outcome'] == 'contact' for row in arrived),
                    'newly_accepted_vs_old_strict': sum(not row['performance_success'] for row in arrived)}
            roles[role] = {'disturbed_total': len(group),
                'old_strict_performance_success': sum(row['performance_success'] for row in group),
                'survived_whole_window': sum(row['physical_outcome'] == 'survived' for row in group),
                'radii_m': radii}
        result['runs'][root.name] = {'results_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                   'results_path': str(source.resolve()), 'roles': roles}
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
