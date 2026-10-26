"""Recount completed STTW traces without final-hold task requirements.

This derives metrics only: it does not run physics, select a checkpoint, or alter
the frozen historical labels. Runs may share cached traces; do not pool counts.
"""
import argparse
import hashlib
import json
from collections import defaultdict
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_endpoints(root):
    root = Path(root).resolve()
    for name in ('status.json', 'analysis/status.json'):
        assert json.loads((root / name).read_text())['phase'] == 'completed'
    source = root / 'analysis/results.json'
    rows = json.loads(source.read_text())
    assert len({r['case_id'] for r in rows}) == len(rows)
    for r in rows:
        assert sha(r['raw_file']) == r['raw_sha256'], r['case_id']
        assert r['goal_radius_m'] == .5
        assert r['arrival_outcome'] != 'unknown'
        distance = r['minimum_upright_goal_distance_m']
        assert r['goal_arrived'] == (distance is not None and distance <= .5)

    def stats(group):
        return dict(total=len(group), old_strict_success=sum(r['performance_success'] for r in group),
                    arrival=sum(r['goal_arrived'] for r in group),
                    old_failure_now_arrival=sum(r['goal_arrived'] and not r['performance_success'] for r in group),
                    survived=sum(r['physical_outcome'] == 'survived' for r in group),
                    alive_but_missed=sum(r['physical_outcome'] == 'survived' and not r['goal_arrived'] for r in group),
                    radii={str(v):sum(r['minimum_upright_goal_distance_m'] is not None and
                                     r['minimum_upright_goal_distance_m'] <= v for r in group)
                           for v in (.1,.2,.3,.5)})

    roles = sorted({r['role'] for r in rows})
    totals = {role:{label:stats([r for r in rows if r['role'] == role and
                                bool(r['force_N']) == disturbed])
                   for label,disturbed in [('disturbed',True),('zero_force',False)]}
              for role in roles}
    endpoint = [r for r in rows if r['alpha'] in (0.,1.) and r['role'] in
                ('feedback_010','steer_only_feedback')]
    pairs = defaultdict(dict)
    for r in endpoint:
        key = (r['context'],r['alpha'],r['event_id'])
        assert r['role'] not in pairs[key]
        pairs[key][r['role']] = r
    selected=[]
    for cell in pairs.values():
        assert set(cell) == {'feedback_010','steer_only_feedback'}
        a,b = cell.values()
        for field in ('alpha','force_N','pulse_onset_task_tick','pulse_duration_s','world_direction',
                      'requested_speed_m_s','bank_index','feedback_gain_rad'):
            assert a[field] == b[field], field
        selected.append(cell['feedback_010' if a['alpha'] == 0. else 'steer_only_feedback'])
    endpoints = {}
    for label,disturbed in [('disturbed',True),('zero_force',False)]:
        chosen=[r for r in selected if bool(r['force_N']) == disturbed]
        contrasts={}
        for role in ('feedback_010','steer_only_feedback'):
            matched=[(r,pairs[(r['context'],r['alpha'],r['event_id'])][role]) for r in chosen]
            contrasts[role]=dict(gained=sum(a['goal_arrived'] and not b['goal_arrived'] for a,b in matched),
                                 lost=sum(b['goal_arrived'] and not a['goal_arrived'] for a,b in matched),
                                 mean_original_alpha_cost_difference=sum(a['scored_cost']-b['scored_cost']
                                                                         for a,b in matched)/len(matched))
        endpoints[label]=dict(selected=stats(chosen),paired=contrasts)
    return dict(source=str(source), source_sha256=sha(source), raw_files_authenticated=len(rows),
                totals=totals, exploratory_R3_endpoints=endpoints,
                selected_case_ids=[r['case_id'] for r in selected],
                physics_steps=0, optimizer_updates=0,
                scope='Post-data endpoint reuse only: alpha0=both, alpha1=steering-only. '
                      'No midpoint interpolation, no new closed-loop R3 test, no independent test set. '
                      'The 0.5m radius is a working threshold; strict historical labels stay unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append')
    parser.add_argument('--endpoint-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.run) == bool(args.endpoint_root):
        parser.error('choose --run or --endpoint-root')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.endpoint_root:
        result = review_endpoints(args.endpoint_root)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    else:
        result = review(args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
