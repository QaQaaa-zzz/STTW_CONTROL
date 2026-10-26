"""Audit completed zero-event traces; no simulation, fitting or new labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sttw_control.finite_value_evaluation_report import evaluation_inputs
from sttw_control.preference_best import digest
from sttw_control.preference_phase_audit import phase_metrics, first_hit_count, endpoint_direction
from sttw_control.rejoin_comparison import geometric_task_trace


def audit(root, output):
    root = root.resolve()
    if output.exists():
        raise ValueError('preserve previous audit')
    for name in ['status.json', 'analysis/status.json']:
        if json.loads((root / name).read_text())['phase'] != 'completed':
            raise ValueError('completed raw analysis required')
    plan = json.loads((root / 'plan.json').read_text())
    if digest(root / 'plan.json') != json.loads((root / 'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('changed declaration')
    cfg = json.loads((root / 'common_environment_config.json').read_text())
    rows = json.loads((root / 'analysis/results.json').read_text())
    _, _, indices, _ = evaluation_inputs(root, plan)
    index = {r['case_id']: r for r in indices}
    traces, records = {}, []
    dt = float(plan['lower_dt_s'])
    for row in rows:
        if row['force_N']:
            continue
        if row['physical_outcome'] != 'survived' or row['physical_ticks'] != 2000:
            raise ValueError('nominal full-window survival required, no survivor selection')
        item = index[row['case_id']]
        file = root / item['file']
        if digest(file) != item['sha256'] or item['sha256'] != row['raw_sha256']:
            raise ValueError('changed raw input')
        with np.load(file, allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        if not np.array_equal(raw['physical_tick'][raw['active_tick']], np.arange(2000)):
            raise ValueError('incomplete nominal clock')
        context = plan['contexts'][row['context']]
        if digest(context['path']) != context['path_sha256']:
            raise ValueError('changed original path')
        with np.load(context['path'], allow_pickle=False) as z:
            path = {k: z[k] for k in z.files}
        geom = geometric_task_trace(raw, path, cfg, initial_xy=item['initial_xy'])
        goal = np.array([np.interp(float(path['goal']), path['s'], path['xy'][:,k]) for k in [0,1]])
        arrival = first_hit_count(np.linalg.norm(raw['actual_xy'] - goal, axis=1) <= plan['goal_radius_m'])
        if (arrival is not None) != row['goal_arrived']:
            raise ValueError('arrival outcome changed')
        if arrival is not None:
            np.testing.assert_allclose(arrival * dt, row['first_arrival_s'], atol=1e-10, rtol=0)
        turn = first_hit_count(geom['path_progress'] >= float(path['turn_end']))
        signals = (raw['actual_forward_speed'] - row['requested_speed_m_s'],
                   geom['path_cross_track'], -raw['scored_tick_reward'])
        metric = lambda count: phase_metrics(*signals, count, dt)
        full = metric(2000)
        np.testing.assert_allclose(full['original_alpha_cost'], row['scored_cost'], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(full['path_abs_error_time_integral_m_s'],
                                   row['task_path_abs_error_time_integral_m_s'], rtol=2e-5, atol=2e-5)
        before = metric(arrival or 2000)
        record = dict(case_id=row['case_id'], context=row['context'], role=row['role'], alpha=row['alpha'],
            raw_file=str(file.resolve()), raw_sha256=item['sha256'], arrival_ticks=arrival, turn_end_ticks=turn,
            full=full, own_prearrival=before, own_turn_prefix=metric(turn) if turn else None,
            after_arrival_cost=full['original_alpha_cost'] - before['original_alpha_cost'] if arrival else None,
            after_arrival_path_fraction=(full['path_abs_error_time_integral_m_s'] - before['path_abs_error_time_integral_m_s']) / full['path_abs_error_time_integral_m_s'])
        records.append(record)
        traces[row['case_id']] = signals
    pairs = []
    for role in plan['evaluation_roles']:
        for context in plan['contexts']:
            cell = {r['alpha']: r for r in records if r['role'] == role and r['context'] == context}
            if set(cell) != {0., .5, 1.}:
                raise ValueError('full three-alpha nominal group required')
            a, b = cell[0.], cell[1.]
            count = min(a['arrival_ticks'] or 2000, b['arrival_ticks'] or 2000)
            common_a = phase_metrics(*traces[a['case_id']], count, dt)
            common_b = phase_metrics(*traces[b['case_id']], count, dt)
            pairs.append(dict(role=role, context=context, common_prefix_ticks=count,
                both_arrived=bool(a['arrival_ticks'] and b['arrival_ticks']),
                directions={phase: endpoint_direction(a[phase], b[phase])
                    for phase in ['full', 'own_prearrival', 'own_turn_prefix'] if a[phase] and b[phase]},
                common_prearrival=endpoint_direction(common_a, common_b)))
    summary = {}
    for role in plan['evaluation_roles']:
        group = [p for p in pairs if p['role'] == role]
        summary[role] = {}
        for phase in ['full', 'own_prearrival', 'own_turn_prefix', 'common_prearrival']:
            values = [p['common_prearrival'] if phase == 'common_prearrival' else p['directions'][phase]
                      for p in group if phase == 'common_prearrival' or phase in p['directions']]
            summary[role][phase] = dict(total=len(values), **{key: sum(v[key] for v in values)
                for key in ['alpha1_better_speed', 'alpha0_better_path', 'both_expected_directions']})
    result = dict(records=records, pairs=pairs, summary=summary, physics_steps=0, optimizer_updates=0,
        plan_sha256=digest(root/'plan.json'), results_sha256=digest(root/'analysis/results.json'),
        script_sha256=digest(__file__), module_sha256=digest(Path(__file__).parents[1]/'src/sttw_control/preference_phase_audit.py'),
        definition='Exploratory nominal phase audit. Own prefixes have different durations; common endpoint prefix uses the earlier arrival or10s, same duration for alpha0/1. Turn prefixes stop at original turn_end. Means accompany integrals; no new success gate or interpolation.',
        limits='Four correlated contexts, no independent training repeats. Common duration does not mean common subsequent physical states. No phase is selected as a new adoption rule; full-window physical outcome stays unchanged.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    print(json.dumps(result['summary'], indent=2))
