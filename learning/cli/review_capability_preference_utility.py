"""Retrospective V3 cross-utility of authenticated historical capability90.

Rank only nominal or common complete-survival events; no terminal/failure
penalty can create the reported preference advantage. No simulator is loaded.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sttw_control.goal_arrival_analysis import digest
from sttw_control.path_command_reward import configured_cost_keys
from sttw_control.preference_trace_rescore import rescore_trace_alpha


ALPHAS = [0., .25, .5, .75, 1.]
ROLES = ['task_only', 'task_constant']


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.worktree.resolve()
    grid = root / 'runs/preference_capability_grid_v1'
    table = grid / 'analysis/all_cases.csv'
    config_file = grid / 'evaluation/common_environment_config.json'
    plan_file = grid / 'evaluation/plan.json'
    plan = json.loads(plan_file.read_text())
    if digest(config_file) != plan['sealed_files'][str(config_file)]:
        raise ValueError('Historical effective config identity changed')
    config = json.loads(config_file.read_text())
    source = list(csv.DictReader(table.open()))
    if len(source) != 90 or config['timing']['lower_dt_s'] != .005:
        raise ValueError('Expected capability90 and its original clock')
    keys = list(configured_cost_keys(config))
    task_indices = [keys.index(k) for k in ['speed', 'path', 'primary']]
    scale_dt = config['path_reward']['scale'] * config['timing']['lower_dt_s']
    records = []; audits = []; raw_ids = {}; identities = set()
    for row in source:
        identity = (row['role'], float(row['alpha']), float(row['force_N']))
        if identity in identities or row['role'] not in ROLES or identity[1] not in ALPHAS:
            raise ValueError('Duplicate or unexpected executed candidate')
        identities.add(identity)
        file = Path(row['raw_file'])
        if digest(file) != row['raw_sha256']:
            raise ValueError('Historical raw identity changed')
        raw_ids[str(file)] = row['raw_sha256']
        with np.load(file, allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        active = raw['active_tick'].astype(bool)
        if not np.array_equal(raw['physical_tick'][active], np.arange(active.sum())):
            raise ValueError('Complete physical prefix required')
        if np.any(active & ~raw['task_reward_valid'].astype(bool)):
            raise ValueError('Held/invalid reward tail cannot be silently ranked')
        if not np.allclose(raw['requested_speed'][active], 1.5, rtol=0, atol=1e-6):
            raise ValueError('Original user speed must remain unchanged')
        fault = active & (raw['nonfinite'] | raw['lower_fault'] | raw['policy_fault'])
        physical = ('unknown' if fault.any() else 'contact' if
                    np.any(active & raw['contact_failure']) else
                    'survived' if active.sum() == 2000 else 'unknown')
        if physical != row['physical_outcome'] or physical == 'unknown':
            raise ValueError('Physical label must be known and authenticated')
        own = rescore_trace_alpha(raw, alpha=identity[1], config=config)
        np.testing.assert_allclose(own['running_reward'][active], raw['tick_reward'][active], rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(own['running_cost'], float(row['running_cost']), rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(own['scored_cost'], float(row['scored_cost']), rtol=3e-5, atol=3e-5)
        audits.append(dict(case_id=row['case_id'], active_ticks=int(active.sum()),
            domain_invalid_ticks=int(np.sum(active & raw['domain_invalid'])),
            task_valid_ticks=int(np.sum(active & raw['task_reward_valid'])),
            max_own_tick_reward_error=float(np.max(abs(own['running_reward'][active]-raw['tick_reward'][active]))),
            own_running_error=own['running_cost']-float(row['running_cost']),
            own_scored_error=own['scored_cost']-float(row['scored_cost'])))
        for user_alpha in ALPHAS:
            score = rescore_trace_alpha(raw, alpha=user_alpha, config=config)
            components = scale_dt * score['effective_costs'][active].sum(axis=0)
            records.append(dict(case_id=row['case_id'], role=row['role'],
                execution_alpha=identity[1], user_alpha=user_alpha, force_N=identity[2],
                physical_outcome=physical, observed_s=float(active.sum()*.005),
                running_cost=score['running_cost'], scored_cost=score['scored_cost'],
                terminal_replacement_delta=score['scored_cost']-score['running_cost'],
                task_channel_cost=float(components[task_indices].sum()),
                other_running_cost=float(components.sum()-components[task_indices].sum()),
                **{k+'_cost': float(v) for k, v in zip(keys, components)}))
    forces = sorted({r['force_N'] for r in records})
    if len(identities) != 90 or len(forces) != 9:
        raise ValueError('Expected full2x5x9 execution panel')
    # A single shared subset across BOTH rules and all five executed alphas.
    common = [f for f in forces if f and all(r['physical_outcome'] == 'survived'
              for r in records if r['force_N'] == f)]
    panels = []
    for scope, selected in [('nominal', [0.]), ('common_survival_disturbed', common)]:
        if not selected:
            raise ValueError('No common complete-survival comparison')
        for metric in ['running_cost', 'task_channel_cost']:
            matrix = []
            for role in ROLES:
                for executed in ALPHAS:
                    costs = []
                    for user in ALPHAS:
                        rr = [r for r in records if r['role'] == role and
                              r['execution_alpha'] == executed and r['user_alpha'] == user and
                              r['force_N'] in selected]
                        if len(rr) != len(selected) or any(r['physical_outcome'] != 'survived' for r in rr):
                            raise ValueError('Rank requires matched full-survival events')
                        costs.append(float(np.mean([r[metric] for r in rr])))
                    matrix.append(dict(role=role, execution_alpha=executed, costs=costs))
            winners = []
            for i, user in enumerate(ALPHAS):
                best = min(matrix, key=lambda r: r['costs'][i])
                winners.append(dict(user_alpha=user, role=best['role'],
                    execution_alpha=best['execution_alpha'], cost=best['costs'][i]))
            within = {}
            for role in ROLES:
                candidates = [r for r in matrix if r['role'] == role]
                within[role] = [dict(user_alpha=user,
                    execution_alpha=min(candidates, key=lambda r:r['costs'][i])['execution_alpha'])
                    for i, user in enumerate(ALPHAS)]
            panels.append(dict(scope=scope, forces_N=selected, metric=metric,
                matrix=matrix, hindsight_best_by_user=winners,
                hindsight_best_within_rule=within))
    args.output.mkdir(exist_ok=False)
    write_csv(args.output/'cross_scores.csv', records)
    write_csv(args.output/'reward_scope_audit.csv', audits)
    result = dict(panels=panels, user_alphas=ALPHAS, common_surviving_forces_N=common,
        sources_sha256={str(p):digest(p) for p in [table, config_file, plan_file,
            Path(__file__), Path(root/'learning/src/sttw_control/preference_trace_rescore.py'),
            Path(root/'learning/src/sttw_control/path_command_reward.py')]},
        raw_sha256=raw_ids, new_physics_steps=0, optimizer_updates=0,
        limits='Retrospective deterministic DEV cross-utility, not new policy execution or a deployable future-aware selector. Full10s historical V3 is not goal-only task success. Task channels exclude roll/effort costs but retain original weights; not a newly validated utility. Common-survival subset is selected on observed outcomes; no population or strong-control superiority inference. All-event scored and terminal costs are diagnostics only; no failure-prefix ranking.')
    (args.output/'review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(common_forces=common, panels=[{k:p[k] for k in
        ['scope','metric','hindsight_best_by_user','hindsight_best_within_rule']} for p in panels]),indent=2))


if __name__ == '__main__':
    main()
