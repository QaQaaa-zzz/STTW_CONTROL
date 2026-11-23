"""CPU-only audit of the archived raised-platform planning contracts."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'docs/chapter3_paper'))
import run_journal_closure_experiments as closure

OUT = REPO / 'docs/evidence/chapter3_continuous_planning/p0_a'
CACHE = REPO / 'runs/chapter3_continuous_p0_audit'
LAYOUTS = ('config_r', 'config_double', 'config_l', 'test_a', 'test_b')
KEY = ['layout', 'root', 'force', 'section', 'side']
ACTION = ['approach', 'approach_alpha', 'planned_dep']


def panel(layout):
    folders = ['alpha_library'] if layout.startswith('test_') else ['veto_fallback', 'veto_fallback_followup']
    frames = [pd.read_csv(closure.PR / folder / 'oracle_cases.csv') for folder in folders]
    cases = pd.concat(frames, ignore_index=True)
    cases = cases[(cases.layout == layout) & (cases.skill == 'rb_run1')].copy()
    source = closure.PR / folders[0]
    settings = json.loads((source / 'test_freeze_manifest.json').read_text())
    return cases, source, settings


def cache_forecasts():
    CACHE.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        target = CACHE / f'{layout}.pkl'
        if target.exists():
            print('using cached', layout, flush=True)
            continue
        cases, source, settings = panel(layout)
        prefixes = cases[(cases.planned_dep == 0) & (cases.section == 'near') & (cases.side == 1) & cases.verified].drop_duplicates('prefix')
        ev = closure.CpuStage2Evaluator(source / layout / 'part_0/rb_run1', settings['section_x'])
        started = time.monotonic()
        forecasts = {}
        # A verified handback can terminate before the next control tick. Such an
        # episode has real end states, but the forecast was never actionable.
        for row in prefixes[prefixes.departure_applied_tick < 0].itertuples():
            tick = round(row.handoff_s / 0.005) + 1
            with np.load(row.trajectory) as z:
                if tick >= len(z['qpos']):
                    forecasts[row.prefix] = dict(valid_horizon=False, q0=None, v0=None,
                                                  unavailable='ended_before_query')
        prefixes = prefixes[~prefixes.prefix.isin(forecasts)]
        groups = prefixes.groupby(np.rint(prefixes.handoff_s / 0.005).astype(int))
        print(layout, 'verified prefixes', len(prefixes), 'tick groups', groups.ngroups, flush=True)
        for i, (_, group) in enumerate(groups, 1):
            forecasts.update(ev.evaluate_prefixes(group))
            if i == 1 or i % 15 == 0 or i == groups.ngroups:
                print(layout, 'groups', i, '/', groups.ngroups, 'prefixes', len(forecasts), 'elapsed_s', round(time.monotonic()-started,1), flush=True)
        with target.with_suffix('.partial').open('wb') as fh:
            pickle.dump(forecasts, fh)
        target.with_suffix('.partial').replace(target)


def decision(forecast, section, side, bank, route, state, incumbent, margin=closure.EPS_RB):
    """Run the archived downstream policy, including its unverified/no-query path."""
    if forecast is None or not forecast['valid_horizon']:
        return incumbent, incumbent, None
    mask = (forecast['clear'] > margin) & forecast['pass_goal'] & forecast['gates'][(section, side)]
    stage2, admitted = closure.stage2_decision(forecast, section, side, incumbent, margin)
    chosen = stage2
    if bank is not None and not admitted:
        chosen = closure.veto_choose_variant(bank, state, route, closure.SECTIONS.index(section),
                                             int(side > 0), stage2)
    return stage2, chosen, mask


def endpoint(row):
    if row.success:
        return 'completed'
    if row.termination == 'goal':
        return 'goal_with_gate_unmet' if row.verified else 'goal_without_verified_jump'
    return row.termination


def trace_fields(row, tag):
    fields = dict(trajectory=row.trajectory, success=bool(row.success), wall=bool(row.collision),
                  verified=bool(row.verified), handoff_s=row.handoff_s if row.verified else np.nan,
                  handoff_frame=int(round(row.handoff_s / 0.005)) + 1 if row.verified else pd.NA,
                  termination=row.termination, endpoint=endpoint(row), approach=row.approach,
                  alpha=row.approach_alpha, dep=row.planned_dep, prefix=row.prefix,
                  departure_applied_tick=row.departure_applied_tick)
    return {f'{tag}_{name}': val for name, val in fields.items()}


def encode_mask(values):
    return sum(1 << j for j, yes in enumerate(values) if yes)


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    three_wall = set(LAYOUTS[:3])
    banks = closure.build_twosided_veto_banks()
    bank_manifest = json.loads((closure.PR / 'veto_fallback/test_freeze_manifest.json').read_text())
    with np.load(bank_manifest['bank']) as initial:
        initial_bank = dict(initial)
    contracts, alternatives, historical, parity_rows = [], [], [], []
    for layout in LAYOUTS:
        cases, source, cfg = panel(layout)
        with (CACHE / f'{layout}.pkl').open('rb') as f:
            forecasts = pickle.load(f)
        states = {key: closure.handback_features(torch.tensor(v['q0'][None]), torch.tensor(v['v0'][None])).float().numpy()[0]
                  for key, v in forecasts.items() if v['valid_horizon']}
        bank = banks.get(layout)
        roots = sorted(cases.root.unique())
        prior = pd.read_csv(source / layout / 'rb_run1_prior.csv')
        plans = closure.stage1_actions_twosided(prior, initial_bank, roots, closure.EPS_RB)
        folders = ['veto_fallback', 'veto_fallback_followup'] if layout in three_wall else ['alpha_library']
        actual = pd.concat([pd.read_csv(closure.PR / f / 'online_cases.csv') for f in folders], ignore_index=True)
        actual = actual[(actual.layout == layout) & (actual.skill == 'rb_run1')]
        online_rows = {(r.root, r.force, r.section, r.side, r.planner): r for r in actual.itertuples()}
        assert len(online_rows) == len(actual)
        for key, contract in cases.groupby(KEY, sort=False):
            _, root, force, section, side = key
            ident = dict(zip(KEY, key))
            choices = plans[(root, section, side)]
            branches = {}
            for row in contract.itertuples():
                branches[(row.approach, row.approach_alpha, row.planned_dep)] = row
            assert len(branches) == len(contract) == 15
            evaluated = {}
            for route, alpha in closure.ACTIONS:
                outcomes = [branches[route, alpha, dep] for dep in range(3)]
                p = outcomes[0]
                assert len({o.prefix for o in outcomes}) == 1 and len({o.verified for o in outcomes}) == 1
                rec = forecasts[p.prefix] if p.verified else None
                for inc in range(3):
                    action = (route, alpha, inc)
                    ch2, chosen, mask = decision(rec, section, side, bank, route, states.get(p.prefix), inc)
                    evaluated[action] = (ch2, chosen, mask)
                    alternatives.append(dict(**ident, approach=route, alpha=alpha, incumbent=inc,
                                             selected_dep=chosen, won=int(outcomes[chosen].success),
                                             wall=int(outcomes[chosen].collision),
                                             raw_branch_won=int(outcomes[inc].success)))
            fixed_act = choices['fixed']
            enhanced_act = choices['alpha_enabled']
            alpha0_act = choices['alpha0']
            route, alpha, inc = enhanced_act
            ch2, chosen, mask = evaluated[enhanced_act]
            fixed = branches[fixed_act]
            selected = branches[route, alpha, chosen]
            same = [branches[route, alpha, d] for d in range(3)]
            post_wins = np.array([b.success for b in same], dtype=bool)
            post_walls = np.array([b.collision for b in same], dtype=bool)
            viable = bool(post_wins.any())
            valid_query = mask is not None
            veto = valid_query and not mask.any()
            is_lost = fixed.success and not selected.success
            rescued = not fixed.success and selected.success
            fixed_post = evaluated[fixed_act][1]
            rec = forecasts[selected.prefix] if selected.verified else None
            margin_zero = decision(rec, section, side, bank, route, states.get(selected.prefix), inc, margin=0.)
            zero_alt = branches[route, alpha, margin_zero[1]]
            prev_inc_won = branches[route, alpha, inc].success
            args = dict(**ident,
                        suite='three_wall' if bank is not None else 'test_ab',
                        selected_planner='Full 3-Stage' if bank is not None else 'C-R (Exp. a)',
                        fixed_incumbent=fixed_act[2], selected_incumbent=inc, selected_stage2_dep=ch2,
                        selected_query_valid=valid_query, selected_all_veto=bool(veto),
                        selected_stage3_used=bank is not None and veto,
                        selected_stage3_switch=ch2 != chosen, same_approach=fixed_act[:2] == enhanced_act[:2],
                        paired_lost=int(is_lost), paired_rescued=int(rescued),
                        selected_any_old_exit_wins=viable if selected.verified else pd.NA,
                        selected_all_old_exits_hit_wall=bool(post_walls.all()) if selected.verified else pd.NA,
                        actual_success_mask=encode_mask(post_wins) if selected.verified else pd.NA,
                        admitted_mask=encode_mask(mask) if valid_query else pd.NA,
                        actual_wall_mask=encode_mask(post_walls) if selected.verified else pd.NA,
                        falsely_vetoed_feasible_branch_mask=encode_mask(post_wins & ~mask) if valid_query else pd.NA,
                        solvable_all_veto=bool(veto and viable),
                        post_veto_collision=bool(veto and selected.collision),
                        chosen_exit_missed_solution=bool(selected.verified and not selected.success and viable),
                        selected_entry_3_exits_fail=bool(selected.verified and not viable),
                        any_other_old_entry_wins=any(r.success for k, r in branches.items() if k[:2] != enhanced_act[:2]),
                        fixed_pre_same_post_success=bool(branches[*fixed_act[:2], fixed_post].success),
                        selected_pre_precommitted_exit_success=bool(prev_inc_won),
                        margin_zero_full_success=bool(zero_alt.success),
                        margin_zero_full_collision=bool(zero_alt.collision),
                        margin_zero_valid_query=margin_zero[2] is not None,
                        margin_zero_full_exit=margin_zero[1],
                        margin_zero_admitted_mask=encode_mask(margin_zero[2]) if margin_zero[2] is not None else pd.NA,
                        **trace_fields(fixed, 'fixed'), **trace_fields(selected, 'selected'))
            contracts.append(args)
            historic = [('Fixed (a=0)', fixed), ('C-R (Exp. a)', branches[route, alpha, ch2])]
            historic.append(('C-R (a=0)', branches[*alpha0_act[:2], evaluated[alpha0_act][0]]))
            if bank is not None:
                historic.append(('Full 3-Stage', selected))
            historical.extend(dict(**ident, system=label, success=int(r.success), wall=int(r.collision))
                              for label, r in historic)
            if side == 1:
                onarms = [('expanded', enhanced_act), ('veto_bank', enhanced_act)] if bank is not None else [
                    ('alpha0', alpha0_act), ('alpha_enabled', enhanced_act)]
                for arm, planned in onarms:
                    log = online_rows[root, force, section, side, arm]
                    # Prefix identifiers of online rows need not share the oracle hash. Locate by action.
                    p = branches[log.approach, log.approach_alpha, 0]
                    f_rec = forecasts[p.prefix] if p.verified else None
                    s2, s3, mask_log = decision(f_rec, section, side, bank, log.approach,
                                                states.get(p.prefix), log.planned_dep)
                    real_choice = s3 if arm == 'veto_bank' else s2
                    check = bool(log.verified and log.query_batch_ms > 0)
                    out = branches[log.approach, log.approach_alpha, real_choice]
                    parity_rows.append(dict(**ident, arm=arm, checked_query=check,
                        checked_veto=check and mask_log is not None and not mask_log.any(),
                        plan_mismatch=tuple(planned) != (log.approach, log.approach_alpha, log.planned_dep),
                        decision_mismatch=real_choice != log.chosen_dep,
                        admission_mismatch=bool(mask_log is not None and mask_log.any()) != bool(log.query_feasible),
                        outcome_mismatch=out.success != log.success or out.collision != log.collision))
        print('assembled', layout, len(contracts), flush=True)
    trial = pd.DataFrame(contracts).sort_values(KEY).reset_index(drop=True)
    assert len(trial) == 6840 and not trial.duplicated(KEY).any()
    # Make the trajectory index available to the independent plotting worker immediately.
    trial.to_csv(OUT / 'contracts.partial', index=False)
    (OUT / 'contracts.partial').replace(OUT / 'contracts.csv')
    h = pd.DataFrame(historical)
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(OUT / 'online_parity.csv.gz', index=False)
    b = pd.DataFrame(alternatives)
    commitment_keys = ['layout','root','section','side']
    action_keys = ['approach','alpha','incumbent']
    assert not b.duplicated(KEY + action_keys).any()
    certificate = b.pivot(index=commitment_keys + action_keys, columns='force', values='won')
    certificate = certificate[[-5, 0, 5]].set_axis(['win_minus5', 'win_zero', 'win_plus5'], axis=1)
    assert certificate.isin([0, 1]).all().all()
    certificate.to_csv(OUT / 'finite_library_actions.csv.gz', compression={'method':'gzip','mtime':0})
    shared = b.groupby(commitment_keys + action_keys, as_index=False).agg(won=('won','sum'),
                    raw_branch_won=('raw_branch_won','sum'), trials=('force','size'))
    assert (shared.trials == 3).all()
    shared = shared.sort_values(commitment_keys + ['won'] + action_keys,
                                ascending=[True]*4 + [False,True,True,True])
    best = shared.drop_duplicates(commitment_keys).rename(columns={'won':'best_pre_shared_wins', 'trials':'n_forces',
                'approach':'best_approach', 'alpha':'best_alpha','incumbent':'best_incumbent'}).drop(columns='raw_branch_won')
    hindsight = b.groupby(KEY, as_index=False).agg(branch_if_force_known=('won','max'),
                        best_raw_branch=('raw_branch_won','max'))
    counts = hindsight.groupby(commitment_keys, as_index=False).agg(
        best_pre_force_known_wins=('branch_if_force_known','sum'), best_physical_branch_wins=('best_raw_branch','sum'))
    measured = trial.groupby(commitment_keys, as_index=False).agg(actual_wins=('selected_success','sum'))
    bounds = best.merge(counts, on=commitment_keys, validate='one_to_one').merge(measured, on=commitment_keys, validate='one_to_one')
    assert (bounds.actual_wins <= bounds.best_pre_shared_wins).all()
    assert (bounds.best_pre_shared_wins <= bounds.best_pre_force_known_wins).all()
    assert (bounds.best_pre_force_known_wins <= bounds.best_physical_branch_wins).all()
    bounds.to_csv(OUT / 'bounds_by_root.csv', index=False)
    bounds.groupby(['layout','side'])[['n_forces','actual_wins','best_pre_shared_wins','best_pre_force_known_wins','best_physical_branch_wins']].sum().to_csv(OUT / 'bounds.csv')
    summary = trial.groupby(['suite','layout','side']).agg(n=('selected_success','size'),
        fixed_wins=('fixed_success','sum'), selected_wins=('selected_success','sum'),
        fixed_walls=('fixed_wall','sum'), selected_walls=('selected_wall','sum'),
        losses=('paired_lost','sum'), rescues=('paired_rescued','sum'),
        selected_verified=('selected_verified','sum'), valid_queries=('selected_query_valid','sum'),
        all_veto=('selected_all_veto','sum'), solvable_all_veto=('solvable_all_veto','sum'),
        all_old_exits_hit_wall=('selected_all_old_exits_hit_wall','sum'), veto_walls=('post_veto_collision','sum'),
        same_entry_missed=('chosen_exit_missed_solution','sum'),
        selected_pre_precommitted_wins=('selected_pre_precommitted_exit_success','sum'),
        fixed_pre_same_post_wins=('fixed_pre_same_post_success','sum'), margin_zero_wins=('margin_zero_full_success','sum'),
        margin_zero_walls=('margin_zero_full_collision','sum'))
    summary.to_csv(OUT / 'paired_summary.csv')
    h.groupby(['layout','side','system']).agg(n=('success','size'),done=('success','sum'),wall=('wall','sum')).to_csv(OUT / 'historical_summary.csv')
    trial.groupby(['layout','side','selected_endpoint','fixed_endpoint'],dropna=False).size().rename('n').to_csv(OUT / 'paired_endpoints.csv')
    grouped = trial.groupby(['layout','side','paired_lost','paired_rescued'])
    grouped[['selected_success','selected_verified','selected_all_veto','solvable_all_veto',
            'chosen_exit_missed_solution','post_veto_collision','selected_entry_3_exits_fail',
            'any_other_old_entry_wins','selected_stage3_switch']].sum().to_csv(OUT / 'overlapping_tags.csv')
    reference = pd.read_csv(closure.OUT / 'exp3_twosided_wall_layouts.csv')
    wall_counts = h[h.layout.isin(three_wall)].groupby(['layout','side','system']).agg(done=('success','sum'),wall=('wall','sum'))
    for r in reference[reference.scope.isin(three_wall) & reference.gate.str.startswith(('Right','Left'))].itertuples():
        assert tuple(wall_counts.loc[(r.scope, 1 if r.gate.startswith('Left') else -1, r.system)]) == (r.done,r.wall)
    reference = pd.read_csv(closure.OUT / 'exp3_twosided_testab.csv')
    ab = h[~h.layout.isin(three_wall)].groupby(['side','system']).agg(done=('success','sum'),wall=('wall','sum'))
    for r in reference[reference.gate.str.startswith(('Right','Left'))].itertuples():
        assert tuple(ab.loc[1 if r.gate.startswith('Left') else -1, r.system]) == (r.done,r.wall)
    right = trial[(trial.side == -1) & trial.layout.isin(three_wall)]
    assert len(right)==2700 and right.paired_lost.sum()-right.paired_rescued.sum()==290
    assert (right.groupby('layout').selected_success.sum().to_dict() == {'config_double':576,'config_l':693,'config_r':492})
    queried = parity[parity.checked_query]
    assert not queried.decision_mismatch.any() and not queried.admission_mismatch.any()
    parity_summary = parity.groupby(['layout','arm']).agg(n=('arm','size'), queries=('checked_query','sum'),
        vetoes=('checked_veto','sum'), plan_mismatches=('plan_mismatch','sum'), decisions=('decision_mismatch','sum'),
        admissions=('admission_mismatch','sum'), outcomes=('outcome_mismatch','sum'))
    parity_summary.to_csv(OUT / 'online_parity_summary.csv')
    assert not parity[['plan_mismatch','decision_mismatch','admission_mismatch','outcome_mismatch']].any().any()
    write_receipt()
    audit_right_losses()
    print(summary.to_string(), flush=True)
    print(parity_summary.to_string(), flush=True)
    print('P0-A complete', flush=True)


def audit_right_losses():
    """Audit frozen admission tests on physically successful exits from each lost right-gate case."""
    trial = pd.read_csv(OUT / 'contracts.csv')
    loss = trial[(trial.suite == 'three_wall') & (trial.side == -1) & (trial.paired_lost == 1)]
    details, cases = [], []
    for layout, group in loss.groupby('layout'):
        with (CACHE / f'{layout}.pkl').open('rb') as f:
            forecasts = pickle.load(f)
        for row in group.itertuples():
            ident = {k: getattr(row, k) for k in KEY}
            choices = []
            if row.selected_query_valid:
                rec = forecasts[row.selected_prefix]
                for dep in range(3):
                    if not int(row.actual_success_mask) & (1 << dep):
                        continue
                    clear = float(rec['clear'][dep])
                    goal = bool(rec['pass_goal'][dep])
                    gate = bool(rec['gates'][row.section, row.side][dep])
                    admitted = bool(int(row.admitted_mask) & (1 << dep))
                    zero = clear > 0 and gate and goal
                    assert admitted == (clear > closure.EPS_RB and gate and goal)
                    choices.append(dict(**ident, dep=dep, clear_m=clear, gate_pass=gate,
                        goal_hold_and_domain_pass=goal, fails_gate=not gate, fails_goal_hold_or_domain=not goal,
                        clearance_nonpositive=clear <= 0, clearance_positive_within_margin=0 < clear <= closure.EPS_RB,
                        clearance_exceeds_margin=clear > closure.EPS_RB, admitted=admitted,
                        admitted_at_zero_margin=zero, restored_at_zero_margin=zero and not admitted))
            details.extend(choices)
            # Each physical success branch is evaluated separately; only "any" columns
            # are then reduced over successful exits. Non-geometric failures may overlap.
            if not row.selected_verified:
                diagnosis = 'no_verified_relock'
            elif not row.selected_any_old_exit_wins:
                diagnosis = 'none_of_three_exits_physically_successful'
            elif not row.selected_query_valid:
                diagnosis = 'successful_exit_without_actionable_query'
            elif any(c['admitted'] for c in choices):
                diagnosis = 'successful_exit_admitted_but_another_exit_chosen'
            elif any(c['restored_at_zero_margin'] for c in choices):
                diagnosis = 'successful_exit_becomes_admissible_at_zero_margin'
            else:
                diagnosis = 'no_successful_exit_admissible_even_at_zero_margin'
            cases.append(dict(**ident, diagnosis=diagnosis, selected_prefix=row.selected_prefix,
                fixed_trajectory=row.fixed_trajectory, selected_trajectory=row.selected_trajectory,
                all_veto=bool(row.selected_all_veto), actual_success_mask=row.actual_success_mask,
                admitted_mask=row.admitted_mask, selected_dep=row.selected_dep,
                physical_successful_exits=len(choices), full_wins_at_zero_margin=bool(row.margin_zero_full_success),
                any_successful_exit_nonpositive_clearance=any(c['clearance_nonpositive'] for c in choices),
                any_successful_exit_positive_within_margin=any(c['clearance_positive_within_margin'] for c in choices),
                any_successful_exit_fails_gate=any(c['fails_gate'] for c in choices),
                any_successful_exit_fails_goal_or_domain=any(c['fails_goal_hold_or_domain'] for c in choices)))
    case_table, exit_table = pd.DataFrame(cases), pd.DataFrame(details)
    assert len(case_table) == 295 and (case_table.physical_successful_exits > 0).sum() == 256
    case_table.to_csv(OUT / 'lost_right_case_causes.csv', index=False)
    exit_table.to_csv(OUT / 'lost_right_feasible_exit_predicates.csv', index=False)
    summary = case_table.groupby(['layout','diagnosis','all_veto'], as_index=False).agg(
        cases=('diagnosis','size'), full_wins_at_zero_margin=('full_wins_at_zero_margin','sum'))
    summary.to_csv(OUT / 'lost_right_causes_summary.csv', index=False)


def write_receipt():
    trials = pd.read_csv(OUT / 'contracts.csv')
    parity = pd.read_csv(OUT / 'online_parity.csv.gz')
    historical = pd.read_csv(OUT / 'historical_summary.csv')
    counts = pd.read_csv(OUT / 'bounds_by_root.csv')
    gaps = []
    for layout in LAYOUTS:
        with (CACHE / f'{layout}.pkl').open('rb') as f:
            forecasts = pickle.load(f)
        for prefix, rec in forecasts.items():
            if not rec['valid_horizon']:
                gaps.append(dict(layout=layout, prefix=prefix,
                                 reason=rec.get('unavailable','insufficient_prediction_horizon')))
    mismatch_cols = ['plan_mismatch','decision_mismatch','admission_mismatch','outcome_mismatch']
    receipt = dict(
        mode='CPU replay on archived measured trajectories',
        new_physics_rollouts=0, training_updates=0, gpu_rollouts=0,
        archived_post_review=str(closure.PR), reference_tables=str(closure.OUT),
        panel_contracts=trials.suite.value_counts().to_dict(), counted_precommitment_groups=len(counts),
        branch_count_per_task=15, archived_summary_groups_checked=len(historical),
        online_left_arm_rows=len(parity), online_mismatches=parity[mismatch_cols].sum().to_dict(),
        physical_prefixes_unavailable_at_query=gaps,
        trajectory_fields=['fixed_trajectory','selected_trajectory'],
        planning_reference_fields=['fixed_dep','selected_dep','departure_applied_tick (prefixed)'],
        scope='Three-wall selected=Full including Stage 3; A/B selected=C-R expanded. Both right panels reconstructed. All online parity reference rows are left gate.')
    (OUT / 'audit_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', action='store_true', help='Recompute missing per-layout CPU forecast caches only.')
    if parser.parse_args().cache:
        cache_forecasts()
    else:
        cache_forecasts()
        audit()
