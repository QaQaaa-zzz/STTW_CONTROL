"""Join authenticated existing nominal utility and finite physical panels; no physics."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    inputs = {}

    def read(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())

    nominal_root = root/'runs/nominal_preference_transfer_v1'
    physical_root = root/'runs/midpoint_physical_check_v1'
    nominal = read(nominal_root/'utility/analysis.json')
    physical = read(physical_root/'analysis/analysis.json')
    authenticated = []
    historical = {x['file']: x for x in physical['authenticated_historical_sources']}
    for dataset, field in [(nominal, 'input_sha256'), (physical, 'raw_sha256')]:
        for path, expected in dataset[field].items():
            actual = digest(path)
            if actual != expected:
                entry = historical[path]
                assert entry['recorded_sha256'] == expected
                rel = str(Path(path).relative_to(root))
                data = subprocess.check_output(['git', 'show', entry['source_commit']+':'+rel])
                assert hashlib.sha256(data).hexdigest() == expected
                authenticated.append(dict(path=path, historical_commit=entry['source_commit'],
                                          expected_sha256=expected, live_sha256=actual))
            inputs[path] = actual

    declaration = read(nominal_root/'declaration.json')
    plan = read(nominal_root/'plan.json')
    original = read(root/'runs/formula_prospective_confirmation_v1/declaration.json')
    for field in ['config', 'source', 'routes']:
        assert declaration[field] == original[field], field
    source = declaration['source']
    assert digest(source['prepared_bank']) == source['prepared_bank_sha256']
    for kind in ['actor', 'checkpoint']:
        assert digest(source['candidate'][kind]) == source['candidate'][kind+'_sha256']
    model = plan['model']
    for kind in ['actor', 'checkpoint']:
        assert digest(model[kind]) == model[kind+'_sha256']
    for campaign in ['formula_prospective_confirmation_v1', 'midpoint_physical_check_v1',
                     'new_geometry_zero_control_v1']:
        for side in ['left', 'right']:
            campaign_root = root/'runs'/campaign/side
            prep = read(campaign_root/'prepare_plan.json')
            assert read(campaign_root/'campaign_status.json')['phase'] == 'completed'
            for prefix in read(campaign_root/'prefix_results.json')['cases']:
                assert digest(prefix['snapshot']) == prefix['snapshot_sha256']
            assert Path(prep['source_declaration']).resolve() == root/'runs/formula_prospective_confirmation_v1/declaration.json'
            if campaign != 'new_geometry_zero_control_v1':
                assert prep['model'] == model
            else:
                assert prep['model']['parent_actor_sha256'] == model['actor_sha256']
                assert digest(prep['model']['actor']) == prep['model']['actor_sha256']
                assert prep['verified_zero_inputs'] >= 116
            assert prep['forces_N'] == [0., -80., 80.]
            assert prep['pulse_duration_s'] == .1 and prep['observation_s'] == 4.

    systems = ['zero_middle', 'H75_a0', 'H75_a0.5', 'H75_a1']
    labels = ['zero middle', 'H75 alpha=0', 'H75 alpha=0.5', 'H75 alpha=1']
    colors = ['#555555', '#0072B2', '#D55E00', '#009E73']
    contexts = [(side+'110_R2p6', speed) for side in ['left', 'right'] for speed in [2.35, 2.75]]
    conditions = [(phase, force) for phase in [.2, .7] for force in [-80., 80.]]
    rows = []
    assert len(physical['cases']) == 96 and len(nominal['trajectories']) == 24
    for route, speed in contexts:
        for system in systems:
            n = [x for x in nominal['trajectories'] if (x['route'], x['speed'], x['policy']) == (route, speed, system)]
            assert len(n) == 1
            n = n[0]
            assert n['complete_nominal_horizon'] and not n['physical_failure'] and not n['domain_exit']
            cases = [x for x in physical['cases'] if (x['route'], x['requested_speed'], x['model']) == (route, speed, system)]
            assert len(cases) == 6
            disturbed = []
            for phase, force in conditions:
                matching = [x for x in cases if (x['fraction'], x['force_N']) == (phase, force)]
                assert len(matching) == 1
                case = matching[0]
                safe = case['balance_label'] == 'NO_FLOOR_CONTACT'
                assert case['balance_label'] in ['NO_FLOOR_CONTACT', 'FLOOR_CONTACT']
                assert not case['final_fault']
                if safe:
                    assert case['full_observation'] and not case['event_contact']
                disturbed.append(dict(phase=phase, force_N=force, no_contact=safe,
                                      fallback=case['fallback_triggered'], snapshot=case['snapshot']))
            zero = [x for x in cases if x['force_N'] == 0.]
            assert len(zero) == 2 and all(x['balance_label'] == 'NO_FLOOR_CONTACT' for x in zero)
            rows.append(dict(route=route, speed=speed, system=system, nominal_metrics=n['metrics'],
                             utility_weights=n['utility_alphas'], nominal_returns=n['returns'],
                             nonzero_conditions=disturbed, no_contact_count=sum(x['no_contact'] for x in disturbed),
                             no_contact_with_fallback=sum(x['no_contact'] and x['fallback'] for x in disturbed)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout='constrained')
    for ax, (route, speed) in zip(axes.flat, contexts):
        rr = [x for x in rows if (x['route'], x['speed']) == (route, speed)]
        values = np.array([[int(c['no_contact']) for c in x['nonzero_conditions']] for x in rr])
        ax.imshow(values, cmap=ListedColormap(['#EEEEEE', '#0072B2']), vmin=0, vmax=1, aspect='auto')
        for i, row in enumerate(rr):
            for j, case in enumerate(row['nonzero_conditions']):
                ax.text(j, i, ('S' if case['no_contact'] else 'F')+('*' if case['fallback'] else ''),
                        ha='center', va='center', color='white' if case['no_contact'] else 'black')
        ax.set_yticks(range(4), labels)
        ax.set_xticks(range(4), ['.2 / -80', '.2 / +80', '.7 / -80', '.7 / +80'])
        ax.set_xlabel('turn fraction / signed force (N)')
        ax.set_title(f'{route}, requested {speed:g} m/s')
    fig.suptitle('Existing finite push panel: S = no contact for 4 s; F = contact\n* = domain-hold used; each cell is one deterministic branch, not a probability')
    fig.savefig(out/'physical_panel.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, (route, speed) in zip(axes.flat, contexts):
        for system, label, color, marker in zip(systems, labels, colors, ['o', 's', '^', 'D']):
            row = next(x for x in rows if (x['route'], x['speed'], x['system']) == (route, speed, system))
            ax.plot(row['utility_weights'], row['nominal_returns'], marker=marker, color=color,
                    label=f'{label}; push S={row["no_contact_count"]}/4')
        ax.set_title(f'{route}, requested {speed:g} m/s')
        ax.set_xlabel('utility weight w (rescoring the same nominal trajectory)')
        ax.set_ylabel('nominal 10 s return (higher is better)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Nominal preference utility alongside separately measured push outcomes\nNo online policy selection; no disturbed full-task return inferred')
    fig.savefig(out/'nominal_utility.png', dpi=180)
    plt.close(fig)
    # Compare condition identities, not just totals. This is a descriptive menu:
    # policy-specific prefixes are different, and no action is chosen online.
    comparisons, frontiers = [], []
    utility_tie = 1e-4  # above the recorded nominal reconstruction error; not a task gate
    for route, speed in contexts:
        rr = [x for x in rows if (x['route'], x['speed']) == (route, speed)]
        passed = {x['system']: {(c['phase'], c['force_N']) for c in x['nonzero_conditions'] if c['no_contact']} for x in rr}
        for wi, weight in enumerate([0., .25, .5, .75, 1.]):
            dominated = set()
            for a in rr:
                for b in rr:
                    if a is b:
                        continue
                    sa, sb = passed[a['system']], passed[b['system']]
                    delta = a['nominal_returns'][wi]-b['nominal_returns'][wi]
                    dominates = sa >= sb and delta >= -utility_tie and (sa > sb or delta > utility_tie)
                    if dominates:
                        dominated.add(b['system'])
                    comparisons.append(dict(route=route, speed=speed, utility_weight=weight,
                                            candidate=a['system'], comparator=b['system'],
                                            nominal_utility_difference=delta,
                                            gain_conditions=sorted(sa-sb), loss_conditions=sorted(sb-sa),
                                            no_contact_set_superset=sa >= sb,
                                            descriptive_joint_dominance=dominates))
            frontiers.append(dict(route=route, speed=speed, utility_weight=weight,
                                  nondominated_systems=[x for x in systems if x not in dominated]))
    result = dict(rows=rows, comparisons=comparisons, descriptive_frontiers=frontiers,
                  utility_numerical_tie=utility_tie,
                  inputs_live_sha256=inputs, authenticated_historical_sources=authenticated,
                  join_contract=dict(config_equal=True, lower_source_equal=True, route_definitions_equal=True,
                                     H75_identity_equal=True, same_snapshot_across_alpha=False,
                                     initial_state_live_replay_performed=False),
                  source_sha256=digest(__file__), new_physics_ticks=0, optimizer_updates=0,
                  scope='Retrospective DEV descriptive map. Four correlated nonzero conditions per cell; '
                        'not success probabilities, continuous force thresholds, alpha-switch causality, or safety certification. '
                        'Nominal utility and disturbed survival come from different rollouts; all alpha0.25/0.75 physical outcomes unmeasured.')
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), joined_rows=len(rows), physical_cases=96,
                          nominal_trajectories=24, authenticated_historical_sources=len(authenticated),
                          new_physics_ticks=0)))


if __name__ == '__main__':
    main()
