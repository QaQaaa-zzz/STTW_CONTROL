"""Declare one finite alpha-dependent physical-boundary diagnostic."""
import argparse
import json
import sys
from pathlib import Path
from sttw_control.preference_best import atomic_json, digest
from sttw_control.feature_comparison_evaluation import authenticated_parent_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--route', choices=['left125_R2p2', 'right125_R2p2'], default='left125_R2p2')
    p.add_argument('--forces', type=float, nargs='+', default=[0., -48., 48., -80., 80., -112., 112.])
    p.add_argument('--design', default=str(Path(__file__).with_name('DESIGN.md')))
    a = p.parse_args()
    base, root = Path.cwd(), Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    old = base/'runs/reserve_feature_expansion_v1/post_fixed_final_v1'
    model = json.loads((old/'models.json').read_text())['records']['H75']
    authenticated_parent_record(model)
    declaration = old/'high_nominal_retention/H75/declaration.json'
    if 0. not in a.forces or len(set(a.forces)) != len(a.forces):
        raise ValueError('Unique force grid with zero control required')
    design = Path(a.design).resolve()
    files = [Path(__file__), design,
             Path(__file__).with_name('collect_prefixes.py'),
             Path(__file__).with_name('execute.py'), Path(__file__).with_name('analyze.py'), declaration,
             base/'learning/cli/probe_sensor_gate_domain_hold.py']
    files += [base/'learning/src/sttw_control'/f'{name}.py' for name in
              ['path_command_policy', 'path_command_training', 'path_command_env',
               'reference_reserve_gate', 'balance_probe', 'controller', 'closed_loop_kernel']]
    sha = {str(file.resolve()): digest(file) for file in files}
    sha[model['actor']] = model['actor_sha256']
    sha[model['checkpoint']] = model['checkpoint_sha256']
    plan = dict(project='STTW_CONTROL', owner='Codex tolerance-calibration',
                source_declaration=str(declaration), model=model,
                path_evidence_directory=str(old/'high_nominal_retention/H75/evaluation'),
                route=a.route, builder='left90_R2' if a.route.startswith('left') else 'right90_R2', speeds=[2.45, 2.65],
                alphas=[0., 1.], phase_fractions=[.2, .7],
                forces_N=a.forces,
                pulse_duration_s=.1, observation_s=4., original_task_horizon_s=10.,
                declared_prefixes=8, declared_cases=8*len(a.forces),
                maximum_prepare_lower_ticks=8000, maximum_scan_lower_ticks=8*len(a.forces)*800,
                maximum_total_lower_ticks=8000+8*len(a.forces)*800, training_steps=0,
                design=str(design), design_sha256=digest(design),
                source_sha256=sha, prefix_source='actual frozen upper rollouts; complete state/history',
                stop_rule='Finite grid once; no automatic force extension or engineering retry')
    atomic_json(root/'prepare_plan.json', plan)
    atomic_json(root/'prepare_status.json', dict(phase='planned', completed_cases=0, declared_cases=8, active_lower_ticks=0, training_steps=0))
    atomic_json(root/'status.json', dict(phase='planned', completed_cases=0, declared_cases=plan['declared_cases'], active_lower_ticks=0, training_steps=0))
    atomic_json(root/'campaign_status.json', dict(phase='prepared', completed_cases=0, declared_cases=plan['declared_cases'], training_steps=0))
    sys.path.insert(0, str(base/'learning/cli'))
    from prepare_short_recovery_campaign import register
    for name, title, status in [('ACTIVE_PREPARE', '真实几何相位前缀', 'prepare_status.json'),
                                ('ACTIVE_SCAN', '有符号防摔力档', 'status.json'),
                                ('ACTIVE_CAMPAIGN', 'α偏好与物理容扰度总控', 'campaign_status.json')]:
        register(root/(name+'.json'), 'STTW_CONTROL '+title+' '+root.name, root/status)
    print(root, plan['declared_prefixes'], 'prefixes', plan['declared_cases'], 'branches;', plan['maximum_total_lower_ticks'], 'maximum lower ticks;0 training')


if __name__ == '__main__':
    main()
