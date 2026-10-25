"""Seal the endpoint case matrix only: no physics, checkpoint selection or training."""
import argparse
import json
from pathlib import Path

from sttw_control.preference_best import atomic_json, digest
from sttw_control.survival_endpoint_protocol import make_endpoint_cases


def prepare(training_run, nominal_declaration, output, *, seed):
    plan_path = training_run/'plan.json'
    plan = json.loads(plan_path.read_text())
    for key in ('source_plan', 'source_declaration'):
        if digest(plan[key]) != plan[key+'_sha256']:
            raise ValueError(f'changed training input: {key}')
    effective_path = training_run/'effective_config.json'
    if digest(effective_path) != plan['effective_config_sha256']:
        raise ValueError('changed effective training config')
    if plan['updates'] != 64 or plan['critic_only_updates'] != 2 or plan['epochs'] != 4:
        raise ValueError('unexpected training schedule')
    reference = json.loads(Path(plan['source_plan']).read_text())['cases']
    nominal = json.loads(nominal_declaration.read_text())
    nominal_contexts = [dict(route=r['name'], speed=v, force_N=0.)
                        for r in nominal['routes'] for v in nominal['speeds']]
    cases = make_endpoint_cases(reference, nominal_contexts)
    config = json.loads(effective_path.read_text())
    # Geometry is supplied separately. Old nominal permissions and reward
    # templates are not substituted for the active common environment config.
    geometry = dict(routes=nominal['routes'], path=nominal['config']['path'])
    files = [plan_path, effective_path, Path(plan['source_plan']),
             Path(plan['source_declaration']), nominal_declaration, Path(__file__),
             Path(__import__('sttw_control.survival_endpoint_protocol', fromlist=['x']).__file__)]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'cases.json', cases)
    atomic_json(output/'common_environment_config.json', config)
    atomic_json(output/'nominal_geometry.json', geometry)
    atomic_json(output/'plan.json', dict(
        project='STTW_CONTROL', purpose='Fixed endpoint preference/physical comparison',
        stage='case_matrix_prepared_only', training_run=str(training_run),
        required_completed_updates=64, roles=['parent', 'task_only', 'physical_credit',
                                            'parent_speed_floor', 'zero_middle'],
        seed=seed, data_role='DEV', horizon_s=10., lower_dt_s=.005,
        declared_actual_episodes=len(cases), maximum_lower_ticks=len(cases)*2000,
        optimizer_updates=0, physics_steps_executed=0, evaluation_launched=False,
        cases_sha256=digest(output/'cases.json'),
        common_config_sha256=digest(output/'common_environment_config.json'),
        nominal_geometry_sha256=digest(output/'nominal_geometry.json'),
        source_sha256={str(p.resolve()): digest(p) for p in files},
        pending=['authenticate terminal native checkpoints and exports',
                 'connect explicit inference modes and speed floor to physical collector',
                 'execute panels and publish all XY/reward/error diagnostics'],
        guard='No intermediate winner, auto-launch, extra training or safe-interval claim.'))
    atomic_json(output/'status.json', dict(phase='prepared', stage='case_matrix_only',
                completed_cases=0, declared_cases=len(cases), active_lower_ticks=0,
                optimizer_updates=0, endpoint_authenticated=False))
    print(output/'plan.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True, type=Path)
    parser.add_argument('--nominal-declaration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    prepare(args.training_run.resolve(), args.nominal_declaration.resolve(),
            args.output.resolve(), seed=args.seed)
