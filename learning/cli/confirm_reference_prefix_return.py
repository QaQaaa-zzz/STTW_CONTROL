"""Prepare and run one frozen bank0/7 confirmation, with no candidate reselection."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
AUDIT = BASE / 'docs/evidence/tolerance_progress_20261022/baseline_provenance_audit'
COLLECTOR = AUDIT.parent / 'alpha_physical_boundary/collect_prefixes.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    import jax
    import numpy as np
    from sttw_control.path_command_training import load_environment
    from sttw_control.geometric_path import build_path
    from sttw_control.preference_best import atomic_json
    if (root / 'plan.json').exists():
        raise FileExistsError('confirmation already prepared')
    design = json.loads((root / 'design.json').read_text())
    for key in ('prepared_bank', 'source_declaration', 'source_results'):
        if digest(design[key]) != design[key + '_sha256']:
            raise ValueError('frozen design identity mismatch: ' + key)
    source = json.loads(Path(design['source_declaration']).read_text())
    if (source['source']['prepared_bank'] != design['prepared_bank'] or
            source['source']['prepared_bank_sha256'] != design['prepared_bank_sha256']):
        raise ValueError('design and actual source bank differ')
    banks = []
    files = [Path(__file__), COLLECTOR, BASE / 'learning/cli/probe_reference_prefix_return.py',
             AUDIT / 'report_reference_confirmation.py', root / 'design.json',
             Path(design['source_declaration']), Path(design['prepared_bank'])]
    files += sorted((BASE / 'learning/src/sttw_control').glob('*.py'))
    files += [Path(source['policies']['H75'][key]) for key in ('actor', 'checkpoint')]
    files += [Path(source['source']['candidate'][key]) for key in ('actor', 'checkpoint', 'config')]
    prefix_plans = []
    for index in design['bank_indices']:
        bank = root / f'bank{index}'
        bank.mkdir(exist_ok=False)
        declaration = copy.deepcopy(source)
        declaration['source']['bank_index'] = index
        declaration['experiment_id'] = f"{design['experiment']}_bank{index}"
        declaration['data_role'] = 'prospective DEV confirmation of frozen lookup; not final TEST'
        declaration['source_sha256'] = {design['source_declaration']: design['source_declaration_sha256']}
        atomic_json(bank / 'declaration.json', declaration)
        env, initial = load_environment(declaration['config'], declaration['source'], 'terminal_contract_v2')
        pose = np.asarray(env.physics.helpers.pose(initial.data))
        evidence = bank / 'path_evidence'
        evidence.mkdir()
        for route in declaration['routes']:
            path = build_path(route['builder'], pose, declaration['config'])
            file = evidence / (route['name'] + '_path.npz')
            np.savez_compressed(file, xy=np.asarray(path.xy), s=np.asarray(path.s), goal=np.asarray(path.goal))
            files.append(file)
        files.append(bank / 'declaration.json')
        groups = []
        for number, context in enumerate(design['contexts']):
            group = bank / f'capture_{number}'
            group.mkdir()
            route = next(r for r in declaration['routes'] if r['name'] == context['route'])
            plan = dict(project='STTW_CONTROL', source_declaration=str(bank / 'declaration.json'),
                        path_evidence_directory=str(evidence), route=context['route'], builder=route['builder'],
                        model=declaration['policies']['H75'], policy_name='H75',
                        speeds=[context['speed']], alphas=[context['alpha']], phase_fractions=[.5],
                        forces_N=[0.], declared_prefixes=1, declared_cases=1,
                        path_evidence_atol_m=1e-4,
                        maximum_prepare_lower_ticks=2000, maximum_scan_lower_ticks=0,
                        stop_rule='One authentic first crossing; unavailable states retained, no phase/state replacement')
            groups.append(str(group))
            prefix_plans.append((group / 'prepare_plan.json', plan))
        banks.append(dict(bank_index=index, directory=str(bank), capture_groups=groups,
                          initial_pose=pose.tolist(), prepared_physical_tick=int(initial.physical_tick),
                          prepared_raw=np.asarray(initial.raw).tolist()))
    files = list(dict.fromkeys(files))
    base_sources = {str(file): digest(file) for file in files}
    for file, plan in prefix_plans:
        atomic_json(file, dict(plan, source_sha256=base_sources))
    sources = {**base_sources, **{str(file): digest(file) for file, _ in prefix_plans}}
    archive = root / 'source_at_launch'
    archive.mkdir()
    index = []
    for number, (file, sha) in enumerate(sources.items()):
        target = archive / f'{number:03d}_{Path(file).name}'
        target.write_bytes(Path(file).read_bytes())
        index.append(dict(original=file, archive=str(target), sha256=sha))
    atomic_json(archive / 'index.json', index)
    atomic_json(root / 'plan.json', dict(design, banks=banks, source_sha256=sources,
                code_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE, text=True).strip(),
                preparation_physics_ticks=0, preparation_model_fits=0))
    atomic_json(root / 'status.json', dict(phase='prepared', stage='ready', declared_cases=30,
                completed_cases=0, unavailable_cases=0, prefix_lower_ticks=0, suffix_lower_ticks=0,
                active_lower_ticks=0, training_steps=0, model_fits=0))
    print(json.dumps(dict(prepared=str(root), source_files=len(sources), banks=banks)))


def execute(root):
    from sttw_control.preference_best import atomic_json
    from sttw_control.reference_confirmation import confirmation_cases, capture_result_action
    plan = json.loads((root / 'plan.json').read_text())
    start_file = root / 'execution_started.json'
    with start_file.open('x') as stream:
        json.dump(dict(pid=os.getpid(), started_unix=time.time()), stream)
    status = dict(phase='running', pid=os.getpid(), stage='checking', declared_cases=30,
                  completed_cases=0, unavailable_cases=0, completed_prefixes=0,
                  prefix_lower_ticks=0, suffix_lower_ticks=0, active_lower_ticks=0,
                  training_steps=0, model_fits=0)

    def update(**changes):
        status.update(changes, checked_unix=time.time())
        status['active_lower_ticks'] = status['prefix_lower_ticks'] + status['suffix_lower_ticks']
        atomic_json(root / 'status.json', status)

    def child(command, log):
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=BASE, stdout=stream, stderr=subprocess.STDOUT)
            update(child_pid=process.pid, child_log=str(log))
            return process.wait()

    update()
    try:
        for file, sha in plan['source_sha256'].items():
            if digest(file) != sha:
                raise ValueError('execution source changed: ' + file)
        bank_receipts = []
        for bank in plan['banks']:
            directory = Path(bank['directory'])
            prefixes = []
            missing = []
            for group_string in bank['capture_groups']:
                group = Path(group_string)
                update(stage='capturing', bank_index=bank['bank_index'], group=str(group))
                code = child([sys.executable, '-u', str(COLLECTOR), '--run', str(group)], group / 'execution.log')
                receipt = json.loads((group / 'prepare_status.json').read_text())
                status['prefix_lower_ticks'] += receipt['active_lower_ticks']
                update()
                action = capture_result_action(receipt, code)
                if action == 'UNAVAILABLE':
                    error = receipt.get('error', '')
                    missing.append(dict(group=str(group), error=error, role='UNAVAILABLE, not a physical negative',
                                        partial_raw_availability='partial_capture_states.pkl and prefix_partial NPZ retained'))
                    continue
                rows = json.loads((group / 'prefix_results.json').read_text())['cases']
                if len(rows) != 1 or receipt['completed_cases'] != 1:
                    raise ValueError('capture case count differs')
                prefixes.extend(rows)
                status['completed_prefixes'] += 1
                (group / 'scan_plan.json').rename(group / 'generated_scan_NOT_EXECUTED.json')
            cases, pairs = confirmation_cases(prefixes, plan['contexts'], plan['forces_N'])
            atomic_json(directory / 'pairing.json', dict(pairs=pairs, unavailable_captures=missing,
                        real_suffix_cases=len(cases), declared_real_suffix_cases=15,
                        alias_rule='lookup continue reuses the identical actual baseline case'))
            status['unavailable_cases'] += 15 - len(cases)
            evaluation = directory / 'evaluation'
            evaluation.mkdir(exist_ok=True)
            if (evaluation / 'manifest.json').exists():
                raise FileExistsError('immutable bank evaluation exists')
            if cases:
                sources = {**plan['source_sha256'], **{row['snapshot']: row['snapshot_sha256'] for row in prefixes}}
                eval_plan = dict(project='STTW_CONTROL', source_declaration=str(directory / 'declaration.json'),
                                 path_evidence_directory=str(directory / 'path_evidence'), cases=cases,
                                 source_sha256=sources, prefix_duration_s=.4, original_task_deadline_s=10.,
                                 pulse_duration_s=.1, maximum_new_lower_ticks=len(cases) * 2000,
                                 declared_cases=len(cases), role='frozen lookup confirmation; no reselection')
                eval_plan['path_evidence_atol_m'] = 1e-4
                atomic_json(evaluation / 'plan.json', eval_plan)
                update(stage='evaluating', bank_index=bank['bank_index'])
                code = child([sys.executable, '-u', 'learning/cli/probe_reference_prefix_return.py',
                              '--plan', str(evaluation / 'plan.json')], directory / 'evaluation.log')
                receipt = json.loads((evaluation / 'status.json').read_text())
                status['suffix_lower_ticks'] += receipt['active_lower_ticks']
                status['completed_cases'] += receipt['completed_cases']
                update()
                if code or receipt['phase'] != 'completed':
                    raise RuntimeError('suffix evaluation error: ' + str(evaluation))
            else:
                atomic_json(evaluation / 'status.json', dict(phase='completed', stage='all_captures_unavailable',
                            completed_cases=0, declared_cases=0, unavailable_case_slots=15,
                            active_lower_ticks=0, training_steps=0))
            bank_receipts.append(dict(bank_index=bank['bank_index'], directory=str(directory),
                                      prefixes=len(prefixes), executed_cases=len(cases), unavailable_cases=15 - len(cases)))
            atomic_json(root / 'bank_receipts.json', bank_receipts)
        if status['active_lower_ticks'] > plan['maximum_total_new_lower_ticks']:
            raise ValueError('confirmation total budget exceeded')
        if status['completed_cases'] + status['unavailable_cases'] != 30:
            raise ValueError('declared suffix roster not reconciled')
        update(stage='analyzing')
        code = child([sys.executable, '-u', str(AUDIT / 'report_reference_confirmation.py'),
                      '--run', str(root)], root / 'analysis.log')
        if code:
            raise RuntimeError('confirmation report failed; preserve completed raw physics')
        update(phase='completed', stage='all_declared_states_accounted', report=str(root / 'analysis/REPORT.md'))
    except BaseException as exc:
        update(phase='error', error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    (execute if args.execute else prepare)(args.run.resolve())


if __name__ == '__main__':
    main()
