"""One preselected midpoint own-prefix check, using the existing physical runner."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sttw_control.preference_best import atomic_json, digest
from prepare_predictive_confirmation import register


def prepare(root):
    prior = Path('runs/formula_prospective_confirmation_v1').resolve()
    zero = Path('runs/new_geometry_zero_control_v1').resolve()
    original = json.loads((prior/'plan.json').read_text())
    for campaign in [prior, zero]:
        status = json.loads((campaign/'campaign_status.json').read_text())
        if status['phase'] != 'completed':
            raise ValueError('Comparator incomplete')
    for file, sha in original['source_sha256'].items():
        if digest(file) != sha:
            raise ValueError('Frozen runtime changed '+file)
    root.mkdir(parents=True, exist_ok=False)
    design = Path(__file__).with_name('MIDPOINT_PHYSICAL_DESIGN.md').resolve()
    sources = dict(original['source_sha256'])
    for file in [Path(__file__).resolve(), design, prior/'plan.json', zero/'plan.json']:
        sources[str(file)] = digest(file)
    for side in ['left', 'right']:
        child = root/side
        child.mkdir()
        plan = json.loads((prior/side/'prepare_plan.json').read_text())
        plan.update(alphas=[.5], policy_name='H75_midpoint', declared_prefixes=4,
                    declared_cases=12, maximum_prepare_lower_ticks=4000,
                    maximum_scan_lower_ticks=9600, maximum_total_lower_ticks=13600,
                    source_sha256=sources, design=str(design),
                    stop_rule='One fixed midpoint only; no alpha/force/phase extension or retry')
        plan.pop('pre_scan_forecast_script', None)
        atomic_json(child/'prepare_plan.json', plan)
        for name, count in [('prepare_status.json', 4), ('status.json', 12), ('campaign_status.json', 12)]:
            atomic_json(child/name, dict(phase='prepared', completed_cases=0,
                        declared_cases=count, active_lower_ticks=0, training_steps=0))
            register(child/('ACTIVE_'+name.replace('.json', '').upper()+'.json'),
                     'STTW_CONTROL 固定中间偏好 '+side+' '+name, child/name)
    atomic_json(root/'plan.json', dict(project='STTW_CONTROL', alphas=[.5],
                declared_cases=24, declared_prefixes=8, maximum_lower_ticks=27200,
                training_steps=0, source_sha256=sources,
                paired_H75_source=str(prior), paired_zero_source=str(zero),
                nominal_utility_source=str(Path('runs/nominal_preference_transfer_v1/utility/analysis.json').resolve()),
                code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                design=str(design)))
    atomic_json(root/'campaign_status.json', dict(phase='prepared', completed_cases=0,
                declared_cases=24, active_lower_ticks=0, training_steps=0))
    register(root/'ACTIVE_RUN.json', 'STTW_CONTROL 固定中间偏好24分支', root/'campaign_status.json')
    print('Prepared one midpoint,24 branches, maximum27200 lower ticks,0 training')


def execute(root):
    plan = json.loads((root/'plan.json').read_text())
    if json.loads((root/'campaign_status.json').read_text())['phase'] != 'prepared':
        raise ValueError('Not a fresh prepared campaign; do not replay completed stages')
    state = dict(phase='running', pid=os.getpid(), completed_cases=0,
                 declared_cases=24, active_lower_ticks=0, training_steps=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(root/'campaign_status.json', state)
    update(stage='authenticate')
    try:
        for file, sha in plan['source_sha256'].items():
            if digest(file) != sha:
                raise ValueError('Source changed '+file)
        for side in ['left', 'right']:
            child = root/side
            with (child/'controller.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u',
                    'docs/evidence/tolerance_progress_20261022/alpha_physical_boundary/execute.py',
                    '--run', str(child)], stdout=log, stderr=subprocess.STDOUT)
                update(stage=side, child_pid=process.pid)
                code = process.wait()
            result = json.loads((child/'campaign_status.json').read_text())
            if code or result['phase'] != 'completed':
                raise RuntimeError(side+' incomplete; preserve error and cost; no retry')
            state['completed_cases'] += result['completed_cases']
            state['active_lower_ticks'] += result['active_lower_ticks']
            if state['active_lower_ticks'] > plan['maximum_lower_ticks']:
                raise ValueError('Declared budget exceeded')
            update()
        update(phase='completed', stage='all24_midpoint_physical_outcomes_ready', child_pid=None)
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    (execute if args.execute else prepare)(args.run.resolve())
