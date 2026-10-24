"""Sequential authentic-prefix preparation and fixed physical grid, no retries."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from sttw_control.preference_best import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    root = Path(a.run).resolve()
    plan = json.loads((root/'prepare_plan.json').read_text())
    state = dict(phase='running', pid=os.getpid(), stage='preparing', completed_cases=0,
                 declared_cases=plan['declared_cases'], active_lower_ticks=0, training_steps=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(root/'campaign_status.json', state)
    update()
    try:
        for stage, script, arguments, result_name in [
            ('preparing', str(Path(__file__).with_name('collect_prefixes.py')), ['--run', str(root)], 'prepare_status.json'),
            ('scanning', 'learning/cli/probe_sensor_gate_domain_hold.py', ['--plan', str(root/'scan_plan.json')], 'status.json')]:
            update(stage=stage)
            with (root/(stage+'.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', script, *arguments], stdout=log, stderr=subprocess.STDOUT)
                update(child_pid=child.pid)
                code = child.wait()
            if code:
                raise RuntimeError(f'{stage} child exited{code}; preserve original error')
            result = json.loads((root/result_name).read_text())
            if result['phase'] != 'completed':
                raise ValueError(stage+' incomplete')
            state['active_lower_ticks'] += result['active_lower_ticks']
            if state['active_lower_ticks'] > plan['maximum_total_lower_ticks']:
                raise ValueError('Total physics budget exceeded')
            update()
        update(phase='completed', stage='all_declared_physical_outcomes_ready', completed_cases=plan['declared_cases'])
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
