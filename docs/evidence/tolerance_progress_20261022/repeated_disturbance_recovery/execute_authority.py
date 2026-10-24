"""Sequential finite A3 probe campaign; no retries, training, or budget extension."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from sttw_control.preference_best import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    root = Path(a.run).resolve()
    declaration = json.loads((root/'declaration.json').read_text())
    if digest(declaration['design']) != declaration['design_sha256']:
        raise ValueError('Declared design changed')
    state = dict(phase='running', pid=os.getpid(), stage='starting', completed_cases=0,
                 declared_cases=240, active_lower_ticks=0, training_steps=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(root/'status.json', state)
    update()
    try:
        for job in declaration['jobs']:
            folder = Path(job['plan']).parent
            if (folder/'manifest.json').exists():
                raise FileExistsError('Do not silently rerun existing probe '+str(folder))
            update(stage='scanning', job=job['name'])
            with (folder/'run.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', 'learning/cli/probe_balance_boundary.py',
                                          '--plan', job['plan']], stdout=log, stderr=subprocess.STDOUT)
                update(child_pid=child.pid)
                result = child.wait()
            if result:
                raise RuntimeError(f"Probe {job['name']} exited {result}; preserve original error")
            status = json.loads((folder/'status.json').read_text())
            if status['phase'] != 'completed' or status['completed_cases'] != job['cases']:
                raise ValueError('Probe case coverage incomplete')
            state['completed_cases'] += status['completed_cases']
            state['active_lower_ticks'] += status['active_lower_ticks']
            if state['active_lower_ticks'] > declaration['maximum_lower_ticks']:
                raise ValueError('Physics budget exceeded')
            update()
        update(phase='completed', stage='all_declared_probes_complete')
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
