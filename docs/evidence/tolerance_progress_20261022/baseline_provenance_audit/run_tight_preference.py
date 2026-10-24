"""Execute exactly the prepared nominal panels; no automatic extension."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    from sttw_control.preference_best import atomic_json
    root = args.run.resolve()
    plan = json.loads((root/'plan.json').read_text())
    state = dict(phase='running', pid=os.getpid(), completed_cases=0,
                 declared_cases=plan['total_cases'], active_lower_ticks=0,
                 maximum_lower_ticks=plan['maximum_lower_ticks'], training_steps=0)
    def update(**kwargs):
        state.update(kwargs, checked_unix=time.time())
        atomic_json(root/'status.json',state)
    update(stage='starting')
    try:
        for panel in plan['panels']:
            command = [sys.executable, '-u', 'learning/cli/screen_contact_preference.py',
                       '--declaration', panel['declaration'], '--output', panel['output']]
            with (root/(panel['name']+'.log')).open('x') as stream:
                child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                update(stage=panel['name'], child_pid=child.pid,
                       child_status=str(Path(panel['output'])/'status.json'))
                code = child.wait()
            if code:
                raise RuntimeError(f"{panel['name']} exited {code}; preserve partial output")
            receipt = json.loads((Path(panel['output'])/'status.json').read_text())
            if receipt['phase']!='completed' or receipt['completed_cases']!=panel['cases']:
                raise ValueError('incomplete nominal receipt')
            state['completed_cases'] += receipt['completed_cases']
            state['active_lower_ticks'] += receipt['active_lower_ticks']
            if state['active_lower_ticks']>plan['maximum_lower_ticks']:
                raise ValueError('declared physics exceeded')
            update()
        update(phase='completed',stage='nominal_raw_ready',child_pid=None)
    except BaseException as exc:
        update(phase='error',error=repr(exc))
        raise


if __name__=='__main__':
    main()
