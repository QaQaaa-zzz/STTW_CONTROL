"""Process-exit driven combined analysis; zero polling or extra physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--nominal-run',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    state=dict(phase='running',pid=os.getpid(),stage='waiting_for_dual_process_exit',
               declared_cases=48,completed_cases=0,new_physics_steps=0,training_steps=0)
    def update(**kw):
        state.update(kw,checked_unix=time.time())
        tmp=root/'analysis_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n')
        os.replace(tmp,root/'analysis_status.json')
    update()
    try:
        plan=json.loads((root/'analysis_plan.json').read_text())
        for file,sha in plan['source_sha256'].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:
                raise ValueError('analysis source changed: '+file)
        current=json.loads((root/'status.json').read_text())
        if current['phase']=='running':
            try:
                fd=os.pidfd_open(current['pid'])
            except ProcessLookupError:
                fd=None
            if fd is not None:
                try:select.select([fd],[],[])
                finally:os.close(fd)
            current=json.loads((root/'status.json').read_text())
        if current['phase']!='completed' or current['completed_cases']!=16:
            raise RuntimeError('dual physics did not complete; partial cases retained')
        update(stage='combined_analysis')
        command=['/home/qy/mujoco_playground/.venv/bin/python',plan['analyzer'],
                 '--run',str(args.nominal_run.resolve()),'--dual-run',str(root),
                 '--output',str(root/'combined_analysis')]
        with (root/'analysis.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((root/'combined_analysis/analysis.json').read_text())
        if len(result['cases'])!=48:raise ValueError('combined analysis missing cases')
        update(phase='completed',stage='combined_all_cases_ready',completed_cases=48,
               inherited_nominal_cases=32,new_dual_cases=16,
               analyzed_lower_ticks=result['active_lower_ticks'])
    except BaseException as exc:
        update(phase='error',error=repr(exc))
        raise


if __name__=='__main__':main()
