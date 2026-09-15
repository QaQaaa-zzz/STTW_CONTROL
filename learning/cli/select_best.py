#!/usr/bin/env python3
"""Maintain fixed-development best_model aliases, optionally for a running stage."""
import argparse,json,time
from pathlib import Path
from sttw_control.selection import refresh_best_reward_model

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=float,default=15.)
    a=p.parse_args()
    if not a.interval>0:p.error('interval must be positive')
    previous=None
    while True:
        try:
            result=refresh_best_reward_model(a.training)
            if result and result['update']!=previous:
                print(json.dumps(result),flush=True);previous=result['update']
            status=a.training/'status.json'
            complete=status.exists() and json.loads(status.read_text()).get('complete',False)
        except (FileNotFoundError,json.JSONDecodeError):
            if not a.watch:raise
            complete=False  # writer may be replacing a validation file
        if not a.watch or complete:break
        pipeline=a.training.parent/'status.json'
        if pipeline.exists():
            state=json.loads(pipeline.read_text())
            if state.get('phase') in ('error','failed','cancelled'):
                raise RuntimeError('pipeline ended unsuccessfully: '+str(state))
        time.sleep(a.interval)
