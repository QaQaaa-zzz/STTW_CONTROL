"""Prepare only: no shape, environment construction, numerical run or training."""
import argparse
from pathlib import Path
from sttw_control.legacy_goal_preparation import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worktree',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=prepare(a.worktree,a.output)
    print('prepared',plan['unique_cases'],'unique cases; execution_ready=False; physics_steps=0')
