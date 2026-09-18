#!/usr/bin/env python3
"""Run the frozen rho=34.2 and rho=10 RSL training arms sequentially."""
import argparse
import json
from pathlib import Path

from sttw_control.training import TrainingConfig, train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-task',type=Path,required=True)
    parser.add_argument('--control-task',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    config=TrainingConfig(**json.loads(args.training.read_text()))
    arms=(('rho34p2',args.main_task),('rho10',args.control_task))
    completed=[]
    def status(phase,**extra):
        target=root/'status.json';temporary=root/'status.tmp'
        temporary.write_text(json.dumps({'phase':phase,'complete':phase=='complete',
            'completed_arms':completed,**extra},indent=2,ensure_ascii=False)+'\n')
        temporary.replace(target)
    try:
        for name,task in arms:
            status('training',active_arm=name)
            result=train(task,root/name/'training',config)
            if not result.get('complete'):raise RuntimeError(name+' training did not complete')
            completed.append({'name':name,'training':str(root/name/'training'),
                'control_transitions':result['control_transitions'],
                'last_checkpoint':result['last_checkpoint'],
                'best_reward_checkpoint':result.get('best_reward_checkpoint')})
        status('complete')
    except Exception as exc:
        status('error',error=repr(exc));raise


if __name__=='__main__':main()
