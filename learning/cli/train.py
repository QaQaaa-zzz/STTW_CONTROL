#!/usr/bin/env python3
"""Train the declared MJX/PPO residual budget; never automatically extend it."""
import argparse
import json
from pathlib import Path
from sttw_control.training import train,TrainingConfig,train_experts
from dataclasses import replace

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initialize-actor',type=Path,help='Actor-only warm start, checkpoint or training best')
    p.add_argument('--expert-modes',action='store_true',help='Three independent fixed-alpha training processes, declared total budget')
    p.add_argument('--resume-experts',action='store_true',help='Only skip completed verified stages; never auto-resume partial training')
    args=p.parse_args()
    c=TrainingConfig(**json.loads(args.config.read_text()))
    if args.initialize_actor:c=replace(c,initialize_actor=str(args.initialize_actor.resolve()))
    if args.resume_experts and not args.expert_modes:p.error('--resume-experts requires --expert-modes')
    result=(train_experts(args.task,args.output,c,skip_completed=args.resume_experts) if args.expert_modes
            else train(args.task,args.output,c))
    print(json.dumps(result,indent=2))
