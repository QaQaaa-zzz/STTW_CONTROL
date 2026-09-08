#!/usr/bin/env python3
"""Train the declared MJX/PPO residual budget; never automatically extend it."""
import argparse
import json
from pathlib import Path
from sttw_control.training import train,TrainingConfig

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(train(args.task,args.output,TrainingConfig(**json.loads(args.config.read_text()))),indent=2))
