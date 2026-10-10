#!/usr/bin/env python3
"""Train a fresh lower residual policy with randomized v/steer commands."""
import os,sys,argparse
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.lower_command_training import run
p=argparse.ArgumentParser()
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
p.add_argument('--smoke-only',action='store_true');p.add_argument('--formal-only',action='store_true')
p.add_argument('--authorize-local-training',action='store_true')
p.add_argument('--resume-checkpoint');p.add_argument('--target-updates',type=int)
a=p.parse_args()
run(a.config,a.output,a.smoke_only,a.authorize_local_training,a.formal_only,a.resume_checkpoint,a.target_updates)
