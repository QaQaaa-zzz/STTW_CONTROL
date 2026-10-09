#!/usr/bin/env python3
"""Thin V5.1 benchmark entry; never launches a formal training campaign."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.smooth_performance import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['env','rollout','ppo','evaluation','end-to-end'],required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--num-envs',type=int)
    p.add_argument('--rollout-steps',type=int,default=128)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--evaluation-update',type=int,choices=[25,100],default=100)
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    if args.smoke and args.num_envs is not None:p.error('smoke and dimensions are mutually exclusive')
    if args.warmups<2 or args.repeats<3:p.error('need >=2 warmups and >=3 measurements')
    run(args)
if __name__=='__main__':main()
