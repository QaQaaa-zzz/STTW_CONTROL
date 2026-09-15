#!/usr/bin/env python3
"""Audit alpha-conditioned geometric recovery, without training or padding failures."""
import argparse
from pathlib import Path
from sttw_control.tracking_evaluation import evaluate_tracking

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[47011])
    p.add_argument('--alphas',type=float,nargs='+',default=[0.,.25,.5,.75,1.])
    a=p.parse_args()
    evaluate_tracking(a.task,a.training,a.output,a.checkpoint,seeds=a.seeds,alphas=a.alphas)
