#!/usr/bin/env python3
"""One bounded training run, explicit priority panels and complete event media."""
import argparse
from pathlib import Path
from sttw_control.pipeline import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('task','training','panel','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Resume evaluation/media after completed training using identical frozen inputs')
    a=p.parse_args()
    run(a.task,a.training,a.panel,a.output,resume=a.resume)
