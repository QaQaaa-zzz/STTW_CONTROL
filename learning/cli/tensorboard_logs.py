#!/usr/bin/env python3
"""Import declared PPO metrics into TensorBoard without rerunning training."""
import argparse
from pathlib import Path
from sttw_control.tensorboard_logging import import_metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,action='append',required=True,help='Repeat in lineage order')
    p.add_argument('--output',type=Path,required=True,help='New event directory; never overwrites')
    a=p.parse_args()
    print(f'Imported {import_metrics(a.metrics,a.output)} PPO iterations into {a.output}')
