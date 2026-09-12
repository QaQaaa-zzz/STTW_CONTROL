#!/usr/bin/env python3
"""Plot one PPO training run or compare all logged runs."""
import argparse
from pathlib import Path
from sttw_control.training_diagnostics import plot_training,compare_runs
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--training-run',type=Path)
    group.add_argument('--runs-root',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.runs_root:
        if a.output is None:p.error('--runs-root requires --output')
        print(compare_runs(a.runs_root,a.output))
    else:plot_training(a.training_run)
