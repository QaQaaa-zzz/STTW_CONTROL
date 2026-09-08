#!/usr/bin/env python3
"""Plot same-contract baseline/candidate recordings without rerunning physics."""
import argparse
from pathlib import Path
from sttw_control.media import compare_runs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-label',default='Learned residual')
    a=p.parse_args()
    compare_runs(a.baseline,a.candidate,a.output,candidate_label=a.candidate_label)
