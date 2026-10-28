"""Evaluate frozen tolerance models on the declared task6 same-state DEV labels."""
import argparse
from pathlib import Path
from sttw_control.tolerance_late_evaluation import evaluate_late

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--q-export',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();evaluate_late(args.root,args.q_export,args.output)
