"""Prepare or execute one declared matched optimizer engineering round."""
import argparse
from pathlib import Path
from sttw_control.survival_update_run import prepare,execute

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    (execute if args.execute else prepare)(args.run.resolve())
