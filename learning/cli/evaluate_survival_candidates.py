"""Prepare or run the fixed12-case paired candidate evaluation."""
import argparse
from pathlib import Path
from sttw_control.survival_candidate_evaluation import prepare,execute
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();(execute if args.execute else prepare)(args.run.resolve())
