"""Wait on the original training process, then run the declared fixed300 once."""
import argparse
from pathlib import Path
from sttw_control.finite_value_storage_sequence import execute_sequence

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    execute_sequence(args.training,args.evaluation,args.output)
