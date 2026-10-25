"""Wait on the original training process, then execute the finite declared evaluation once."""
import argparse
from pathlib import Path
from sttw_control.survival_endpoint_sequence import execute_sequence
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();execute_sequence(a.training,a.evaluation,a.output)
