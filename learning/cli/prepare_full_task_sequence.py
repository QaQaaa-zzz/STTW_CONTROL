"""Seal74 nominal pure-reference trajectories and10 inherited strong baselines."""
import argparse
from pathlib import Path
from sttw_control.full_task_sequence_protocol import prepare_full_sequence

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('runs/anticipatory_reference_pilot_v2'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare_full_sequence(a.source,a.output)
