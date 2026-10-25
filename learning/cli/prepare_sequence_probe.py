"""Seal16 new force branches for two fixed nominal-selected reference processes."""
import argparse
from pathlib import Path
from sttw_control.sequence_disturbance_protocol import prepare_sequence_probe

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('runs/full_task_sequence_nominal_v1'))
    p.add_argument('--baselines',type=Path,default=Path('runs/anticipatory_reference_pilot_v2'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare_sequence_probe(a.source,a.baselines,a.output)
