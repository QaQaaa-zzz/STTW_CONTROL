"""Seal35 new endpoint episodes plus five authenticated inherited H75 traces."""
import argparse
from pathlib import Path
from sttw_control.middle_lineage_evaluation import prepare_history

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('runs/anticipatory_reference_pilot_v2'))
    p.add_argument('--registry',type=Path,default=Path('learning/configs/frozen_middle_registry.json'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prepare_history(a.source,a.output,a.registry)
