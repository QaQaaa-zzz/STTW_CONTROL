"""Prepare, trace or execute a separately recorded anticipatory-reference stage."""
import argparse
from pathlib import Path
from sttw_control.anticipatory_run import prepare,execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','shape','execute'],required=True)
    p.add_argument('--stage',choices=['nominal','disturbed','endpoints','expert_relay','sequence_nominal','sequence_probe','timing','reentry'],default='nominal')
    p.add_argument('--inventory',type=Path,default=Path('docs/evidence/tolerance_progress_20261022/anticipatory_reference/preparation/case_inventory.json'))
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.inventory.resolve(),a.output.resolve())
    else:execute(a.output.resolve(),a.stage,shape_only=a.phase=='shape')
