#!/usr/bin/env python3
"""Four explicit geometric scenarios x three alphas; four baseline runs, not 12."""
import argparse
from pathlib import Path
from sttw_control.tracking_diagnostics import review_checkpoint

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--selection',type=Path,help='selection.json from explicit fixed checkpoint comparison')
    p.add_argument('--compact',action='store_true',help='Transfer summary, events and all-step reduced traces; retain full evidence locally')
    a=p.parse_args()
    review_checkpoint(a.training_run,a.panel,a.output,checkpoint=a.checkpoint,selection=a.selection,compact=a.compact)
