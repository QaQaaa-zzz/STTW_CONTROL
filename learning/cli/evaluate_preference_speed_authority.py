"""One fixed physical speed-authority midpoint check; no parameter search."""
import argparse
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--phase',choices=['prepare','shape','execute','analyze'],required=True)
    a=p.parse_args()
    if a.phase=='prepare':
        if a.source is None:p.error('--source required')
        from sttw_control.preference_speed_authority import prepare_midpoint
        prepare_midpoint(a.output,a.source)
    elif a.phase=='analyze':
        from sttw_control.finite_value_evaluation_report import analyze
        analyze(a.output)
    else:
        from sttw_control.finite_value_evaluation_run import execute
        execute(a.output,shape_only=a.phase=='shape')
