"""Missing same-event zero-neural-middle arms; no training or gain selection."""
import argparse
from pathlib import Path

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--phase',choices=['prepare','shape','execute','analyze'],required=True)
    args=p.parse_args()
    if args.phase=='prepare':
        if args.source is None:p.error('--source is required')
        from sttw_control.zero_middle_feedback import prepare
        prepare(args.output,args.source)
    elif args.phase=='analyze':
        from sttw_control.finite_value_evaluation_report import analyze
        analyze(args.output)
    else:
        from sttw_control.finite_value_evaluation_run import execute
        execute(args.output,shape_only=args.phase=='shape')
