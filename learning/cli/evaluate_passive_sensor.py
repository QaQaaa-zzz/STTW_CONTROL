"""Collect missing existing-sensor telemetry on60 fixed zero-neural cases."""
import argparse
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--phase',choices=['prepare','shape','execute','analyze'],required=True)
    args=p.parse_args()
    if args.phase=='prepare':
        if args.source is None:p.error('prepare requires original source')
        from sttw_control.passive_sensor_probe import prepare
        prepare(args.output,args.source)
    elif args.phase=='analyze':
        from sttw_control.finite_value_evaluation_report import analyze
        analyze(args.output)
    else:
        from sttw_control.finite_value_evaluation_run import execute
        execute(args.output,shape_only=args.phase=='shape')
