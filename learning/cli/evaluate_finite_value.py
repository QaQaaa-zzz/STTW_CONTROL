"""Fixed final64 six-role evaluation and read-only original-alpha analysis."""
import argparse
from pathlib import Path
from sttw_control.finite_value_evaluation_run import prepare,execute,prepare_reference_confirmation

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training',type=Path)
    parser.add_argument('--source-evaluation',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--phase',choices=['prepare','shape','execute','analyze'],required=True)
    args=parser.parse_args()
    if args.phase=='prepare':
        if args.protocol is not None:
            if args.source_evaluation is None:parser.error('--source-evaluation is required with --protocol')
            prepare_reference_confirmation(args.output,source=args.source_evaluation,protocol_file=args.protocol)
        else:
            if args.training is None:parser.error('--training is required for prepare')
            prepare(args.output,training=args.training)
    elif args.phase=='analyze':
        from sttw_control.finite_value_evaluation_report import analyze
        analyze(args.output)
    else:execute(args.output,shape_only=args.phase=='shape')
