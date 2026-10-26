"""Explicit authenticated retained-boundary recovery of the original64 run."""
import argparse
from pathlib import Path
from sttw_control.finite_value_storage_recovery import prepare,execute

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--boundary',type=int,default=45)
    parser.add_argument('--phase',choices=['prepare','shape','execute'],required=True)
    args=parser.parse_args()
    if args.phase=='prepare':
        if args.source is None:parser.error('--source required for explicit recovery')
        prepare(args.output,source=args.source,boundary=args.boundary)
    else:execute(args.output,shape_only=args.phase=='shape')
