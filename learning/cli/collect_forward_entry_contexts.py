"""Collect only the sealed real forward entry prefixes; no policy training."""
import argparse
from pathlib import Path
from sttw_control.forward_entry_contexts import collect

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--shape-only',action='store_true')
    args=p.parse_args()
    collect(args.output,args.source,shape_only=args.shape_only)
