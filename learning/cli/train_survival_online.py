"""Prepare or execute the fixed64-update matched online training stage."""
import argparse
from pathlib import Path
from sttw_control.survival_online_run import prepare,execute
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--recover-first-warmup',type=Path)
    args=p.parse_args()
    if args.execute:
        if args.recover_first_warmup is not None:p.error('recovery source belongs to preparation, not execution')
        execute(args.run.resolve())
    else:prepare(args.run.resolve(),recover_first_warmup=args.recover_first_warmup)
