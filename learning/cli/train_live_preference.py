"""Matched ordinary PPO on the common live/reentry task contract."""
import argparse
from pathlib import Path
from sttw_control.live_ppo_run import prepare, execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','shape','execute'],required=True)
    a=p.parse_args()
    prepare(a.output) if a.phase=='prepare' else execute(a.output,shape_only=a.phase=='shape')
