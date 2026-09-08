#!/usr/bin/env python3
"""Export MP4 and state plots from an existing run without simulating again."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
from sttw_control.media import render_run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--fps',type=int,default=30)
    args=p.parse_args()
    result=render_run(args.run,fps=args.fps)
    print(json.dumps({k:v for k,v in result.items() if k!='frame_indices'},indent=2))
