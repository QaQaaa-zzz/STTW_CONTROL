#!/usr/bin/env python3
import argparse
from pathlib import Path
from sttw_control.stability_screen import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run at most two frozen PPO stability checks')
    p.add_argument('--plan',type=Path,required=True)
    run(p.parse_args().plan)
