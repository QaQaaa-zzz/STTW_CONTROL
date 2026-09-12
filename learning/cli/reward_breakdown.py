#!/usr/bin/env python3
"""Reconstruct reward components and plot all recorded alpha comparisons."""
import argparse
from pathlib import Path
from sttw_control.reward_breakdown import generate
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    print(generate(p.parse_args().run))
