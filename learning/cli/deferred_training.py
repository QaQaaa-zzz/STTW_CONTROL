#!/usr/bin/env python3
"""Wait for a declared external run, then launch one frozen STTW pipeline."""
import argparse
from pathlib import Path
from sttw_control.deferred_training import watch
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    watch(parser.parse_args().plan)
