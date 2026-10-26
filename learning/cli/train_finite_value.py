"""Matched finite-value and complete-return online comparison."""
import argparse
from pathlib import Path
from sttw_control.finite_value_run import prepare, execute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare','shape','execute'], required=True)
    args = parser.parse_args()
    prepare(args.output) if args.phase == 'prepare' else execute(args.output, shape_only=args.phase == 'shape')
