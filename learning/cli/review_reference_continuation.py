"""Inspect completed supported-Actor trajectories without simulation or fitting."""
import argparse
import json
from sttw_control.reference_continuation_diagnostics import review

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation')
    print(json.dumps(review(parser.parse_args().evaluation), indent=2))
