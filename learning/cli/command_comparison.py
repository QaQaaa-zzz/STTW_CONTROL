#!/usr/bin/env python3
"""Overlay saved methods from a JSON manifest with a 'runs' label-to-path mapping."""
import argparse
import json
from sttw_control.command_comparison import generate_comparison

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(generate_comparison(json.load(open(a.manifest))['runs'],a.output))
