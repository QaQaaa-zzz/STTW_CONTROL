#!/usr/bin/env python3
"""Measure one MJX batch size per fresh process; separate compilation and runtime."""
import argparse
import json
from pathlib import Path
from sttw_control.benchmark import benchmark

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=61)
    a = p.parse_args()
    print(json.dumps(benchmark(a.task, a.output, num_envs=a.num_envs,
                              steps=a.steps, repeats=a.repeats, seed=a.seed), indent=2))
