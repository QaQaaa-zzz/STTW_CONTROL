"""Read-only goal arrival review of the two named completed DEV datasets."""
import argparse
from pathlib import Path

from sttw_control.goal_arrival_analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=.3)
    args = parser.parse_args()
    root = args.worktree
    grid = root / 'runs/preference_capability_grid_v1'
    latest = root / 'runs/finite_value_online_v1_recovery55_fixed64_evaluation'
    summary = analyze([
        ('force_grid', grid / 'analysis/all_cases.csv', grid / 'evaluation/plan.json'),
        ('fixed300', latest / 'analysis/results.csv', latest / 'plan.json')], args.output,
        radius_m=args.radius)
    for row in summary:
        if row['radius_m'] == args.radius:
            print(row)


if __name__ == '__main__':
    main()
