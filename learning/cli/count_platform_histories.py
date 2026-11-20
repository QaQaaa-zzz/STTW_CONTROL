"""Count exact physical histories across section assignments in each blind TEST group."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(path):
    digest = hashlib.sha256()
    with np.load(path) as trajectory:
        for key in ('qpos', 'qvel'):
            values = trajectory[key]
            digest.update(str((values.shape, values.dtype.str)).encode())
            digest.update(values.tobytes())
    return digest.hexdigest()


def count(cases, out):
    rows = pd.read_csv(cases)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows['history'] = list(pool.map(fingerprint, rows.trajectory))
    keys = ['skill', 'planner', 'layout', 'root', 'force', 'side']
    paths = rows.groupby(keys + ['history']).agg(
        contracts=('section', 'size'), wall=('collision', 'max'),
        admitted=('query_feasible', 'max'), all_admitted=('query_feasible', 'min'))
    paths['admitted_wall'] = paths.admitted & paths.wall
    paths['mixed_admission'] = paths.admitted != paths.all_admitted
    summary = paths.groupby(['skill', 'planner']).agg(
        contracts=('contracts', 'sum'), histories=('contracts', 'size'), walls=('wall', 'sum'),
        ever_admitted=('admitted', 'sum'), ever_admitted_walls=('admitted_wall', 'sum'),
        mixed_admission=('mixed_admission', 'sum'))
    summary.to_csv(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    count(args.cases, args.out)
