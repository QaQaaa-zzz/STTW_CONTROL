"""Pair every task across two physical latency budgets and bootstrap the original root clusters."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(original, repeat):
    keys = ['skill', 'planner', 'layout', 'root', 'section', 'side', 'force']
    def metrics(path):
        x = pd.read_csv(path / 'online_cases.csv').set_index(keys)
        approved = x.query_feasible & (x.query_batch_ms > 0)
        x['approved'] = approved
        x['approved_wall'] = approved & x.collision
        x['approved_done'] = approved & x.success
        return x[['success', 'collision', 'approved', 'approved_wall', 'approved_done']].astype(int).sort_index()
    old, new = metrics(original), metrics(repeat)
    if not old.index.is_unique or not old.index.equals(new.index):
        raise ValueError('Assignment keys differ between clocks')
    plan = json.loads((repeat / 'test_freeze_manifest.json').read_text())
    rng = np.random.default_rng(plan['root_bootstrap_seed'])
    draws = rng.integers(len(plan['roots']), size=(10000, len(plan['roots'])))
    rows = []
    for (skill, planner), base in old.groupby(level=['skill', 'planner']):
        trial = new.loc[base.index]
        for name in ['success', 'collision', 'approved_wall', 'approved_done']:
            rates = []
            counts = []
            for x in (base, trial):
                den = x.approved if name.startswith('approved_') else pd.Series(1, index=x.index)
                a = pd.concat([x[name], den], axis=1).groupby(level='root').sum().reindex(plan['roots'])
                n, d = a.to_numpy().T
                counts.extend([int(n.sum()), int(d.sum())])
                total = d[draws].sum(1)
                rates.append(np.divide(n[draws].sum(1), total, out=np.full(len(total), np.nan), where=total>0))
            if not counts[1] or not counts[3]:
                continue
            low, high = 100 * np.nanquantile(rates[1] - rates[0], [.025, .975])
            rows.append([skill, planner, name, *counts, 100*(counts[2]/counts[3]-counts[0]/counts[1]), low, high])
    pd.DataFrame(rows, columns=['skill','planner','metric','old_n','old_d','new_n','new_d','delta_pp','low_pp','high_pp']).to_csv(repeat/'paired_clock_effect.csv', index=False)
    cols = ['layout','skill','root','force','section','side']
    pairs = [pd.read_csv(p/'matched_cases.csv').set_index(cols).sort_index() for p in (original, repeat)]
    if not pairs[0].index.equals(pairs[1].index):
        raise ValueError('C/C-R paired keys differ between clocks')
    net = pd.concat([x[['rescued','lost']].groupby(level='skill').sum() for x in pairs],axis=1,keys=['original','repeat'])
    net.columns = ['_'.join(column) for column in net.columns]
    conditions = pd.concat([old.add_prefix('old_'), new.add_prefix('new_')], axis=1)
    cr = conditions.xs('C-R',level='planner').groupby(level=['skill','layout','force'])
    per_condition = cr.sum()
    per_condition.insert(0, 'assignments', cr.size())
    per_condition.to_csv(repeat/'clock_conditions.csv')
    net.to_csv(repeat/'clock_rescues.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original', type=Path)
    p.add_argument('repeat', type=Path)
    args = p.parse_args()
    compare(args.original, args.repeat)
