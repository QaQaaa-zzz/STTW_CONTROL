"""Summarize every predeclared training condition without selecting test roots."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
a = p.parse_args()
m = json.loads(a.manifest.read_text())
rows = []
for c in m['cases']:
    out = Path(json.loads(Path(c['config']).read_text())['output'])
    s = json.loads((out / 'chain_summary.json').read_text())
    rows.append(dict(origin_id=c['origin_id'], speed_m_s=c['speed_m_s'], alpha=c['alpha'],
                     complete=s['complete'], stages=s['stages'], checks=s['checks'],
                     decision_prefix=s['decision_prefix'], xy=str(out / 'xy_jump.png'),
                     trajectory=str(out / 'trajectory.npz'), summary=str(out / 'chain_summary.json')))
actions = sorted(set((r['speed_m_s'], r['alpha']) for r in rows))
fixed = [dict(speed_m_s=v, alpha=alpha,
              successes=sum(r['complete'] for r in rows if (r['speed_m_s'], r['alpha']) == (v, alpha)),
              denominator=len(m['roots'])) for v, alpha in actions]
oracle = sum(any(r['complete'] for r in rows if r['origin_id'] == root) for root in m['roots'])
result = dict(role='TRAIN descriptive action coverage, not held-out planner performance', rows=rows,
              fixed_actions=fixed, hindsight_per_root_successes=oracle, root_denominator=len(m['roots']),
              best_fixed_successes=max(x['successes'] for x in fixed), optimizer_updates=0)
(a.manifest.parent / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
lines = ['# A2 short120 training coverage', '',
         'Frozen update2; 5.6 m to platform front, 4 m nominal to jump activation. All upstream failures retained.',
         'These are training outcomes and hindsight coverage; no learned planner or generalization claim.', '',
         '| Root | Speed m/s | Alpha | Strict complete | XY |', '|---|---:|---:|---|---|']
for r in rows:
    xy = Path(r['xy']).relative_to(a.manifest.parent)
    lines.append(f"| {r['origin_id']} | {r['speed_m_s']} | {r['alpha']} | {r['complete']} | [XY]({xy}) |")
(a.manifest.parent / 'INDEX.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(best_fixed=result['best_fixed_successes'], hindsight=oracle, denominator=len(m['roots']))))
