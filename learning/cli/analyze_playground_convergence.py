"""Report training-window trends and all frozen checkpoint task results."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--evaluation', type=Path)
p.add_argument('--window', type=int, default=25)
a = p.parse_args()
rows = [json.loads(line) for line in (a.run / 'training.jsonl').read_text().splitlines()]
windows = []
for start in range(0, len(rows), a.window):
    block = rows[start:start + a.window]
    means = {key: float(np.mean([r[key] for r in block])) for key in (
        'mean_step_reward', 'loss_value', 'kl_first_rollout_step', 'sample_s', 'optimize_s')}
    windows.append(dict(first_update=block[0]['update'], last_update=block[-1]['update'],
                        means=means, terminal_flags=np.sum([r['terminal_counts'] for r in block], axis=0).tolist(),
                        max_kl=max(r['kl_first_rollout_step'] for r in block)))
last = rows[-min(50, len(rows)):]
x = np.asarray([r['update'] for r in last], dtype=float)
y = np.asarray([r['mean_step_reward'] for r in last])
slope = float(np.polyfit(x, y, 1)[0]) if len(last) > 1 else None
result = dict(windows=windows, last_window_reward_slope_per_update=slope,
              interpretation='Descriptive training trends, not proof of convergence; training episodes and windows are correlated. KL is the logged first-rollout-step estimate, not full-batch KL.',
              checkpoint_results=[])
lines = ['# Training convergence and checkpoint comparison', '', result['interpretation'], '',
         '| Updates | Mean step reward | Value loss | KL mean / max | Success / failure / timeout flags | Done |',
         '|---|---:|---:|---:|---|---:|']
for w in windows:
    m=w['means'];v=w['terminal_flags']
    lines.append(f"|{w['first_update']}–{w['last_update']}|{m['mean_step_reward']:.6f}|{m['loss_value']:.5f}|{m['kl_first_rollout_step']:.5f} / {w['max_kl']:.5f}|{v[:3]}|{int(v[3])}|")
lines += ['', f'Last {len(last)} updates reward slope: {slope} per update. Terminal flags may overlap; done is the union, not their sum.', '',
          'The sampled training-reward best is not the fixed-route best. Reward is associated with the pre-update model. Do not compare rewards across reward configurations.']
if a.evaluation:
    status = json.loads((a.evaluation / 'status.json').read_text())
    declaration = json.loads((a.evaluation / 'declaration.json').read_text())
    lines += ['', '[All actual/reference XY](%s)' % (a.evaluation / 'xy.png'), '',
              '| Checkpoint label | Goal / evaluated | Per-route termination |', '|---|---:|---|']
    for skill in declaration['skills']:
        cases = [r for r in status['results'] if r['skill'] == skill]
        record = dict(skill=skill, goals=sum(r['goal'] for r in cases), evaluated=len(cases),
                      routes={str(r['route']): r['termination'] for r in cases})
        result['checkpoint_results'].append(record)
        lines.append(f"|{skill}|{record['goals']}/{len(cases)}|{record['routes']}|")
    lines += ['', 'Nominal DEV candidate selection only. Every candidate uses all declared routes; per-route winners cannot be combined into a single successful controller.']
(a.run / 'convergence.json').write_text(json.dumps(result, indent=2) + '\n')
(a.run / 'CONVERGENCE.md').write_text('\n'.join(lines) + '\n')
print(a.run / 'CONVERGENCE.md')
