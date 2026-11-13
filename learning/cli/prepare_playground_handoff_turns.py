"""Prepare approach-turn x departure-turn routes for continuous handoff testing."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
cfg = json.loads(a.source.read_text())
h = cfg['ground_handoff']
bank = np.load(h['reference_path'])
a.output.mkdir(parents=True, exist_ok=False)
s = bank['s']
xy_bank = []
cases = []
for approach, sign in [('straight', 0), ('left_return', 1), ('right_return', -1)]:
    for departure, old_id in [('straight', 60), ('left', 61), ('right', 62)]:
        xy = bank['xy'][old_id].copy()
        # Smooth excursion over upstream terrain, tangent and curvature zero at ends.
        t = np.clip((xy[:, 0] - 2.5) / (8.3 - 2.5), 0, 1)
        xy[:, 1] += sign * .55 * np.sin(np.pi * t) ** 4
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        xy = np.column_stack([np.interp(s, arc, xy[:, j]) for j in (0, 1)])
        xy_bank.append(xy)
        cases.append(dict(route_id=len(cases), approach=approach, departure=departure,
                          source_departure_route=old_id))
xy = np.asarray(xy_bank)
heading = np.unwrap(np.arctan2(np.gradient(xy[..., 1], s, axis=1),
                              np.gradient(xy[..., 0], s, axis=1)), axis=1)
curvature = np.gradient(heading, s, axis=1)
np.savez_compressed(a.output / 'routes.npz', s=s, xy=xy, heading=heading, curvature=curvature)
h['reference_path'] = str((a.output / 'routes.npz').resolve())
# First compare geometry at one nominal start; perturbation acceptance follows separately.
h['handoff_variants'] = [dict(h['handoff_variants'][0], start_y_m=0., start_yaw_rad=0., start_roll_rad=0.)]
(a.output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
protocol = dict(status='prepared_not_executed', role='nominal DEV handoff coverage',
                cases=cases, routes=list(range(9)), episodes_per_checkpoint=9,
                maximum_seconds_each=10, optimizer_updates=0,
                approach_curve=dict(x_interval_m=[2.5, 8.3], lateral_excursion_m=.55),
                flat_intervals_m=cfg['flat_intervals_m'], terrain_unchanged=True,
                acceptance='Original full-chain goal and hold; actual approach, RB, lock, departure; no reset or teleport.',
                checkpoint='Freeze one common candidate before execution; current source candidate is not adopted.',
                required_evidence='All nine actual/reference XY and per-case failures, entry and handback states; verify actual initial state.')
(a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
print(a.output / 'config.json')
