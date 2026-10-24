"""Test whether existing successful physical branches exceed the training roll stop."""
import json
from pathlib import Path

import numpy as np

from sttw_control.preference_best import atomic_json, digest


def main():
    receipts, rows = {}, []
    for run in ['alpha_physical_boundary_v1', 'alpha_physical_boundary_mirror_v1']:
        root = Path('runs')/run
        results = root/'results.json'
        receipts[str(results.resolve())] = digest(results)
        for row in json.loads(results.read_text())['cases']:
            if row['balance_label'] != 'NO_FLOOR_CONTACT':
                continue
            path = root/(row['name']+'.npz')
            receipts[str(path.resolve())] = digest(path)
            with np.load(path, allow_pickle=False) as z:
                mask = z['active_tick'].astype(bool)
                peak = z['peak_roll'][mask]
                crossed = np.flatnonzero(peak > .7)
                rows.append(dict(run=run, name=row['name'], peak_roll_rad=float(peak.max()),
                                 exceeds_training_roll_stop=bool(len(crossed)),
                                 first_crossing_s=float(z['time'][mask][crossed[0]]) if len(crossed) else None))
    source = Path('learning/src/sttw_control/teleop_env.py')
    if '(peak_roll>.7)' not in source.read_text():
        raise ValueError('Training roll termination source changed; audit assumption needs review')
    receipts[str(source.resolve())] = digest(source)
    atomic_json(Path('runs/training_roll_stop_reuse_audit_v1.json'), dict(
        hypothesis='Training stops at0.7rad and could omit physically successful recoveries in these panels',
        safe_branches=len(rows), safe_over_stop=sum(x['exceeds_training_roll_stop'] for x in rows),
        cases=rows, input_sha256=receipts, audit_source_sha256=digest(__file__),
        new_physics_ticks=0, new_optimizer_updates=0,
        conclusion='No support in these completed branches; do not launch new training on this explanation alone'))
    print(len(rows), 'safe branches;', sum(x['exceeds_training_roll_stop'] for x in rows), 'above training roll stop')


if __name__ == '__main__':
    main()
