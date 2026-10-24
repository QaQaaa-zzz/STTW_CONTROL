"""Hindsight opportunity audit, never a deployable controller or TEST result."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'runs/selection_opportunity_audit_v1'
OUT.mkdir(exist_ok=True)
records, sources = [], []
for campaign in ('low_speed_reference_authority_v1', 'reference_authority_expansion_v1'):
    run = ROOT / 'runs' / campaign
    path = run / 'analysis.json'
    raw = path.read_bytes()
    data = json.loads(raw)
    sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    for sid in sorted({r['state_id'] for r in data['cases']}):
        modes = {}
        for mode in ('hold', 'pre_accel', 'react_accel'):
            rows = [r for r in data['cases'] if r['state_id'] == sid and r['mode'] == mode]
            assert len(rows) == 5
            assert all(r['balance_label'] in ('NO_FLOOR_CONTACT', 'FLOOR_CONTACT') for r in rows)
            zero = next(r for r in rows if r['force_N'] == 0)
            assert zero['balance_label'] == 'NO_FLOOR_CONTACT' and zero['full_observation']
            trace = run / (zero['name'] + '.npz')
            with np.load(trace) as d:
                active = d['active_tick'].astype(bool)
                reference = d['reference'][active]
                assert len(reference) == 1000  # fixed five-second zero-force window
                # Command intervention exposure, not task reward or actual tracking error.
                exposure = float(np.abs(reference[:, 0] - zero['speed_ref_m_s']).sum() * .005)
            sources.append(dict(path=str(trace), sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
            safe = sorted(r['force_N'] for r in rows if r['force_N'] != 0 and r['balance_label'] == 'NO_FLOOR_CONTACT')
            modes[mode] = dict(safe_forces=safe, safe_count=len(safe), speed_command_exposure_m=exposure)
        # Preserve every passing event of the best constant rule, not merely its count.
        target = set(modes['react_accel']['safe_forces'])
        feasible = [m for m in modes if target <= set(modes[m]['safe_forces'])]
        selected = min(feasible, key=lambda m: modes[m]['speed_command_exposure_m'])
        records.append(dict(campaign=campaign, state_id=sid, modes=modes,
                            hindsight_selected=selected,
                            scope='Selection uses outcome labels: opportunity bound only.'))
summary = {}
for mode in ('hold', 'pre_accel', 'react_accel', 'hindsight'):
    chosen = [r['modes'][r['hindsight_selected'] if mode == 'hindsight' else mode] for r in records]
    summary[mode] = dict(nonzero_safe=sum(x['safe_count'] for x in chosen),
                         nonzero_cases=4*len(records),
                         summed_nominal_command_exposure_m=sum(x['speed_command_exposure_m'] for x in chosen))
result = dict(records=records, summary=summary, sources=sources, new_physics_steps=0,
              optimization_steps=0, independent_training_repeats=0,
              warning='Post-hoc DEV opportunity audit; not learned selection, safety guarantee, or evidence of formula benefit.')
(OUT / 'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(summary=summary, selections=[(r['state_id'], r['hindsight_selected']) for r in records]), indent=2))
