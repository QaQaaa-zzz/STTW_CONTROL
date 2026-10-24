"""Read actual lower-only snapshots, without physics or upper-history invention."""
from pathlib import Path
import hashlib
import json
import pickle
import jax.numpy as j
import numpy as np
from sttw_control.physical_continuation import lower_risk_observation

ROOT = Path(__file__).resolve().parents[4]
out = ROOT/'runs/physical_continuation_views_v1'
out.mkdir(exist_ok=True)
rows = []
for campaign,prefix in [('low_speed_reference_authority_v1','low_speed_balance_v1'),
                        ('reference_authority_expansion_v1','reference_authority_expansion_prefix_v1')]:
    plan = json.loads((ROOT/'runs'/campaign/'plan.json').read_text())
    prepared = {r['state_id']:r for r in json.loads((ROOT/'runs'/prefix/'prefix_results.json').read_text())['cases']}
    snapshots = {c['state_id']:p for c,p in zip(plan['cases'],plan['case_snapshots'])}
    for sid,path in snapshots.items():
        file = Path(path); raw = file.read_bytes(); state = pickle.loads(raw); r = prepared[sid]
        measurement = {k:j.asarray(r[k]) for k in ('wheel_proxy_m_s','steer_rad',
            'steer_rate_rad_s','roll_rad','roll_rate_rad_s')}
        pose = j.concatenate([state.physical.data.qpos[:2],j.atleast_1d(state.physical.yaw_unwrapped)])
        view = lower_risk_observation(state,measurement=measurement,pose=pose,
            true_speed=j.asarray(r['actual_speed_m_s']),
            held_reference=j.asarray(r['governed_reference']),remaining_s=j.float32(5.))
        assert bool(view['valid']) and not bool(view['upper_action_valid'])
        assert 'upper_history' not in view
        assert float(view['measurement']['roll_rad']) == float(j.asarray(r['roll_rad']))
        assert hashlib.sha256(file.read_bytes()).digest() == hashlib.sha256(raw).digest()
        rows.append(dict(state_id=sid,snapshot=path,sha256=hashlib.sha256(raw).hexdigest(),
            view_valid=bool(view['valid']), upper_action_valid=False,
            true_roll_rad=r['roll_rad'], stale_upper_roll_rad=float(state.history.frames[-1,0]),
            controller_disturbance=float(state.physical.controller.disturbance),
            actual_physical_tick=int(state.physical.physical_tick),task_tick=int(state.tick)))
assert len(rows) == 12
receipt = dict(rows=rows,valid_views=12,new_physics_steps=0,optimization_steps=0,
    scope='Actual pre-disturbance lower snapshots only. No continuation physics or Critic training executed.',
    module_sha256=hashlib.sha256((ROOT/'learning/src/sttw_control/physical_continuation.py').read_bytes()).hexdigest())
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(valid_views=len(rows),maximum_stale_upper_roll_error_rad=max(abs(r['true_roll_rad']-r['stale_upper_roll_rad']) for r in rows)),indent=2))
