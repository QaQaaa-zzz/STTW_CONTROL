"""Read-only diagnostics on the sealed 12 engineering episodes; no fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parent
receipt = json.loads((root/'execution/receipt.json').read_text())
for name, key in [('decisions.npz','raw_decisions_sha256'), ('physical.npz','raw_physical_sha256')]:
    assert hashlib.sha256((root/'execution'/name).read_bytes()).hexdigest() == receipt[key]
with np.load(root/'execution/decisions.npz', allow_pickle=False) as z:
    decisions = {k:z[k] for k in z.files}
with np.load(root/'execution/physical.npz', allow_pickle=False) as z:
    physical = {k:z[k] for k in z.files}
rows=[]
for i in range(12):
    valid=decisions['valid'][:,i]
    active=physical['active_tick'][:,i]
    upper_active=active.reshape(-1,4).any(1)
    # Count resumptions, not all subsequent legal decisions after first exit.
    starts=np.flatnonzero(valid & np.r_[False, ~valid[:-1]])
    before=physical['original_actor_latent'][::4,i][valid]
    after=physical['issued_latent'][::4,i][valid]
    assert np.isfinite(before).all() and np.isfinite(after).all()
    speed_changed=np.abs(before[:,0]-after[:,0])>1e-6
    steer_difference=np.abs(before[:,1]-after[:,1])
    fallback=active & physical['fallback_active'][:,i]
    all_reward=physical['scored_tick_reward'][:,i]
    rows.append(dict(case_index=i, legal_decisions=int(valid.sum()),
        active_upper_steps=int(upper_active.sum()),
        legal_decision_fraction=float(valid.sum()/upper_active.sum()),
        reentry_transitions=int(len(starts)), reentry_task_times_s=(starts*.02).tolist(),
        speed_latent_overwritten_decisions=int(speed_changed.sum()),
        speed_latent_overwritten_fraction=float(speed_changed.mean()),
        steering_latent_max_difference=float(steer_difference.max()),
        fallback_reward_sum=float(all_reward[fallback].sum(dtype=np.float64)),
        total_reward=float(all_reward[active].sum(dtype=np.float64))))
out=dict(scope='All 12 frozen stochastic engineering episodes, alpha=.5, two old routes; no efficacy estimate.',
    interpretation='Speed overwrite compares recorded original and issued latent at actual legal decisions; it is not a derivative or full closed-loop controllability test.',
    raw_decisions_sha256=receipt['raw_decisions_sha256'],raw_physical_sha256=receipt['raw_physical_sha256'],
    tolerance_for_recorded_latent_comparison=1e-6, cases=rows,
    total_reentry_transitions=sum(x['reentry_transitions'] for x in rows),
    physics_steps=0,optimizer_updates=0)
path=root/'analysis/control_authority_audit.json'
assert not path.exists(), 'preserve existing audit'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
