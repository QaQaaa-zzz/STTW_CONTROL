from pathlib import Path
import json,numpy as np
r=Path(__file__).resolve().parents[1];cases=json.loads((r/'evaluation/summary.json').read_text())['cases'];out=[]
for c in cases:
 p=r/'evaluation'/c['candidate']['name'];d=np.load(p/'verified_diagnostics.npz');t=d['time'];stage=d['stage'][:,0];b=np.load(p/'physics_contact.npz');root=b['root_state_wxyz'][:,0]
 # Solved-step diagnostics may include the tail of the last20ms control interval.
 failed=np.flatnonzero(d['failed'][:,0]);end=int(failed[0]) if len(failed) else len(t)-1
 def first(mask):
  ids=np.flatnonzero(mask[:end+1]);return float(t[ids[0]]) if len(ids) else None
 row=dict(model=c['candidate']['name'],duration_s=c['duration_s'],terminal_code=c['terminal_code'],terminal_physics_s=float(t[end]),stage=c['verified_stage'],reward=c['reward_total'],success=c['verified_success'],peak_root_z=c['peak_root_z'],final_xyz=c['final_xyz'],stage_times_s={str(k):first(stage>=k) for k in range(1,5)},first_target_support_s={name:first(d['target'][:,0,i]) for i,name in enumerate(['front','rear'])},both_target_samples=int(d['target'][:end+1,0].all(1).sum()),max_patch_count=d['patch_count'][:end+1,0].max(0).tolist(),out_of_bounds_samples=d['patch_out_of_bounds'][:end+1,0].sum(0).tolist(),terminal_roll_deg=float(np.degrees(d['roll'][end,0])),terminal_pitch_deg=float(np.degrees(d['pitch'][end,0])),terminal_vx=float(root[end,7]),terminal_vz=float(root[end,9]))
 out.append(row)
(r/'analysis/replay_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
