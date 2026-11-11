"""Prepare bounded nominal-ground calibration, independent of jump outcomes."""
import argparse, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); cfg=json.loads(a.source_config.read_text()); a.output.mkdir(parents=True,exist_ok=False)
for k in ('moving_event_planner','event_candidate_parameters','candidate_label'):
    cfg.pop(k,None)
n=24; cfg['num_envs']=n
spec=json.loads(Path(cfg['task_spec']).read_text()); spec['task']['ppo']['episode_horizon']=300
(a.output/'task.json').write_text(json.dumps(spec,indent=2)+'\n'); cfg['task_spec']=str(a.output/'task.json')
# The existing trench remains outside the declared six-second calibration area.
# All sampled ground from x=-5 to20 is flat; collision/contact interfaces unchanged.
cfg['ditch_terrain'].update(front_x_m=20.,stones_per_m2=0,stone_height_min_m=0.,stone_height_max_m=0.,ridges=[],rough_x_intervals_m=[[2.5,24.]])
cfg['policy_obstacle_reference_x_m']=20.3
h=cfg['ground_handoff']; h.pop('approach',None)
h.update(speed_m_s=1.5,alpha=.5,ground_only_lanes=[True]*n,reference_route_ids=[0]*n,observer_initialization_lanes=['cold']*n)
h['goal'].update(x_min_m=22.,x_max_m=24.)
s=np.arange(0,30.01,.02); xy=np.column_stack((s-5,np.zeros_like(s)))
np.savez(a.output/'reference.npz',s=s,xy=xy[None],heading=np.zeros_like(s)[None],curvature=np.zeros_like(s)[None])
h['reference_path']=str(a.output/'reference.npz'); cfg['task_reference_path']=h['reference_path']
rng=np.random.default_rng(2026111201)
cfg['ground_initial_conditions']=[dict(y_m=float(rng.uniform(-.02,.02)),yaw_rad=float(rng.uniform(-.01,.01)),roll_rad=float(rng.uniform(-.01,.01)),speed_m_s=float(rng.uniform(1.45,1.55))) for _ in range(n)]
cfg['data_role']='Nominal straight-ground calibration:12 CALIB and12 HOLDOUT conditions; no jump, no training; horizon timeout is not failure of the observation-window criterion'
cfg['report_directory']=str(a.output)
(a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
declaration=dict(question='Calibrate a dynamic continuation envelope without conditioning on successful jumps',source_config=str(a.source_config),calibration_lanes=list(range(12)),holdout_lanes=list(range(12,24)),budget=dict(environments=24,episode_s=6,physical_lane_steps_max=57600,training_updates=0,wall_limit_min=25),window_s=[3.5,5.5],scope='Straight nominal ground,1.5m/s,alpha0.5 only; no curved-road recovery certification',envelope_rule='For each absolute roll,pitch,yaw,world-y,vx-1.5 feature:1.2 times CALIB-window maximum, with floors[1deg,1deg,1deg,0.02m,0.1m/s]; report coverage of untouched HOLDOUT. Do not adopt if any CALIB terminates before5.5s, violates corridor, or bounds exceed original hard constraints.',capture_hold_s=.2,posthandoff_window_s=2.,capture_deadline_s=1.,minimum_forward_progress_m=1.,hard_limits=dict(road_half_width_m=.6,roll_deg=35,pitch_deg=75),stop='25min wall timeout or evaluator exception; no automatic extension',resources='24lane batched GPU evaluation; user authorized small parallel evaluations',interpretation='Empirical nominal envelope, not a viability set. Original task goal is deliberately outside run; goal success and calibrated continuation are separate.',contact_gate='Use recorded400Hz contact modes and original256contact buffer; known-ground contact must register. No new multimap configuration.',code_revision='b733aae9 plus prepare_continuation_calibration.py')
(a.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n'); print(a.output)
