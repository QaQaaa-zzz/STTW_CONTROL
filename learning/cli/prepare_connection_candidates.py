"""Prepare fixed A-to-left/right routes and a six-candidate nominal DEV contract."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.geometric_path import build_path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--bundle',type=Path,default=Path('runs/h75_selected_local300_flat/bundle'))
p.add_argument('--settings',type=Path,default=Path('learning/configs/isaaclab_h75_offroad_selected.json'))
p.add_argument('--height',type=float,default=.03)
p.add_argument('--oblique',type=float,default=.4)
p.add_argument('--source-speed',type=float,default=2.3)
p.add_argument('--connection-only',action='store_true')
p.add_argument('--connection-length',type=float,default=1.)
p.add_argument('--source-exit-progress',type=float,default=3.1)
p.add_argument('--corridor-half-width',type=float,default=2.)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads((a.bundle/'config.json').read_text())
original=np.load(a.bundle/'route.npz');pose=np.r_[original['xy'][0],original['heading'][0]]
cfg['path'].update(goal_section_offset_after_turn_m=5.,straight_prefix_m=a.connection_length)
source=build_path('straight',pose,cfg)
def save(name,route):
    path=a.output/(name+'.npz')
    np.savez(path,**{key:np.asarray(getattr(route,key)) for key in ('s','xy','heading','curvature','turn_end','goal')})
    return str(path)
source_path=save('source',source);exit_s=a.source_exit_progress
entry=pose.copy();entry[:2]+=exit_s*np.array([np.cos(pose[2]),np.sin(pose[2])])
tasks=[dict(task_id=name,route=save(name,build_path(name,entry,cfg)),seconds=10.) for name in ('left90_R2','right90_R2')]
terrain=dict(route=source_path,bounds_m=[-30,35],horizontal_scale_m=.1,vertical_scale_m=.005,slope_threshold=1.5,height_m=a.height,center_s_m=2.,length_m=.8,oblique_s_per_lateral_m=a.oblique,half_width_m=1.5,role='fixed oblique bump before shared exit; paired DEV')
(a.output/'terrain_scene.json').write_text(json.dumps(terrain,indent=2)+'\n')
settings=json.loads(a.settings.read_text());settings['plant_options']['env_spacing']=80.
settings['plant_options']['gpu_pair_capacities']=[2**22,2**22,2**24]
(a.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
spec=dict(settings=str(a.output/'settings.json'),source_bundle=str(a.bundle),source_route=source_path,terrain=str(a.output/'terrain'),source_alpha=.5,source_speed_m_s=a.source_speed,source_seconds=6.,source_exit_progress_m=exit_s,seed=28,origin_id=a.output.name, tasks=tasks,candidates=[dict(candidate_id=f'v{v}_a{alpha}',speed_m_s=v,alpha=alpha) for v in (1.5,2.3) for alpha in (0.,.5,1.)],replanning=dict(period_s=.5,departure_m=.3,arrival_radius_m=2.,radius_m=1.8,enabled=[True]*6,goal_after_turn_m=5.),corridor_half_width_m=a.corridor_half_width,role='paired nominal DEV; six candidates from accessible closed-loop snapshot, hidden PhysX cache excluded',completion_rule='goal reached and survive entire 10-second window and remain within declared original-path corridor',maximum_physics_lane_steps=round((6.+sum(t['seconds'] for t in tasks))*6/settings['plant_options'].get('physics_dt',.0002)),optimizer_updates=0)
if a.connection_only:
    spec.update(protocol_id='ground_connection_hold_v1',connection_end_progress_m=a.connection_length,common_task_reference=dict(speed_m_s=2.3,alpha=.5))
    spec['replanning']['arrival_reference']='original_route'
    for task in tasks:task['task_id']+=f'_connection_hold_v1_L{a.connection_length}_W{a.corridor_half_width}'
(a.output/'collection.json').write_text(json.dumps(spec,indent=2)+'\n')
print(a.output/'collection.json')
