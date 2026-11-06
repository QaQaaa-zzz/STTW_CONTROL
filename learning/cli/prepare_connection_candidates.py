"""Prepare fixed A-to-left/right routes and a six-candidate nominal DEV contract."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.geometric_path import build_path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--bundle',type=Path,default=Path('runs/h75_selected_local300_flat/bundle'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads((a.bundle/'config.json').read_text())
original=np.load(a.bundle/'route.npz');pose=np.r_[original['xy'][0],original['heading'][0]]
cfg['path'].update(goal_section_offset_after_turn_m=5.,straight_prefix_m=1.)
source=build_path('straight',pose,cfg)
def save(name,route):
    path=a.output/(name+'.npz')
    np.savez(path,**{key:np.asarray(getattr(route,key)) for key in ('s','xy','heading','curvature','turn_end','goal')})
    return str(path)
source_path=save('source',source);exit_s=3.1
entry=pose.copy();entry[:2]+=exit_s*np.array([np.cos(pose[2]),np.sin(pose[2])])
tasks=[dict(task_id=name,route=save(name,build_path(name,entry,cfg)),seconds=10.) for name in ('left90_R2','right90_R2')]
terrain=dict(route=source_path,bounds_m=[-30,35],horizontal_scale_m=.1,vertical_scale_m=.005,slope_threshold=1.5,height_m=.03,center_s_m=2.,length_m=.8,oblique_s_per_lateral_m=.4,half_width_m=1.5,role='fixed oblique bump before shared exit; paired DEV')
(a.output/'terrain_scene.json').write_text(json.dumps(terrain,indent=2)+'\n')
settings=json.loads(Path('learning/configs/isaaclab_h75_offroad_selected.json').read_text());settings['plant_options']['env_spacing']=80.
(a.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
spec=dict(settings=str(a.output/'settings.json'),source_bundle=str(a.bundle),source_route=source_path,terrain=str(a.output/'terrain'),source_alpha=.5,source_speed_m_s=2.3,source_seconds=6.,source_exit_progress_m=exit_s,seed=28,origin_id=a.output.name, tasks=tasks,candidates=[dict(candidate_id=f'v{v}_a{alpha}',speed_m_s=v,alpha=alpha) for v in (1.5,2.3) for alpha in (0.,.5,1.)],replanning=dict(period_s=.5,departure_m=.3,arrival_radius_m=2.,radius_m=1.8,enabled=[True]*6,goal_after_turn_m=5.),corridor_half_width_m=2.,role='paired nominal DEV; six candidates from accessible closed-loop snapshot, hidden PhysX cache excluded',completion_rule='goal reached and survive entire 10-second window and remain within declared original-path corridor',maximum_physics_lane_steps=780000,optimizer_updates=0)
(a.output/'collection.json').write_text(json.dumps(spec,indent=2)+'\n')
print(a.output/'collection.json')
