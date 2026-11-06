import json
from pathlib import Path
import numpy as np
from sttw_control.goal_replanning import dubins_goal_path
root=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_forward_branch');root.mkdir(exist_ok=False)
base=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_jump_six_roots/rb_upstream_offset_01_L9.6_v2.0_a0.5.json');spec=json.loads(base.read_text());source=np.load(spec['ground_route'])
poses=[[-2.,0.,0.],[4.,-2.,0.],[12.6,-2.,0.],[16.6,0.,0.],[21.6,0.,0.]]
segments=[];metas=[];offset=0.
for start,end in zip(poses[:-1],poses[1:]):
 path,meta=dubins_goal_path(start,end[:2],radius_m=1.8,length_m=40.,spacing_m=.02,goal_heading_rad=end[2],numpy_output=True)
 use=path.s<=path.goal+1e-9;idx=np.flatnonzero(use)
 if segments:idx=idx[1:]
 segments.append({key:np.asarray(getattr(path,key))[idx] for key in ('s','xy','heading','curvature')});segments[-1]['s']+=offset;offset+=float(path.goal);metas.append(meta)
route={key:np.concatenate([s[key] for s in segments]) for key in segments[0]};route['turn_end']=sum(m['goal_arclength_m'] for m in metas[:-1]);route['goal']=offset
# Keep forward support after the shared goal for the existing pursuit controller.
extra=np.arange(.02,30.01,.02);route['s']=np.r_[route['s'],offset+extra];route['xy']=np.vstack([route['xy'],np.column_stack((poses[-1][0]+extra,np.zeros_like(extra)))]);route['heading']=np.r_[route['heading'],np.zeros(len(extra))];route['curvature']=np.r_[route['curvature'],np.zeros(len(extra))]
np.savez_compressed(root/'bypass_reference.npz',**route)
normal=np.column_stack((-np.sin(route['heading']),np.cos(route['heading'])));np.savez_compressed(root/'bypass_boundaries.npz',s=route['s'],center=route['xy'],left=route['xy']+.6*normal,right=route['xy']-.6*normal)
end=int(np.argmin(np.linalg.norm(source['xy']-poses[0][:2],axis=1)));prefix=source['xy'][:end];full=np.vstack([prefix,route['xy']]);np.savez_compressed(root/'bypass_original_map.npz',xy=full)
pre=np.all(np.diff(route['xy'][:,0])>=-1e-6);near=(route['xy'][:,0]>=7.6)&(route['xy'][:,0]<=11.6);clearance=float((abs(route['xy'][near,1])-1.).min());assert pre and clearance>=1.-1e-6
contract=dict(status='map_prepared_not_physically_executed',base_config=str(base),planner='Existing pinned PythonRobotics Dubins adapter; no new geometric planner',segments=metas,branch_reference=str(root/'bypass_reference.npz'),allowed_route_map=str(root/'bypass_original_map.npz'),common_goal_xy=poses[-1][:2],original_jump_goal_xy=spec['completion']['final_goal_xy'],goal_change='E5 comparison only: shared goal moved to 21.6m so bypass last turn has 5m straight. Existing J4 goal and labels unchanged.',branch_entry_pose=poses[0],first_entry_progress_m=float(source['s'][end])-.2,last_entry_progress_m=float(source['s'][end])+2.,minimum_dwell_s=.5,half_width_m=.6,monotone_world_x=bool(pre),centerline_min_platform_lateral_clearance_m=clearance,limits='Sampled map geometry only. No physical tracking, vehicle-envelope clearance or validated fallback claim.',proposed_pilot=dict(cases=['forced_original_jump','forced_bypass'],seconds=30.,physics_hz=400,lower_hz=200,upper_hz=50,physics_lane_steps_max=24000,optimizer_updates=0),task_semantics={'optional_platform':'same destination arrival may use either route','mandatory_platform_jump':'bypass only counts as safe exit, never completion of mandatory jump'},next='Connect same executor/map-entry filter, then execute two route pilots when a simulation slot is free')
(root/'contract.json').write_text(json.dumps(contract,indent=2)+'\n');print(json.dumps({k:contract[k] for k in ['monotone_world_x','centerline_min_platform_lateral_clearance_m','common_goal_xy','first_entry_progress_m','last_entry_progress_m']}))
