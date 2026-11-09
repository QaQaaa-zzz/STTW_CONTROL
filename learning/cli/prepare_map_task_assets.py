"""Factor the existing platform/bypass assets into a fixed region/connection graph."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-config',type=Path,required=True);p.add_argument('--map-contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
spec=json.loads(a.base_config.read_text());contract=json.loads(a.map_contract.read_text())
original=Path(spec['ground_route']);bypass=Path(contract['branch_reference']);paths={'original':dict(np.load(original)),'bypass':dict(np.load(bypass))}
coords=dict(start=[spec['start_x'],spec['start_y']],junction=contract['branch_entry_pose'][:2],launch=[spec['handoff_x'],0.],platform_exit=[spec['platform_back_x'],0.],landing=contract['original_jump_goal_xy'],side_entry=contract['segments'][0]['goal_xy'],side_exit=contract['segments'][1]['goal_xy'],merge=contract['segments'][2]['goal_xy'],destination=contract['common_goal_xy'])
regions=[dict(region_id=name,center_xy_m=xy) for name,xy in coords.items()]
def edge(name,start,end,reference,skills,width,role):
 r=paths[reference];i=int(np.linalg.norm(r['xy']-coords[start],axis=1).argmin());j=int(np.linalg.norm(r['xy']-coords[end],axis=1).argmin())
 if j<=i:raise ValueError('Connection endpoints must advance along the existing reference')
 return dict(connection_id=name,from_region=start,to_region=end,reference_asset=reference,s_start_m=float(r['s'][i]),s_end_m=float(r['s'][j]),length_m=float(r['s'][j]-r['s'][i]),half_width_m=width,skills=skills,role=role,physical_validation='not_established_for_new_task_contract')
connections=[edge('upstream','start','junction','original',['ground'],1.,'ground'),edge('prepare_jump','junction','launch','original',['ground'],1.,'jump_preparation'),edge('cross_platform','launch','platform_exit','original',['jump_update2'],1.,'jump'),edge('land_return','platform_exit','landing','original',['land_return'],1.,'landing'),edge('merge_straight','landing','merge','original',['ground'],1.,'ground'),edge('diverge','junction','side_entry','bypass',['ground'],contract['half_width_m'],'ground'),edge('side_passage','side_entry','side_exit','bypass',['ground'],contract['half_width_m'],'ground'),edge('rejoin','side_exit','merge','bypass',['ground'],contract['half_width_m'],'ground'),edge('to_goal','merge','destination','original',['ground'],1.,'ground')]
assets={name:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in [('original',original),('bypass',bypass)]}
assets.update(scene_terrain=spec['scene_terrain'],plant_config=spec['plant_config'],map_contract=str(a.map_contract.resolve()))
map_id='platform_bypass_'+hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()[:12]
graph=dict(schema='map_task_graph_v1',map_id=map_id,assets=assets,regions=regions,connections=connections,skills=[dict(skill_id='ground',family='ground',backend='H75_ECBC_ESO',bundle=spec['ground_bundle'],zero_lower=spec['zero_lower']),dict(skill_id='jump_update2',family='jump',backend='frozen_jump',bundle=str(Path(spec['bundles'])/'update2')),dict(skill_id='land_return',family='landing',backend='existing_landing_handoff',settings=spec['landing_return'])],unsupported_skills=['RecoveryBridge adapter must bind exact checkpoint/interface/evidence before use'],physical_validation='Two topological routes exist; two constraint-satisfying physical routes NOT yet established',source_base_config=str(a.base_config.resolve()))
task=dict(task_id='same_map_destination',start_region='start',goal_region='destination',hard_constraints=dict(forbidden_connections=[],ordered_required_regions=[],required_skill_families=[],maximum_remaining_time_s=30.,completion_protocol_id='E5_shared_road_v1_original_strict'))
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'map.json').write_text(json.dumps(graph,indent=2)+'\n');(a.output/'task.json').write_text(json.dumps(task,indent=2)+'\n')
print(json.dumps(dict(map_id=map_id,regions=len(regions),connections=len(connections),new_physics=False)))
