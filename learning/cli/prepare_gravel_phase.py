"""Prepare P2 constant-curvature route and shape-matched three-cluster layouts."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from sttw_control.geometric_path import build_path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();spec=json.loads(a.config.read_text());out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
(out/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
source=Path(spec['source_bundle']);bundle=out/'bundle';bundle.mkdir();c=json.loads((source/'config.json').read_text());c['path'].update({k:v for k,v in spec['route'].items() if k!='constant_curvature_length_m'});c['path']['turn_angle_rad']=(spec['route']['constant_curvature_length_m']+c['path']['curvature_transition_length_m'])/c['path']['turn_radius_m']
original=np.load(source/'route.npz');anchor=[*original['xy'][0],original['heading'][0]];route=build_path('left90_R2',anchor,c)
np.savez(bundle/'route.npz',**{k:np.asarray(getattr(route,k)) for k in ('s','xy','heading','curvature','turn_end','goal')});shutil.copyfile(source/'actor.npz',bundle/'actor.npz');(bundle/'config.json').write_text(json.dumps(c,indent=2)+'\n')
identity=json.loads((source/'identity.json').read_text());identity.update(route_name='P2_extended_constant_curvature_separate_protocol',world_anchor=[float(x) for x in anchor],source_bundle=str(source.resolve()),role=spec['role']);identity['files']={name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in ('actor.npz','config.json','route.npz')};(bundle/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
rng=np.random.default_rng(spec['seed']);count=spec['stones_per_cluster']*3;offsets=rng.uniform(-spec['within_cluster_s_half_width_m'],spec['within_cluster_s_half_width_m'],count);lateral=rng.uniform(-spec['within_cluster_n_half_width_m'],spec['within_cluster_n_half_width_m'],count);radii=rng.uniform(*spec['radius_range_m'],(count,2));groups=np.repeat(np.arange(3)-1,spec['stones_per_cluster'])
terrains={};scene_rows=[]
for gap in spec['spacing_m']:
 name='gap'+str(gap).replace('.','p');layout=out/(name+'_layout.csv');stones=np.column_stack((spec['cluster_center_s_m']+groups*gap+offsets,lateral,radii));np.savetxt(layout,np.column_stack((np.arange(count),stones)),delimiter=',',header='id,s_m,n_m,radius_s_m,radius_n_m',comments='')
 cfg={k:spec[k] for k in ('seed','height_m','s_range_m','half_width_m','blend_length_m','horizontal_scale_m','vertical_scale_m','bounds_m','radius_range_m')};cfg.update(name=name,route=str(bundle/'route.npz'),density_per_m2=count/((spec['s_range_m'][1]-spec['s_range_m'][0])*2*spec['half_width_m']),reference_layout_csv=str(layout),cluster_centers_s_m=(spec['cluster_center_s_m']+np.array([-gap,0,gap])).tolist(),spacing_m=gap)
 path=out/(name+'.json');path.write_text(json.dumps(cfg,indent=2)+'\n');terrains[name]=str(out/'assets'/name);scene_rows.append(dict(config=str(path),output=terrains[name],parameters=cfg))
(out/'scenes.json').write_text(json.dumps(scene_rows,indent=2)+'\n');(out/'assets.json').write_text(json.dumps(dict(terrains=terrains),indent=2)+'\n')
base=json.loads(Path(spec['base_sweep_config']).read_text());settings=json.loads(Path(base['base_evaluation_config']).read_text());settings.update(base['evaluation_overrides']);settings.update(num_envs=150,trace_lanes=150,episode_s=spec['seconds'],evaluation_terrains=terrains,terrains=terrains,zero_middle=[False]*150,randomization=dict(base['randomization_by_seed'][str(spec['seeds'][0])]),auto_reset_horizon=False,evaluation_role=spec['role'])
conditions=[];cases=[];fixed=[]
for seed in spec['seeds']:
 start=len(conditions)
 for speed in spec['speeds_m_s']:
  for i,(name,path) in enumerate(terrains.items()):
   case=dict(terrain=name,terrain_path=path,user_speed_m_s=speed,seed=seed,lane_count=5,output=str(out/'evaluation'/(name+'_v'+str(speed).replace('.','p')+'_seed'+str(seed))/'evaluation'))
   cases.append(case)
   conditions.extend(dict(terrain=name,terrain_id=i,speed_m_s=speed,parameter_seed=seed,alpha=alpha) for alpha in spec['alphas'])
 fixed.append(dict(config=base['randomization_by_seed'][str(seed)],env_ids=list(range(start,len(conditions)))))
settings['randomization']['latency_s']=[0,max(g['config']['latency_s'][1] for g in fixed)]
settings.update(lane_terrain_ids=[r['terrain_id'] for r in conditions],lane_speeds=[r['speed_m_s'] for r in conditions],lane_alphas=[r['alpha'] for r in conditions],alphas=[r['alpha'] for r in conditions],fixed_randomization_groups=fixed)
(out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n');(out/'conditions.json').write_text(json.dumps(conditions,indent=2)+'\n');batch=out/'batch';batch.mkdir();(batch/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
analysis=dict(base,role=spec['role'],reference_bundle=str(bundle),terrain_overrides=[dict(name=k) for k in terrains],seeds=spec['seeds'],user_speed_m_s=spec['speeds_m_s'],seconds=spec['seconds'],max_physics_lane_steps=spec['max_physics_lane_steps'],include_zero=False)
(out/'analysis_config.json').write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps(dict(lanes=len(conditions),cases=len(cases),turn_end=float(route.turn_end),goal=float(route.goal),actor_sha256=identity['files']['actor.npz'])))
