"""Prepare first75 same-start S6 entry/recovery candidates, without training."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();spec=json.loads(a.config.read_text());out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True);(out/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n');base=json.loads(Path(spec['base_sweep_config']).read_text());cfg=json.loads(Path(base['base_evaluation_config']).read_text());cfg.update(base['evaluation_overrides']);cfg['plant_options']=dict(cfg['plant_options'],env_spacing=spec['env_spacing'])
conditions=[];groups=[];fixed=[];cases=[]
for seed in spec['seeds']:
 start=len(conditions)
 for entry in spec['entry_alphas']:
  lanes=list(range(len(conditions),len(conditions)+5));groups.append(dict(lanes=lanes,entry_alpha=entry,target_alphas=spec['recovery_alphas']))
  conditions.extend(dict(parameter_seed=seed,entry_alpha=entry,recovery_alpha=target,speed_m_s=spec['speeds_m_s'][0],terrain='S6') for target in spec['recovery_alphas'])
  cases.append(dict(terrain='S6',terrain_path=spec['terrain'],user_speed_m_s=spec['speeds_m_s'][0],seed=seed,lane_count=5,entry_alpha=entry,output=str(out/'evaluation'/('entry'+str(entry).replace('.','p')+'_seed'+str(seed))/'evaluation')))
 fixed.append(dict(config=base['randomization_by_seed'][str(seed)],env_ids=list(range(start,len(conditions)))))
n=len(conditions);cfg.update(num_envs=n,trace_lanes=n,episode_s=spec['seconds'],terrains={'S6':spec['terrain']},evaluation_terrains={'S6':spec['terrain']},lane_terrain_ids=[0]*n,lane_alphas=[x['entry_alpha'] for x in conditions],alphas=[x['entry_alpha'] for x in conditions],zero_middle=[False]*n,lane_speeds=[x['speed_m_s'] for x in conditions],fixed_randomization_groups=fixed,randomization=dict(base['randomization_by_seed'][str(spec['seeds'][0])]),auto_reset_horizon=False,evaluation_role=spec['role'],gravel_recovery=dict(groups=groups,terrain=spec['terrain'],ramp_s=spec['ramp_s'],confirmation_s=spec['confirmation_s'],rear_support_force_N=spec['rear_support_force_N'],wheel_bound_radius_m=spec['wheel_bound_radius_m']))
cfg['randomization']['latency_s']=[0,max(x['config']['latency_s'][1] for x in fixed)]
(out/'settings.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'conditions.json').write_text(json.dumps(conditions,indent=2)+'\n');batch=out/'batch';batch.mkdir();(batch/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');(out/'evaluation').mkdir();print(json.dumps(dict(lanes=n,groups=len(groups),physics_lane_steps=n*round(spec['seconds']/.0002))))
