"""Freeze P1 capability table before generating independent P4 terrain layouts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sttw_control.gravel_capability import preview_features,choose_preference
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();spec=json.loads(a.config.read_text());out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True);base=json.loads(Path(spec['base_sweep_config']).read_text());rows=json.loads((Path(spec['training_evaluation'])/'gravel_task_metrics.json').read_text());idx=json.loads((Path(spec['training_evaluation'])/'index.json').read_text());cells=[]
for terrain in [r['name'] for r in base['terrain_overrides']]:
    source=next(r for r in idx if r['terrain']==terrain);contract=json.loads((Path(source['run'])/'contract.json').read_text());path=next(iter(contract['settings']['terrains'].values()))
    for speed in spec['speeds_m_s']:
        features=preview_features(path,speed)
        for alpha,method in zip(spec['candidate_alphas'],['alpha0','alpha025','alpha05','alpha075','alpha1']):
            group=[r for r in rows if r['terrain']==terrain and r['speed_m_s']==speed and r['method']==method]
            for task,t in base['tasks'].items():
                cost=np.mean([(r['peak_path_error_m']/t['W_m'])**2+(r['speed_rmse_m_s']/t['speed_rmse_m_s'])**2+10*int(r['failed']) for r in group]);cells.append(dict(terrain=terrain,task=task,speed_m_s=speed,alpha=alpha,preview_features=features,qualified_fraction=np.mean([r['task_'+task] for r in group]),normalized_cost=cost,n=len(group)))
features=np.array([r['preview_features'] for r in cells]);scale=np.ptp(features,axis=0);scale=np.where(scale>0,scale,1)
model=dict(source_evaluation=str(Path(spec['training_evaluation']).resolve()),source_metrics_sha256=hashlib.sha256((Path(spec['training_evaluation'])/'gravel_task_metrics.json').read_bytes()).hexdigest(),alphas=spec['candidate_alphas'],neighbors=spec['neighbors'],distance_floor=spec['distance_floor'],feature_scale=scale.tolist(),feature_names=['cap_height_m','stone_density_m2','planned_centerline_episodes','median_planned_event_interval_s','planned_interval_cv','planned_centerline_coverage'],cells=cells,state_support=spec['state_support'],scope='Task-only averages same P1 cells over terrain;Task+Phase KNN uses ideal geometry preview. No TEST labels, actual contact or physics seed enters selection.')
(out/'capability_table.json').write_text(json.dumps(model,indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n');template=json.loads(Path('runs/h75_gravel_g0/assets/S2_5mm/scene_config.json').read_text());scenes=[]
for seed in spec['map_seeds']:
    for family in spec['terrain_families']:
        name=family+'_map'+str(seed);cfg=dict(template,name=name,seed=seed);cfg.pop('reference_layout_csv',None)
        if family=='S6':cfg.update(distribution='clustered',cluster_centers_sn_m=[[4.,0.],[6.,0.]],cluster_std_sn_m=[.3,.25],reference_layout_csv=str(out/'assets'/('S2_map'+str(seed))/'stone_centers.csv'))
        file=out/(name+'.json');file.write_text(json.dumps(cfg,indent=2)+'\n');scenes.append(dict(name=name,family=family,map_seed=seed,config=str(file),output=str(out/'assets'/name),parameters=cfg))
(out/'scenes.json').write_text(json.dumps(scenes,indent=2)+'\n');print(json.dumps(dict(capability_cells=len(cells),new_assets=len(scenes),no_test_outcomes_used=True)))
