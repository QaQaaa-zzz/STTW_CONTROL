"""Freeze S4 new-layout/new-physics comparisons, using only pre-obstacle inputs."""
import argparse,copy,csv,hashlib,itertools,json,subprocess
from pathlib import Path
import numpy as np
from sttw_control.route_pebbles import stone_layout

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);a=p.parse_args();r=a.root/'layout_test';r.mkdir(exist_ok=False);model_path=a.root/'S2/mechanism_forecast/mechanism_forecast.json';model=json.loads(model_path.read_text());cal=a.root/'calibration';height=a.runs_root/'h75_height_scan';fixed=[0,.25,.5,.75,1];tasks=model['task_thresholds'];base=json.loads((cal/'single10_v2.6_seed101.json').read_text());dev={f:json.loads((path/'analysis_summary.json').read_text())['rows'] for f,path in [('ridge',cal),('gravel',height)]};labels=[f'fixed{x:g}' for x in fixed]+['Zero']+[f'mechanism_{t}' for t in tasks]+[f'simple_{t}' for t in tasks]
old=[json.loads((cal/f'single10_v2.6_seed{s}.json').read_text())['randomization'] for s in [101,102,103]];keys=['friction_range','mass_scale','servo_gain_scale','steer_zero_offset_rad'];bounds={k:[min(c[k][0] for c in old),max(c[k][0] for c in old)] for k in keys};groups={}
for seed in [401,402,403,404]:
    rng=np.random.default_rng(seed);group=copy.deepcopy(old[0])
    for k in keys:
        v=float(rng.uniform(*bounds[k]));group[k]=[v,v]
    groups[seed]=group
scenes={};scene_jobs=[]
for layout in [0,1]:
    for family,heights in [('ridge',[.04,.06,.08,.1]),('gravel',[.005,.01,.02])]:
        for h in heights:
            name=f'{family}{round(h*1000):03d}_layout{layout}';terrain=r/'terrains'/name
            if family=='ridge':
                cfg=json.loads((cal/f'single{round(h*100):02d}_scene.json').read_text());cfg.update(center_s_m=[3.9,4.5][layout],role='S4 unseen ridge placement; original height/length/width/slope preserved')
            else:
                cfg=json.loads((height/'terrains'/f'height{round(h*1000):02d}mm'/'scene_config.json').read_text());cfg.pop('reference_layout_csv');cfg.update(seed=202611210+layout,name=name,role='S4 unseen stone arrangement; same density/radii/corridor/height')
                layout_path=r/f'gravel_layout{layout}.csv'
                if not layout_path.exists():
                    stones=stone_layout(cfg);stones[:,1]*=cfg['center_lateral_compression']
                    with layout_path.open('w') as stream:
                        w=csv.writer(stream);w.writerow(['id','s_m','n_m','radius_s_m','radius_n_m'])
                        for i,row in enumerate(stones):w.writerow([i,*row])
                cfg['reference_layout_csv']=str(layout_path)
            config=r/(name+'_scene.json');config.write_text(json.dumps(cfg,indent=2));scenes[(family,h,layout)]=terrain;scene_jobs.append(dict(family=family,config=str(config),output=str(terrain)))
jobs=[];decisions=[]
for context in model['contexts']:
    family=context['family'];h=context['height_m'];speed=context['speed_m_s'];simple={}
    for task,(path_tol,speed_tol) in tasks.items():
        candidates=[]
        for alpha in fixed:
            terrain_name=f'single{round(h*100):02d}' if family=='ridge' else f'height{round(h*1000):02d}mm';label=f'full_{alpha:g}' if family=='ridge' else f'fixed{alpha:g}';data=[x for x in dev[family] if x['terrain']==terrain_name and x['requested_speed_m_s']==speed and x['label']==label]
            assert len(data)==3
            cost=np.mean([10*int(x['failed'])+x['path_rmse_m']/path_tol+x['speed_rmse_m_s']/speed_tol for x in data]);candidates.append((-sum(x['tasks'][task] for x in data),float(cost),alpha))
        simple[task]=min(candidates)[2]
    choices={t:context['selected_alpha_tasks'][t]['alpha'] for t in tasks};decisions.append(dict(family=family,height_m=h,speed_m_s=speed,mechanism=choices,simple=simple))
    alphas=fixed+[0]+list(choices.values())+list(simple.values())
    for layout,seed in itertools.product([0,1],groups):
        name=f'{family}{round(h*1000):03d}_v{speed:g}_layout{layout}_seed{seed}';cfg=copy.deepcopy(base);cfg.pop('physics_contact_recording',None)
        cfg.update(num_envs=len(labels),trace_lanes=len(labels),alphas=alphas,zero_middle=[False]*5+[True]+[False]*6,case_labels=labels,zero_reference_channels=[[False,False]]*len(labels),user_speed_m_s=speed,seed=seed,randomization=groups[seed],omit_diagnostic_fields=['observation','policy_measurement'],role='S4 frozen selectors at common original start; no online switching or training',evaluation_role='New layout and new physics TEST; selection uses terrain family/height/request speed/task known before rollout')
        path=r/(name+'.json');path.write_text(json.dumps(cfg,indent=2));jobs.append(dict(name=name,config=str(path),terrain=str(scenes[family,h,layout]),physical_seed=seed,layout=layout,family=family,height_m=h,speed_m_s=speed))
spec=json.loads((a.root/'mechanism_scan/protocol.json').read_text());spec.update(experiment_id='H75_MECHANISM_LAYOUT_TEST',title='S4新布局与新物理组冻结选档检验',case_labels=labels,fixed_alphas=fixed,repeat_unit='No technical repeats; crossed two layouts and four physical groups',scenes=len(jobs),lanes=len(jobs)*len(labels),terrains=sorted({j['name'].split('_v')[0] for j in jobs}),batch_cases=len(jobs),max_physics_lane_steps=len(jobs)*len(labels)*50000,physical_groups=groups,physical_parameter_envelope=bounds,duplicate_pairs=[],prediction_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),data_role='TEST: two new terrain layouts and four new physical groups; no test tuning',stop_rule='176 cases / 2112 episodes or engineering error; no retries or training',estimated_wall_time_minutes=[35,90],comparison='Five fixed alpha, Zero, three task-conditioned mechanism choices, three archived simple DEV-ranking choices',selection_time='Before rollout at original common start; alpha fixed for episode',selection_inputs=['terrain family','terrain height','requested speed','task A/B/C'],fixed_context='same original route, initial state and task tolerances; no state-dependent online switching claim',layout_definition={'ridge_centers_s_m':[3.9,4.5],'old_ridge_center_s_m':4.2,'gravel_seeds':[202611210,202611211],'stone_density_per_m2':24,'lateral_compression':.25},analysis_unit='Report each layout and physical group; no treating 22 repeated contexts as independent physics draws')
spec['prediction_assessment']={'primary':'paired task success selected vs every fixed alpha and same-task simple rule; full-window and final-hold separately','grouping':'two layouts x four physical groups, per-cell and per-physical-group counts; gains and losses','freeze':'original mechanism model unchanged; simple rule uses only pre-S3 DEV101-103; no S3 outcomes to pick new alpha'}
for name,obj in [('jobs.json',jobs),('terrain_jobs.json',scene_jobs),('protocol.json',spec),('frozen_decisions.json',dict(model_sha256=spec['prediction_sha256'],decisions=decisions,simple_rule='Same archived algorithm: maximize DEV task successes, minimize failure-penalty plus normalized path/speed RMSE, then smaller alpha')),('status.json',dict(stage='prepared',planned=len(jobs),episodes=spec['lanes']))]:
    (r/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
(r/'README.md').write_text('# S4 新布局验证\n\n22工况×2新布局×4新物理组×12车道=2112条，10s/条。五固定alpha、Zero、三个任务的机制选档、三个任务的已有简单规则。固定原H75/Local300和共同起点，0训练。\n\n两类选择器在运行前确定alpha，回合内固定。选择只用地形类型/高度、请求速度、任务容差；不使用新布局结果，当前状态与路径几何保持共同配置。\n\n[frozen_decisions.json](frozen_decisions.json) · [protocol.json](protocol.json) · [实际XY（运行后）](RESULTS.md)\n')
q=a.root/'layout_test_queue';q.mkdir(exist_ok=False);(q/'plan.json').write_text(json.dumps(dict(stages=[dict(name='h75_mechanism_layout_test',path=str(r),contexts=len(jobs),episodes=spec['lanes'])],total_episodes=spec['lanes'],optimizer_updates=0),indent=2));print(len(jobs),spec['lanes'],len(scene_jobs))
