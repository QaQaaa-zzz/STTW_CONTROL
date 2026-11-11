"""Freeze the bounded ridge height/speed calibration before fine-alpha validation."""
import argparse,itertools,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();r=a.output;r.mkdir(parents=True,exist_ok=False)
alpha=[0,.25,.5,.75,1,0];labels=['full_0','full_0.25','full_0.5','full_0.75','full_1','Zero']
source_spec=json.loads((a.source/'protocol.json').read_text())
scene=json.loads(Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_dense_corridor_ridge10/single_10cm/identity.json').read_text())
jobs=[]
for h in [.04,.06,.08,.10]:
 name=f'single{round(h*100):02d}';terrain=r/'terrains'/name
 cfg={k:scene[k] for k in ['route','bounds_m','horizontal_scale_m','vertical_scale_m','slope_threshold','center_s_m','length_m','half_width_m','oblique_s_per_lateral_m','switch_s_m']}
 cfg.update(height_m=h,role='S2 height-response calibration DEV; world-fixed ridge, shared geometry and route')
 scene_path=r/(name+'_scene.json');scene_path.write_text(json.dumps(cfg,indent=2))
 for speed,seed in itertools.product([1.8,2.1,2.3,2.6],[101,102,103]):
  c=json.loads((a.source/f'single10_v2.6_seed{seed}.json').read_text());case=f'{name}_v{speed:g}_seed{seed}'
  c.update(num_envs=6,trace_lanes=6,alphas=alpha,zero_middle=[False]*5+[True],case_labels=labels,user_speed_m_s=speed,zero_reference_channels=[[False,False]]*6,role='S2 calibration DEV, original frozen H75 and Local300; no training',evaluation_role='S2 calibration DEV; groups101/102 for fit,103 held out')
  c['plant_options'].update(track_wheel_contacts=True,track_contact_wrench=True,raw_contact_only=True)
  c['physics_contact_recording'].update(chunk_steps=100,record_window_s=[1.,4.],compress=True)
  path=r/(case+'.json');path.write_text(json.dumps(c,indent=2));jobs.append(dict(name=case,config=str(path),terrain=str(terrain),physical_seed=seed,speed_m_s=speed))
spec=dict(source_spec);spec.pop('heights_mm',None)
spec.update(experiment_id='H75_THEORY_HEIGHT_CALIBRATION',title='S2单脊高度—速度—恢复标定',case_labels=labels,terrains=['single04','single06','single08','single10'],speeds=[1.8,2.1,2.3,2.6],scenes=48,lanes=288,max_physics_lane_steps=14400000,duplicate_pairs=[],batch_cases=48,retain_physics_contacts=True,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),data_role='DEV calibration; physical groups101/102 fit,103 held-out assessment',primary_change='Four ridge heights and four speeds with frozen five alpha and paired Zero',stop_rule='One 288-episode bank or engineering error; no automatic repetition/training',retention='200Hz state/reward/control; 5kHz contact only1–4s, compressed100-step chunks; no videos',purpose='Identify height dependence absent from prior single10cm high-rate bank; calibrate before committing S3 forecasts',fit_protocol={'fit_physical_groups':[101,102],'assessment_physical_groups':[103],'future_test':'S3 fine alpha and physical groups separated from this DEV bank','labels':'complete run, roll limit, tracking domain, tasks A/B/C separately'},estimated_wall_time_minutes=[45,120])
(r/'jobs.json').write_text(json.dumps(jobs,indent=2));(r/'protocol.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2));(r/'status.json').write_text(json.dumps(dict(stage='prepared',planned=48,episodes=288)))
(r/'README.md').write_text('# S2高度—速度标定\n\n48工况×(五alpha+Zero)=288条；原模型共同起点；10s/条；0训练。4/6/8/10cm×1.8/2.1/2.3/2.6m/s×物理组101/102/103。拟合101/102、留出103。5kHz接触仅1–4s，其余保留200Hz状态与奖励。完整结果自动写RESULTS.md和ANALYSIS.md，Bark通知。\n')
queue=r.parent/'calibration_queue';queue.mkdir(exist_ok=False)
(queue/'plan.json').write_text(json.dumps(dict(stages=[dict(name='h75_theory_height_calibration',path=str(r),contexts=48,episodes=288)],total_episodes=288,optimizer_updates=0),indent=2))
print(r)
