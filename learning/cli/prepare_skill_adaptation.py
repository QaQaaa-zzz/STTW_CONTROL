"""Freeze symmetric source/RB adaptation configs and a bounded DEV panel; do not launch."""
import argparse,copy,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--cross-skill-manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads(a.cross_skill_manifest.read_text())
cases=manifest['cases'];base=json.loads(Path('learning/configs/isaac_rb_training150.json').read_text())
# Every hard case uses the actual current plant, not the historical training mesh.
source_cases=[c for c in cases if c['skill']=='source']
reference=json.loads(Path(source_cases[0]['config']).read_text())
plant=json.loads(Path(reference['plant_config']).read_text())
original=json.loads(Path('learning/configs/isaac_rb_rounded_jump.json').read_text())
original.update(plant_config=reference['plant_config'],physics_dt=.0025,skills=[],maximum_physics_lane_steps=3200)
configs=[];panels=[]
for skill in ('source','rb_run1_candidate'):
 cfg=copy.deepcopy(base)
 cfg.update(asset=plant['asset'],bundle=str(a.inputs/'bundles'/skill),updates=25,save_interval=5,evaluation_checkpoint_every=5,
  reachable_entry_bank=str(a.inputs/'reachable_entries/entries.npz'),reachable_entry_probability=.5,
  status='Prepared symmetric25 adaptation; not launched',
  optimizer_initialization='Own original actor/critic/normalizer; fresh Adam',
  reset_note='50% recorded reachable entry, 42% original nominal, 8% original airborne RSI')
 cfg['plant_options']=copy.deepcopy(plant['plant_options'])
 cfg['plant_options'].update(physics_dt=.0025,pendulum_locked=False,terrain_layout='replicated',env_spacing=20.,gpu_pair_capacities=base['plant_options']['gpu_pair_capacities'])
 cfgpath=a.output/(skill+'_training.json');cfgpath.write_text(json.dumps(cfg,indent=2)+'\n');configs.append(str(cfgpath))
 panel=[]
 for speed in (1.8,2.,2.2):
  with np.load(original['initial_state']) as data:state={k:data[k].copy() for k in data.files}
  state['root_velocity'][0]=speed
  statepath=a.output/f'nominal_v{speed}.npz'
  if not statepath.exists():np.savez_compressed(statepath,**state)
  spec=copy.deepcopy(original);name=f'nominal_v{speed}'
  spec.update(initial_state=str(statepath),skills=[skill],bundles=str(a.inputs/'bundles'),output=str(a.output/'update0'/skill/name),approach_speed=speed,
   role='Nominal-start capability retention DEV in Isaac; not original MuJoCo donor benchmark',optimizer_updates=0)
  path=a.output/f'{skill}_{name}.json';path.write_text(json.dumps(spec,indent=2)+'\n')
  panel.append(dict(case_id=name,config=str(path),analysis_kind='jump_only',seconds=8.,retention=True))
 for c in source_cases:
  spec=json.loads(Path(c['config']).read_text());candidate=spec['connection_candidate']
  if candidate['speed_m_s']!=2. or candidate['alpha'] not in (0.,1.):continue
  name=c['case_id'].removesuffix('_source')
  spec.update(skills=[skill],bundles=str(a.inputs/'bundles'),output=str(a.output/'update0'/skill/name),role='Frozen difficult full-chain DEV; seen nominal origin, not independent test')
  path=a.output/f'{skill}_{name}.json';path.write_text(json.dumps(spec,indent=2)+'\n')
  panel.append(dict(case_id=name,config=str(path),analysis_kind='continuous',seconds=22.,retention=False))
 panelpath=a.output/f'{skill}_panel.json'
 panelpath.write_text(json.dumps(dict(simulation_python=manifest['simulation_python'],analysis_python=manifest['analysis_python'],cases=panel),indent=2)+'\n')
 panels.append(str(panelpath))
plan=dict(status='prepared_pending_evaluation_orchestration_and_gpu_allocation',training_configs=configs,panels=panels,
 evaluated_updates=[0,5,10,15,20,25],training_transitions=2*25*4096*24,training_physics_lane_steps=2*25*4096*24*8,
 evaluation_rollouts=2*6*9,evaluation_max_physics_lane_steps=2*6*(3*8+6*22)/.0025,
 comparison='Each original policy versus its own adapted checkpoints. Source and RB identical budgets; update2 is separate engineering reference.',
 retention='Three fixed nominal starts at vx1.8/2.0/2.2. Full-window survival and strict support >=1s reported separately; no loss of any update0 successful nominal case.',
 selection='Among retention-qualified candidates maximize complete hard-chain successes; then minimize mean completion time on shared successful cases; ties keep earlier update. If no improvement retain original.',
 limits='Nine-case seen DEV panel only; no claim of preserving the original donor recovery distribution or independent generalization. Final adopted candidates still require wider paired full-chain evaluation. No automatic extension.')
(a.output/'panel_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(plan,indent=2))
