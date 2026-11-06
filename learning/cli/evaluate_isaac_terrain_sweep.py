"""Run the declared terrain/speed matrix through existing evaluator and plots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,default=Path('learning/configs/isaac_h75_terrain_sweep.json'))
p.add_argument('--assets',type=Path,default=Path('runs/h75_terrain_speed_sweep/assets.json'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--bundle',type=Path)
a=p.parse_args();spec=json.loads(a.config.read_text());assets=json.loads(a.assets.read_text())['terrains'];bundle=(a.bundle or Path(spec['reference_bundle'])).resolve()
base=json.loads(Path(spec['base_evaluation_config']).read_text());a.output.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,PYTHONPATH=str(Path('learning/src').resolve()),JAX_PLATFORMS='cpu')
lab='/home/qy/miniconda3/envs/env_isaaclab/bin/python';analysis='/home/qy/miniconda3/envs/mjp/bin/python'
status=dict(stage='evaluation',pid=os.getpid(),completed_conditions=0,physics_lane_steps=0,started=time.time(),bundle=str(bundle));index=[]
for speed in spec['user_speed_m_s']:
    for terrain in spec['terrain_overrides']:
        name=terrain['name'];case=a.output/(name+'_v'+str(speed).replace('.','p'))
        cached={'flat':'flat_mesh','oblique_4cm':'bump'}
        if speed==2.3 and name in cached and bundle==Path(spec['reference_bundle']).resolve():
            run=Path('runs/h75_selected_local300_bend')/cached[name];role='reused original matched run'
        else:
            case.mkdir();config=dict(base,user_speed_m_s=speed,evaluation_role=spec['role'],role=spec['initial_state_protocol']);config_path=case/'config.json';config_path.write_text(json.dumps(config,indent=2));run=case/'evaluation'
            with (case/'evaluation.log').open('w') as log:
                subprocess.run([lab,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',str(bundle),'--config',str(config_path),'--terrain',assets[name],'--output',str(run),'--seconds',str(spec['seconds'])],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status['physics_lane_steps']+=6*round(spec['seconds']/.0002)
            with (case/'analysis.log').open('w') as log:
                subprocess.run([analysis,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(run),'--rescore-zero','--title',f'H75 | {name} | requested {speed} m/s'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            role='new matched six-lane run'
        index.append(dict(terrain=name,user_speed_m_s=speed,run=str(run.resolve()),role=role))
        status['completed_conditions']+=1;(a.output/'status.json').write_text(json.dumps(status,indent=2));(a.output/'index.json').write_text(json.dumps(index,indent=2))
        lines=['# H75 terrain and requested-speed DEV','','All five alphas and matched Zero; same original path. Speed2.6 is a command change from the saved2.3 warm state. No training or generalization claim.','']
        for row in index:lines.extend([f"## {row['terrain']} / {row['user_speed_m_s']} m/s",'',f"{row['role']}",f"![XY]({row['run']}/xy_all_alpha.png)",f"[Per-step/cumulative rewards]({row['run']}/overview.png)",''])
        (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
status.update(stage='complete',finished=time.time(),optimizer_updates=0);(a.output/'status.json').write_text(json.dumps(status,indent=2))
