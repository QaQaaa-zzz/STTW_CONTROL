"""Run the declared terrain/speed matrix through existing evaluator and plots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import select
from itertools import product

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,default=Path('learning/configs/isaac_h75_terrain_sweep.json'))
p.add_argument('--assets',type=Path,default=Path('runs/h75_terrain_speed_sweep/assets.json'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--bundle',type=Path)
p.add_argument('--batch-cases',type=int,default=341,help='Cases per GPU scene, six lanes each; frozen per-seed parameters retained')
p.add_argument('--resume',action='store_true',help='Continue recorded unfinished matrix after an interrupted process')
p.add_argument('--after-pid',type=int,help='Wait for an existing process exit using a kernel pidfd, without polling')
a=p.parse_args();spec=json.loads(a.config.read_text());assets=json.loads(a.assets.read_text())['terrains'];bundle=(a.bundle or Path(spec['reference_bundle'])).resolve()
base=json.loads(Path(spec['base_evaluation_config']).read_text());a.output.mkdir(parents=True,exist_ok=a.resume)
previous_status=json.loads((a.output/'status.json').read_text()) if a.resume else {}
if a.after_pid:
    fd=os.pidfd_open(a.after_pid)
    (a.output/'status.json').write_text(json.dumps(dict(previous_status,stage='queued',waiting_for_pid=a.after_pid,pid=os.getpid())))
    select.select([fd],[],[]);os.close(fd)
env=dict(os.environ,PYTHONPATH=str(Path('learning/src').resolve()),JAX_PLATFORMS='cpu')
lab='/home/qy/miniconda3/envs/env_isaaclab/bin/python';analysis='/home/qy/miniconda3/envs/mjp/bin/python'
status=dict(stage='evaluation',pid=os.getpid(),completed_conditions=0,physics_lane_steps=0,started=time.time(),bundle=str(bundle));index=[]
if a.resume:
    status.update(previous_status);status.update(stage='evaluation',pid=os.getpid(),resumed=time.time())
    index=json.loads((a.output/'index.json').read_text()) if (a.output/'index.json').exists() else []
(a.output/'status.json').write_text(json.dumps(status,indent=2))
def record_case(name,speed,seed,run,role):
    index.append(dict(terrain=name,user_speed_m_s=speed,run=str(run.resolve()),role=role,seed=seed))
    status['completed_conditions']+=1
    (a.output/'status.json').write_text(json.dumps(status,indent=2));(a.output/'index.json').write_text(json.dumps(index,indent=2))
    lines=['# H75 terrain and requested-speed DEV','','All five alphas and matched Zero; same original path. No training or generalization claim.',spec['initial_state_protocol'],spec['role'],'']
    for row in index:lines.extend([f"## {row['terrain']} / {row['user_speed_m_s']} m/s",'',row['role'],f"![XY]({row['run']}/xy_all_alpha.png)",f"[Per-step/cumulative rewards]({row['run']}/overview.png)",''])
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if a.batch_cases>1:
    pending=[(seed,speed,terrain) for seed in spec.get('seeds',[None]) for speed,terrain in product(spec['user_speed_m_s'],spec['terrain_overrides']) if not any(row['terrain']==terrain['name'] and row['user_speed_m_s']==speed and row.get('seed')==seed for row in index)]
    for first in range(0,len(pending),a.batch_cases):
        batch=a.output/'batches'/f'case{len(index)}'
        if batch.exists():batch.rename(batch.with_name(batch.name+'_interrupted_'+str(int(time.time()))))
        config=dict(base,evaluation_role=spec['role'],role=spec['initial_state_protocol']);config.update(spec.get('evaluation_overrides',{}))
        cases=[]
        for seed,speed,terrain in pending[first:first+a.batch_cases]:
            name=terrain['name'];case=a.output/(name+'_v'+str(speed).replace('.','p')+('' if seed is None else '_seed'+str(seed)))
            run=case/'evaluation'
            if run.exists() and (run/'status.json').exists() and json.loads((run/'status.json').read_text())['stage']=='complete':
                with (case/'analysis.log').open('w') as log:subprocess.run([analysis,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(run),'--rescore-zero'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                record_case(name,speed,seed,run,'reused completed batch physics; plotting resumed')
                continue
            if case.exists():case.rename(case.with_name(case.name+'_interrupted_'+str(int(time.time()))))
            if seed is not None:config.update(seed=seed,randomization=spec['randomization_by_seed'][str(seed)])
            case.mkdir();(case/'config.json').write_text(json.dumps(dict(config,user_speed_m_s=speed),indent=2))
            cases.append(dict(terrain=name,terrain_path=assets[name],user_speed_m_s=speed,seed=seed,randomization=spec['randomization_by_seed'][str(seed)] if seed is not None else None,output=str(run)))
        if not cases:continue
        batch.mkdir(parents=True)
        (batch/'config.json').write_text(json.dumps(config,indent=2));(batch/'cases.json').write_text(json.dumps(cases,indent=2))
        with (batch/'evaluation.log').open('w') as log:
            subprocess.run([lab,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',str(bundle),'--config',str(batch/'config.json'),'--terrain',cases[0]['terrain_path'],'--case-manifest',str(batch/'cases.json'),'--output',str(batch/'evaluation'),'--seconds',str(spec['seconds'])],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for case in cases:
            run=Path(case['output']);status['physics_lane_steps']+=6*round(spec['seconds']/.0002)
            with (run.parent/'analysis.log').open('w') as log:subprocess.run([analysis,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(run),'--rescore-zero','--title',f"H75 | {case['terrain']} | requested {case['user_speed_m_s']} m/s"],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            record_case(case['terrain'],case['user_speed_m_s'],case['seed'],run,'batched six-lane case; frozen parameters per seed in shared GPU scene')

for seed in ([] if a.batch_cases>1 else spec.get('seeds',[None])):
    for speed,terrain in product(spec['user_speed_m_s'],spec['terrain_overrides']):
        name=terrain['name'];case=a.output/(name+'_v'+str(speed).replace('.','p')+('' if seed is None else '_seed'+str(seed)))
        if any(row['terrain']==name and row['user_speed_m_s']==speed and row.get('seed')==seed for row in index):continue
        if a.resume and case.exists():case.rename(case.with_name(case.name+'_interrupted_'+str(int(time.time()))))
        cached={'flat':'flat_mesh','oblique_4cm':'bump'}
        if seed is None and speed==2.3 and name in cached and bundle==Path(spec['reference_bundle']).resolve():
            run=Path('runs/h75_selected_local300_bend')/cached[name];role='reused original matched run'
        else:
            case.mkdir();config=dict(base,user_speed_m_s=speed,evaluation_role=spec['role'],role=spec['initial_state_protocol']);config.update(spec.get('evaluation_overrides',{}))
            if seed is not None:config.update(seed=seed,randomization=spec['randomization_by_seed'][str(seed)])
            config_path=case/'config.json';config_path.write_text(json.dumps(config,indent=2));run=case/'evaluation'
            with (case/'evaluation.log').open('w') as log:
                subprocess.run([lab,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',str(bundle),'--config',str(config_path),'--terrain',assets[name],'--output',str(run),'--seconds',str(spec['seconds'])],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status['physics_lane_steps']+=6*round(spec['seconds']/.0002)
            with (case/'analysis.log').open('w') as log:
                subprocess.run([analysis,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(run),'--rescore-zero','--title',f'H75 | {name} | requested {speed} m/s'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            role='new matched six-lane run'
        record_case(name,speed,seed,run,role)
status['physics_lane_steps']=sum(json.loads((Path(row['run'])/'status.json').read_text())['physics_steps'] for row in index)
status.update(stage='complete',finished=time.time(),optimizer_updates=0);(a.output/'status.json').write_text(json.dumps(status,indent=2))
if spec.get('seeds'):
    subprocess.run([analysis,'learning/cli/analyze_tradeoff_repeats.py','--evaluation',str(a.output)],env=env,check=True)
