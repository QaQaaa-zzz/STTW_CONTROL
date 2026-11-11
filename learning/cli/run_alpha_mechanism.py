"""Execute the finite declared same-entry channel matrix using existing tools."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();r=a.run;spec=json.loads((r/'protocol.json').read_text());jobs=json.loads((r/'jobs.json').read_text());lab='/home/qy/miniconda3/envs/env_isaaclab/bin/python';cpu='/home/qy/miniconda3/envs/mjp/bin/python';lines=['# '+spec.get('title','同入障状态通道消融'),'','原H75+Local300；'+spec.get('data_role','DEV机制实验')+'。完整统计见ANALYSIS.md。',''];(r/'RESULTS.md').write_text('\n'.join(lines));status={'stage':'running','completed':0,'planned':len(jobs)}
if spec.get('batch_cases'):
 from concurrent.futures import ThreadPoolExecutor
 from itertools import groupby
 import shutil
 def plot_job(job):
  out=r/job['name']
  contract=json.loads((out/'contract.json').read_text());contract['settings']['compact_results']=True;(out/'contract.json').write_text(json.dumps(contract,indent=2))
  with (r/(job['name']+'_plot.log')).open('w') as log:
   subprocess.run([cpu,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(out),'--lane-labels',json.dumps(spec['case_labels']),'--title',job['name']],stdout=log,stderr=subprocess.STDOUT,check=True)
  contacts=out/'physics_contacts'
  if spec.get('compact_results') and not spec.get('retain_physics_contacts') and contacts.exists():
   index=contacts/'index.json'
   if index.exists():shutil.copy2(index,out/'omitted_contact_record_index.json')
   shutil.rmtree(contacts)
 futures=[]
 with ThreadPoolExecutor(max_workers=1) as plots:
  pending=[]
  for job in jobs:
   out=r/job['name']
   if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage']=='complete':
    futures.append(plots.submit(plot_job,job));status['completed']+=1
   else:pending.append(job)
  batch_index=0
  group=pending
  batch_size=spec.get('max_batch_envs',4096)//len(spec['case_labels'])
  for first in range(0,len(group),batch_size):
    batch=group[first:first+batch_size];cases=[]
    for job in batch:
     cfg=json.loads(Path(job['config']).read_text())
     case={key:cfg[key] for key in ('alphas','zero_middle','user_speed_m_s','seed','randomization','case_labels')}
     for key in ('evaluation_pulse','zero_reference_channels','alpha_switch'):
      if key in cfg:case[key]=cfg[key]
     case.update(terrain_path=job['terrain'],output=str(r/job['name']));cases.append(case)
    folder=r/'batches'/f'batch_{batch_index:03d}'
    while folder.exists():
     batch_index+=1;folder=r/'batches'/f'batch_{batch_index:03d}'
    manifest=r/f'batch_{batch_index:03d}.json';manifest.write_text(json.dumps(cases,indent=2))
    command=[lab,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle','runs/h75_selected_local300_flat/bundle','--config',batch[0]['config'],'--terrain',batch[0]['terrain'],'--output',str(folder),'--seconds','10','--case-manifest',str(manifest)]
    (r/'status.json').write_text(json.dumps(dict(status,batch=[job['name'] for job in batch]),indent=2))
    with (r/f'batch_{batch_index:03d}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    futures.extend(plots.submit(plot_job,job) for job in batch);status['completed']+=len(batch);batch_index+=1
  for future in futures:future.result()
 for job in jobs:
  lines+=['',f"## {job['name']}",'',f"![XY]({job['name']}/xy.png)",f"[奖励与速度]({job['name']}/overview.png) · [执行配置]({job['name']}/contract.json)"]
 (r/'RESULTS.md').write_text('\n'.join(lines)+'\n');status['stage']='complete';(r/'status.json').write_text(json.dumps(status,indent=2));raise SystemExit
for job in jobs:
 out=r/job['name'];(r/'status.json').write_text(json.dumps(status,indent=2))
 protocol=job.get('protocol',str(r/'protocol.json'))
 command=[lab,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle','runs/h75_selected_local300_flat/bundle','--config',job['config'],'--terrain',job['terrain'],'--output',str(out),'--seconds','10']
 if (spec.get('configured_lanes') or spec.get('natural_switch')):
  command+=['--configured-lanes']
 else:
  command+=['--channel-protocol',protocol,'--protocol-alpha',str(job.get('prefix_alpha',.5))]
 if 'alpha' in job:command+=['--fork-alpha',str(job['alpha'])]
 if not (a.resume and (out/'status.json').is_file() and json.loads((out/'status.json').read_text())['stage']=='complete'):
  with (r/(job['name']+'.log')).open('w') as log:
   subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((out/'status.json').read_text())
 assert result['stage']=='complete','Evaluation did not complete'
 forked=(out/'fork.json').exists()
 if not forked and not (spec.get('configured_lanes') or spec.get('natural_switch')):
  assert result['failures'][0]>0,'Source did not reach fork without a recorded physical termination'
  status.setdefault('source_failed_before_fork',[]).append(job['name'])
 with (r/(job['name']+'_plot.log')).open('w') as log:
  subprocess.run([cpu,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(out),'--lane-labels',json.dumps(spec['case_labels']),'--title',job['name']+' '+spec.get('plot_title','same-entry channel DEV')],stdout=log,stderr=subprocess.STDOUT,check=True)
 lines+=['',f"## {job['name']}",'',f"![XY]({job['name']}/xy.png)",f"[奖励与速度]({job['name']}/overview.png) · [执行配置]({job['name']}/contract.json)"]
 (r/'RESULTS.md').write_text('\n'.join(lines)+'\n');status['completed']+=1
status['stage']='complete';(r/'status.json').write_text(json.dumps(status,indent=2))
