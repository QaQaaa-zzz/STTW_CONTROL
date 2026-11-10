"""Execute the finite declared same-entry channel matrix using existing tools."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();r=a.run;spec=json.loads((r/'protocol.json').read_text());jobs=json.loads((r/'jobs.json').read_text());lab='/home/qy/miniconda3/envs/env_isaaclab/bin/python';cpu='/home/qy/miniconda3/envs/mjp/bin/python';lines=['# '+spec.get('title','同入障状态通道消融'),'','原H75+Local300；DEV机制实验，完整统计和选档TEST尚待完成。',''];(r/'RESULTS.md').write_text('\n'.join(lines));status={'stage':'running','completed':0,'planned':len(jobs)}
for job in jobs:
 out=r/job['name'];(r/'status.json').write_text(json.dumps(status,indent=2))
 protocol=job.get('protocol',str(r/'protocol.json'))
 command=[lab,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle','runs/h75_selected_local300_flat/bundle','--config',job['config'],'--terrain',job['terrain'],'--output',str(out),'--seconds','10']
 if spec.get('natural_switch'):
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
 if not forked and not spec.get('natural_switch'):
  assert result['failures'][0]>0,'Source did not reach fork without a recorded physical termination'
  status.setdefault('source_failed_before_fork',[]).append(job['name'])
 with (r/(job['name']+'_plot.log')).open('w') as log:
  subprocess.run([cpu,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(out),'--lane-labels',json.dumps(spec['case_labels']),'--title',job['name']+' '+spec.get('plot_title','same-entry channel DEV')],stdout=log,stderr=subprocess.STDOUT,check=True)
 lines+=['',f"## {job['name']}",'',f"![XY]({job['name']}/xy.png)",f"[奖励与速度]({job['name']}/overview.png) · [执行配置]({job['name']}/contract.json)"]
 (r/'RESULTS.md').write_text('\n'.join(lines)+'\n');status['completed']+=1
status['stage']='complete';(r/'status.json').write_text(json.dumps(status,indent=2))
