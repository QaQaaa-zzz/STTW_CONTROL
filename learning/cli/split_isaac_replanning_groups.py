"""Export configured equal-size replanning groups without rerunning physics."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--group-size',type=int,default=7)
a=p.parse_args()
contract=json.loads((a.run/'contract.json').read_text())
settings=contract['settings']
raw=np.load(a.run/'trajectory.npz')
events=json.loads((a.run/'replanning_decisions.json').read_text())
status=json.loads((a.run/'status.json').read_text())
for n,group in enumerate(settings['fixed_randomization_groups']):
 ids=group['env_ids'];start=n*a.group_size;stop=start+a.group_size
 if ids!=list(range(start,stop)):raise ValueError('Expected contiguous equal-size groups')
 out=a.output/('seed'+str(group['seed']))/'original';out.mkdir(parents=True)
 arrays={key:np.take(raw[key],ids,axis=2 if key in ('lower_trace','wheel_contact_points','wheel_contact_forces') else 1) for key in raw.files}
 np.savez_compressed(out/'trajectory.npz',**arrays)
 c=deepcopy(contract);s=c['settings'];s.pop('fixed_randomization_groups')
 for key in ('alphas','zero_middle','case_labels'):
  s[key]=s[key][start:stop]
 s['replanning']['enabled']=s['replanning']['enabled'][start:stop]
 for key in ('target_alpha','execution_modes'):s['alpha_switch'][key]=s['alpha_switch'][key][start:stop]
 s.update(num_envs=a.group_size,trace_lanes=a.group_size,randomization=group['config'],seed=group['seed'])
 for key in ('origin','terrain'):c[key]=c[key][start:stop]
 for key in ('randomization_readback','delay_steps'):
  if key in c:c[key]={k:v[start:stop] for k,v in c[key].items()}
 meta=c['lab_physics'];meta['num_envs']=a.group_size
 for key in ('env_origins','masses','materials'):meta[key]=meta[key][start:stop]
 meta['terrain_layout']['clone_origins']=meta['terrain_layout']['clone_origins'][start:stop]
 c['batch_source']=dict(output=str(a.run.resolve()),lane_start=start,lane_stop=stop)
 (out/'contract.json').write_text(json.dumps(c,indent=2)+'\n')
 selected=[]
 for event in events:
  if event['lane'] in ids:
   item=dict(event,lane=event['lane']-start);selected.append(item)
   src=a.run/event['file'];dst=out/event['file'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 (out/'replanning_decisions.json').write_text(json.dumps(selected,indent=2)+'\n')
 sub=dict(status,physics_steps=len(arrays['reward'])*a.group_size*100,failures=arrays['failure'].sum(0).tolist(),valid_steps=arrays['valid'].sum(0).tolist(),elapsed_scope='shared batch time, not additive per group')
 (out/'status.json').write_text(json.dumps(sub,indent=2)+'\n')
 print(out)
