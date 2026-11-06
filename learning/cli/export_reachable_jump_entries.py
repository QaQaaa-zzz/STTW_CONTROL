"""Export actual pre-jump states/history for a declared adaptation input bank."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--skill',default='source')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
arrays={k:[] for k in ['root_pose','root_velocity','joint_position','joint_velocity','observation','last_action']};entries=[];joint_order=None
for c in json.loads(a.manifest.read_text())['cases']:
 if c['skill']!=a.skill:continue
 spec=json.loads(Path(c['config']).read_text());run=Path(spec['output']);h=json.loads((run/'handoff.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
 if joint_order is None:joint_order=protocol['joint_names']
 if protocol['joint_names']!=joint_order:raise ValueError('Mixed live joint orders need explicit remapping')
 root=np.asarray(h['root'][0],dtype=np.float32);pose=root[:7].copy();pose[3:]=root[[6,3,4,5]];pose[0]+=3.6-spec['platform_front_x']
 with np.load(run/'trajectory.npz') as data:
  obs_index=round(h['time']/protocol['jump_dt']);obs=data['observations'][obs_index,0].copy()
  trace_index=int(np.argmin(abs(data['trace'][:,0,0]-h['time'])));last=data['action'][trace_index,0].copy()
 arrays['root_pose'].append(pose);arrays['root_velocity'].append(root[7:13]);arrays['joint_position'].append(h['position'][0]);arrays['joint_velocity'].append(h['velocity'][0]);arrays['observation'].append(obs);arrays['last_action'].append(last)
 entries.append(dict(case_id=c['case_id'],origin_id=spec['origin_id'],source_config=c['config'],handoff_time_s=h['time'],observation_index=obs_index,source_platform_front_x=spec['platform_front_x'],translated_platform_front_x=3.6,physics_dt=protocol['physics_dt']))
np.savez_compressed(a.output/'entries.npz',**{k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()},joint_names=np.array(joint_order))
metadata=dict(status='prepared_inputs_only_not_connected_to_training',entries=entries,count=len(entries),independent_origins=len({e['origin_id'] for e in entries}),sha256=hashlib.sha256((a.output/'entries.npz').read_bytes()).hexdigest(),
 pose_contract='world xyz and wxyz; root world velocities, actual live joint ordering; only world x translated with the platform',
 observation_contract='Original 76-D pre-action jump observation at exact handoff control tick; three 25-D history frames plus signal; no rebuilding from synthetic zero history',
 scope='Observed accessible states, not full hidden PhysX snapshots. Same nominal upstream root across approaches; not independent 18-root data. No failed landing outcome is an input. No PPO updates.')
(a.output/'entries.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(dict(count=metadata['count'],independent_origins=metadata['independent_origins'],status=metadata['status'])))
