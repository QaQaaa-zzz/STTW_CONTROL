"""Extract full-joint RB sequences, retaining complete map/rollout split groups."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--stop-at-handoff',action='store_true',help='End at first20ms sampled phase2;exclude subsequent ground-control targets')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
manifest=json.loads((a.collection/'manifest.json').read_text())
rows=[];omitted=[];arrays=[];offset=0
for entry in manifest['entries']:
 source=Path(entry['evaluation']);cfg=json.loads((source/'resolved_config.json').read_text())
 world=json.loads(Path(cfg['world_map']).read_text());terrain=world['terrain']
 # The entire generated terrain layout and all its rollouts have one role.
 split='TRAIN' if entry['map_role']=='TRAIN' else 'DEV'
 family=hashlib.sha256(json.dumps(terrain,sort_keys=True).encode()).hexdigest()
 status=json.loads((source/'status.json').read_text())
 for row in status['results']:
  path=source/row['case_dir']/'trajectory.npz';z=np.load(path)
  phase=z['trace'][:,-1].astype(int);rb=np.flatnonzero(phase==1)
  if not len(rb):
   omitted.append(dict(path=str(path),reason='never entered RB',termination=row['termination'],split=split));continue
  start=int(rb[0]);end=len(phase)-1 if row['handoff_s'] is None else min(len(phase)-1,int(round(row['handoff_s']/.005))+40)
  ids=np.arange(start,end+1,4)
  if a.stop_at_handoff:
   handoff=np.flatnonzero(phase[ids]==2)
   if len(handoff):ids=ids[:handoff[0]+1]
  t=z['trace'][ids,0]
  assert np.allclose(np.diff(t),.02,atol=1e-6)
  qpos=z['qpos'][ids].astype('float32');qvel=z['qvel'][ids].astype('float32')
  history=np.concatenate((z['controller_state'][ids],z['rb_state'][ids]),-1).astype('float32')
  contacts=z['wheel_contact_flags'][ids].astype('float32');phases=phase[ids]
  variant=row['variant'];speed=variant.get('approach_speed_m_s',cfg['ground_handoff']['approach']['speed_m_s'])
  context=np.tile(np.array([terrain['front_x_m'],terrain['width_m'],terrain.get('far_bank_height_m',0.),speed,cfg['ground_handoff']['approach']['alpha']],dtype='float32'),(len(ids),1))
  arrays.append(dict(qpos=qpos,qvel=qvel,history=history,contact=contacts,phase=phases,context=context,time=t,action=z['applied_targets'][ids],action_valid=z['applied_target_valid'][ids]))
  rows.append(dict(path=str(path),evaluation=str(source),map=entry['layout'],map_family=family,split=split,case_dir=row['case_dir'],route=row['route'],variant=variant,start=offset,length=len(ids),raw_start_step=start,termination=row['termination'],rb_entry_s=row['rb_entry_s'],first_far_contact_s=row['first_far_contact_s'],handoff_s=row['handoff_s'],terrain_model=str(source/'terrain.mjb'),snapshot_group=str(source/'rb_run3_initial_snapshot.pkl')));offset+=len(ids)
combined={k:np.concatenate([x[k] for x in arrays]) for k in arrays[0]}
train=np.concatenate([np.arange(r['start'],r['start']+r['length']) for r in rows if r['split']=='TRAIN'])
features=np.concatenate((combined['qpos'],combined['qvel'],combined['history'],combined['contact'],np.eye(5,dtype='float32')[combined['phase']],combined['context']),-1)
# All normalizers come from TRAIN; constant dimensions keep a nonzero engineering floor.
mean=features[train].mean(0);scale=np.maximum(features[train].std(0),.1)
transitions=np.concatenate([np.arange(r['start'],r['start']+r['length']-1) for r in rows if r['split']=='TRAIN'])
velocity_scale=np.maximum((combined['qvel'][transitions+1]-combined['qvel'][transitions]).std(0),.05)
history_scale=np.maximum((combined['history'][transitions+1]-combined['history'][transitions]).std(0),.1)
np.savez_compressed(a.output/'sequences.npz',**combined)
np.savez(a.output/'normalization.npz',input_mean=mean,input_scale=scale,velocity_scale=velocity_scale,history_scale=history_scale,history_state_scale=np.maximum(combined['history'][train].std(0),.1))
train_groups={r['map_family'] for r in rows if r['split']=='TRAIN'};dev_groups={r['map_family'] for r in rows if r['split']=='DEV'}
assert not train_groups&dev_groups
train_sources={r['snapshot_group'] for r in rows if r['split']=='TRAIN'};dev_sources={r['snapshot_group'] for r in rows if r['split']=='DEV'}
assert not train_sources&dev_sources
contract=dict(stop_at_handoff=a.stop_at_handoff,source=str(a.collection),rows=rows,omitted=omitted,counts={s:sum(r['split']==s for r in rows) for s in ['TRAIN','DEV']},dt_s=.02,controller_dim=381,rb_dim=156,context=['gap_front_x_m','gap_width_m','far_bank_height_m','speed_request_m_s','alpha'],time_alignment='Actual RB entry, then each fourth lower tick; contact row t summarizes prior5ms; no future action/contact inputs. Future action retained only as labels. Full history read only at initial model state.',grouping='Entire generated terrain layouts, full rollouts and original snapshot sources stay in one split; old frozen roles preserved. All maps share rough/ridge/gap topology; no novel-terrain-type TEST claim.',normalization='TRAIN rows and within-rollout transitions only',scope='RB entry through first sampled phase2 at or after measured handoff (less than20ms rounding),or real failure;no post-handoff tail' if a.stop_at_handoff else 'RB entry through measured ground handoff +0.2s, or real failure; no padding fabricated beyond terminal',fixed_skill='shared225 adapted RB; verify resolved actor identity before training',train_map_families=sorted(train_groups),dev_map_families=sorted(dev_groups))
(a.output/'manifest.json').write_text(json.dumps(contract,indent=2)+'\n')
print(json.dumps(dict(counts=contract['counts'],omitted=len(omitted),rows=offset,history_dim=combined['history'].shape[-1])))
