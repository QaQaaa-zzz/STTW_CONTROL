"""Screen saved RB checkpoints on frozen causal entries; no retraining."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.skill_world_model import SkillWorldModel
from sttw_control.skill_sequence_rollout import handoff_joint_errors
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--prediction',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
qs=[];vs=[];hs=[];contexts=[];homes=[];pt=[];vt=[];labels=[]
for path in a.prediction:
 z=np.load(path/'predictions.npz');contract=json.loads((path/'contract.json').read_text());cases=json.loads((path/'cases.json').read_text());physical=Path(contract['oracle'])/'map48/base' if contract['oracle'] else Path(contract['evaluation']);cfg=json.loads((physical/'resolved_config.json').read_text());hand=cfg['ground_handoff'];schema=json.loads((physical/'model_state_schema.json').read_text());world=json.loads(((physical.parent/'world.json') if contract['oracle'] else Path(cfg['world_map'])).read_text());native=mujoco.MjModel.from_binary_path(str(physical/'terrain.mjb'));home=native.key('initial_state').qpos[[schema['qpos_names'].index(k) for k in ['hip_joint','knee_joint']]]
 qs.append(z['entry_qpos']);vs.append(z['entry_qvel']);hs.append(z['entry_history'])
 for c in cases:
  contexts.append([world['terrain']['front_x_m'],world['terrain']['width_m'],world['terrain'].get('far_bank_height_m',0.),c['speed'],0.]);homes.append(home);pt.append(hand['lock_position_tolerance_rad']);vt.append(hand['lock_velocity_tolerance_rad_s']);labels.append(dict(prediction=str(path),map=physical.parent.name if contract['evaluation'] else 'map48',root=c['root'],route=c['route']))
q=tensor(np.concatenate(qs));v=tensor(np.concatenate(vs));h=tensor(np.concatenate(hs));context=tensor(contexts);home=tensor(homes);pt=tensor(pt);vt=tensor(vt);n=len(q)
records=[];metrics=[]
for checkpoint in sorted(a.training.glob('seed*/model_*.pt')):
 saved=torch.load(checkpoint,map_location='cuda',weights_only=True);model=SkillWorldModel.from_weights(saved['model']);model.eval()
 with torch.no_grad():
  state=model.initialize(q,v,h[:,:381],h[:,381:],tensor([1,0,0]).expand(n,-1),tensor([0,1,0,0,0]).expand(n,-1),context);pred=model.rollout_until(state,context,125);pos,rate=handoff_joint_errors(pred['next_state'].qpos,pred['next_state'].qvel,home,schema);ready=pred['event_found']&(pos<pt)&(rate<vt)
  values=torch.stack((ready.float(),pos,rate,pred['event_found'].float(),pred['event_step'].float()),-1).cpu().numpy()
 dev=np.array([x['map']!='map48' for x in labels]);mechanism=~dev
 records.append(dict(checkpoint=str(checkpoint),seed=saved['seed'],update=saved['update'],dev_ready=int(values[dev,0].sum()),dev_cases=int(dev.sum()),mechanism_ready=int(values[mechanism,0].sum()),mechanism_cases=int(mechanism.sum()),dev_mean_joint_error_rad=float(values[dev,1].mean())))
 metrics.append(values)
ranked=sorted(records,key=lambda x:(-x['dev_ready'],x['dev_mean_joint_error_rad'],x['seed'],x['update']))
np.savez_compressed(a.output/'metrics.npz',values=np.asarray(metrics));(a.output/'cases.json').write_text(json.dumps(labels,indent=2)+'\n');(a.output/'checkpoints.json').write_text(json.dumps(records,indent=2)+'\n');(a.output/'shortlist.json').write_text(json.dumps(ranked[:3],indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(dict(checkpoints=len(records),conditions=n,runtime_s=time.perf_counter()-started,ranking='Original DEV72 lock-ready count descending, then mean joint error ascending; mechanism outcomes excluded from ranking',limits='Necessary joints/events only, not whole-body/contact safety or whole-chain success'),indent=2)+'\n');print(json.dumps(ranked[:3],indent=2))
