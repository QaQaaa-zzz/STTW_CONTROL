"""Export frozen JIT Actor/normalizer arrays and compare deterministic inference."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from sttw_control.pendulum_skill import PendulumSkill,TorchPendulumSkill
p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,default=Path('learning/configs/pendulum_bike_skill_sources.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--skills',nargs='+');a=p.parse_args()
registry=json.loads(a.registry.read_text());results=[]
for name in (a.skills or registry['records']):
 skill=PendulumSkill(a.registry,name);out=a.output/name;out.mkdir(parents=True,exist_ok=False)
 params=skill.payload.actor_params['params'];normal=skill.payload.observation_normalizer
 arrays=dict(mean=np.asarray(normal.mean['state']),std=np.asarray(normal.std['state']))
 for i in range(4):
  for field in ('kernel','bias'):arrays[f'{field}_{i}']=np.asarray(params[f'hidden_{i}'][field])
 critic=skill.payload.critic_params['params']
 arrays['critic_mean']=np.asarray(normal.mean['privileged_state']);arrays['critic_std']=np.asarray(normal.std['privileged_state'])
 for i in range(4):
  for field in ('kernel','bias'):arrays[f'critic_{field}_{i}']=np.asarray(critic[f'hidden_{i}'][field])
 np.savez(out/'actor.npz',**arrays)
 mapping={k:getattr(skill.bundle.action_mapping,k) for k in ('ctrl_min','ctrl_max','hip_initial','knee_initial','base_rear_speed','rear_speed_delta','joint_target_semantics','knee_target_delta')}
 mapping={k:np.asarray(v).tolist() if not isinstance(v,str) else v for k,v in mapping.items()}
 identity=dict(skill_id=name,source=skill.record,artifact_sha256=hashlib.sha256((out/'actor.npz').read_bytes()).hexdigest(),mapping=mapping,network='76-256-256-256-8 SiLU; tanh first four logits, deterministic',conversion='exact arrays, no training or architecture change')
 (out/'identity.json').write_text(json.dumps(identity,indent=2))
 model=TorchPendulumSkill(out,'cpu');rng=np.random.default_rng(18);obs=(arrays['mean']+rng.normal(size=(16,76))*arrays['std']).astype(np.float32);obs[:,-1]=np.arange(16)%2;knee=np.linspace(-1.5,2.5,16,dtype=np.float32)
 actual,target=model.infer(torch.from_numpy(obs),torch.from_numpy(knee));reference=[skill.infer(o,k) for o,k in zip(obs,knee)]
 error=float(np.max(np.abs(actual.numpy()-np.asarray([x[0] for x in reference]))));target_error=float(np.max(np.abs(target.numpy()-np.asarray([x[1] for x in reference]))))
 np.savez(out/'parity.npz',observation=obs,knee=knee,torch_action=actual.numpy(),torch_target=target.numpy(),jax_action=np.asarray([x[0] for x in reference]),jax_target=np.asarray([x[1] for x in reference]))
 if error>1e-5 or target_error>1e-4:raise ValueError((name,error,target_error))
 results.append(dict(skill=name,samples=16,action_max_error=error,target_max_error=target_error,physics_steps=0,optimizer_updates=0))
(a.output/'parity.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
