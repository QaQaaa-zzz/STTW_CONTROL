"""Bounded deterministic-actor diagnostics using the actual trained ditch environment."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device=cfg['device']).app
import torch
from sttw_control.phase_u_env import PhaseUEnv
from sttw_control.phase_u_actor_critic import PhaseUActorCritic
from sttw_control.jump_contact_recording import PhysicsContactRecorder
torch.manual_seed(cfg['evaluation_seed']);torch.set_num_threads(2)
env=PhaseUEnv(cfg,a.output)
policy=PhaseUActorCritic(env.num_obs,106,4,cfg['bundle']).to(cfg['device']).eval()
(a.output/'config.json').write_text(json.dumps(cfg,indent=2))
(a.output/'randomized_properties.json').write_text(json.dumps(env.properties))
reset=env.reset_idx;report=[];start=time.time()
for candidate in cfg['evaluation_checkpoints']:
 out=a.output/candidate['name'];out.mkdir()
 torch.manual_seed(cfg['evaluation_seed']);env.reset_idx=reset;obs,_=env.reset()
 terrain_id=int(env.terrain_ids[0])
 # One episode per candidate. Keep terminal states for evidence instead of auto-reset.
 env.reset_idx=lambda ids:None
 policy.load_state_dict(torch.load(candidate['path'],map_location=cfg['device'],weights_only=False)['model_state_dict'])
 recorder=PhysicsContactRecorder(env.plant,lambda:(torch.ones(1,dtype=torch.bool,device=cfg['device']),1))
 states=[];observations=[];actions=[];rewards=[];parts=[];codes=[];support=[]
 with torch.no_grad():
  for step in range(env.max_episode_length):
   observations.append(obs.cpu().numpy().copy());latent=policy.act_inference(obs);actions.append(torch.tanh(latent).cpu().numpy())
   obs,_,reward,done,info=env.step(latent)
   states.append(env.physical()[0].cpu().numpy());rewards.append(info['unscaled_reward'].cpu().numpy())
   parts.append({k:v.cpu().numpy() for k,v in info['reward_parts'].items()});codes.append(info['end_code'].cpu().numpy());support.append(env.platform_held.cpu().numpy().copy())
   if bool(done.all()):break
 env.plant.step=recorder.original_step;recorder.save(out/'physics_contact.npz')
 if cfg.get('record_verified_diagnostics'):
  diagnostic=env.verified_jump.diagnostics
  np.savez_compressed(out/'verified_diagnostics.npz',time=(np.arange(len(diagnostic))+1)*env.plant.physics_dt,
                      **{k:torch.stack([row[k] for row in diagnostic]).cpu().numpy() for k in diagnostic[0]})
 np.savez_compressed(out/'trajectory.npz',time=(np.arange(len(states))+1)*env.dt,root=np.stack(states),observation=np.stack(observations),action=np.stack(actions),reward=np.stack(rewards),end_code=np.stack(codes),held=np.stack(support),**{'reward_'+k:np.stack([row[k] for row in parts]) for k in parts[0]})
 tr=np.stack(states)[:,0]
 row=dict(candidate=candidate,terrain_id=terrain_id,terrain_spec=env.terrain_specs[terrain_id],steps=len(states),duration_s=len(states)*env.dt,terminal_code=int(codes[-1][0]),final_xyz=tr[-1,:3].tolist(),peak_root_z=float(tr[:,2].max()),max_root_x=float(tr[:,0].max()),ditch_cross_hold=bool(env.platform_held[0]),reward_total=float(np.stack(rewards).sum()),reward_parts={k:float(np.stack([x[k] for x in parts]).sum()) for k in parts[0]},scope='Single fixed-seed TRAIN-domain diagnostic with frozen sampled Go2 properties/noise and mean actor; no ground-controller handoff or complete-map success claim')
 if cfg.get('verified_ditch_reward'):
  row['verified_stage']=int(info['jump_stage'][0]);row['verified_success']=bool(info['jump_success'][0])
 (out/'summary.json').write_text(json.dumps(row,indent=2));report.append(row)
 (a.output/'status.json').write_text(json.dumps(dict(stage='evaluating',completed=len(report),target=len(cfg['evaluation_checkpoints']))))
(a.output/'summary.json').write_text(json.dumps(dict(cases=report,elapsed_s=time.time()-start),indent=2))
(a.output/'status.json').write_text(json.dumps(dict(stage='completed',completed=len(report),elapsed_s=time.time()-start)))
os.sync();os._exit(0)
