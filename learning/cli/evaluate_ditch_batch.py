"""Evaluate fixed ditch candidates in one Isaac scene, with measured STTW handback."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device=cfg['device']).app
import torch
from sttw_control.phase_u_env import PhaseUEnv
from sttw_control.phase_u_actor_critic import PhaseUActorCritic
from sttw_control.ditch_ground_handoff import VectorDitchHandoff
torch.set_num_threads(2);torch.manual_seed(cfg['evaluation_seed'])
start=time.monotonic();env=PhaseUEnv(cfg,a.output)
task=VectorDitchHandoff(env,cfg['ground_handoff'],a.output)
policy=PhaseUActorCritic(env.num_obs,106,4,cfg['bundle']).to(cfg['device']).eval()
if cfg.get('evaluation_checkpoint'):
 policy.load_state_dict(torch.load(cfg['evaluation_checkpoint'],map_location=cfg['device'],weights_only=False)['model_state_dict'])
(a.output/'config.json').write_text(json.dumps(cfg,indent=2))
(a.output/'randomized_properties.json').write_text(json.dumps(env.properties))
obs,_=task.reset();rows=[];valid=[];rewards=[];stages=[];phases=[];observations=[];actions=[];codes=[];histories=[];road_history=[];route_history=[]
initial_root=env.physical()[0].cpu().numpy();initial_obs=obs.cpu().numpy().copy()
if cfg.get('event_planner'):
 from sttw_control.ditch_event_model import EventPlanner
 query_start=time.monotonic();planner=EventPlanner(cfg['event_planner']['model']);g=task.ground
 history=torch.cat([v.flatten(1) for v in (g.controller.eso,g.controller.disturbance[:,None],g.controller.gains,g.previous_reference,g.previous_command,g.policy.filtered,g.policy.offsets,g.middle.frames,g.middle.mask)],-1).cpu().numpy()
 state=planner.encode(initial_root,initial_obs,history,np.zeros(env.num_envs))
 parameters=np.asarray(planner.meta['candidate_parameters'],np.float32);nc=len(parameters)
 prediction=planner.score(np.repeat(state,nc,axis=0),np.tile(parameters,(env.num_envs,1)),particles=32)
 scores=np.asarray(prediction['score']).reshape(env.num_envs,nc);chosen=scores.argmax(1)
 selected=g.route.route_ids.cpu().numpy().copy();select_lanes=np.asarray(cfg['event_planner']['lanes'],int);selected[select_lanes]=chosen[select_lanes]
 g.route.route_ids.copy_(torch.as_tensor(selected,device=env.device))
 if parameters.shape[1]==8:
  task.ground_only.copy_(torch.as_tensor(parameters[selected,-1]<.5,device=env.device));task.phase.copy_(torch.where(task.ground_only,3,0))
 g.resume_ground_lanes(env.all_ids,task.approach['speed_m_s'],task.approach['alpha'])
 (a.output/'planning.json').write_text(json.dumps(dict(selected=selected.tolist(),model_choice=chosen.tolist(),scores=scores.tolist(),query_s=time.monotonic()-query_start,prediction=prediction,scope='Online pre-motion candidate query from actual reset state and controller history; no measured futures supplied'),indent=2))

moving=None
if cfg.get('moving_event_planner'):
 from sttw_control.moving_ditch_planner import MovingDitchPlanner
 moving=MovingDitchPlanner(task,cfg['moving_event_planner'],a.output)
with torch.no_grad():
 for step in range(env.max_episode_length):
  if moving is not None:moving.tick(step,obs)
  route_history.append(task.ground.route.route_ids.clone())
  active=~task.finished;latent=policy.act_inference(obs)
  observations.append(obs.clone());actions.append(torch.tanh(latent));valid.append(active.clone())
  if cfg.get('record_controller_history'):
   g=task.ground
   histories.append(torch.cat([value.flatten(1) for value in (g.controller.eso,g.controller.disturbance[:,None],g.controller.gains,g.previous_reference,g.previous_command,g.policy.filtered,g.policy.offsets,g.middle.frames,g.middle.mask)],-1).clone())
  obs,_,reward,done,info=task.step(latent)
  road_history.append(info['road_violation'].clone());rows.append(env.physical()[0]);rewards.append(info['unscaled_reward']);stages.append(info['jump_stage']);phases.append(task.phase.clone());codes.append(info['end_code'])
  if step%50==0:
   (a.output/'status.json').write_text(json.dumps(dict(stage='evaluating',steps=step+1,active=int((~task.finished).sum()),elapsed_s=time.monotonic()-start)))
  if bool(task.finished.all()):break
if moving is not None:moving.save()
arrays=dict(time=(np.arange(len(rows))+1)*env.dt,initial_root_xyzw=initial_root,initial_observation=initial_obs)
if histories:arrays['controller_history']=torch.stack(histories).cpu().numpy()
arrays['reference_route_id']=torch.stack(route_history).cpu().numpy()
arrays['road_violation_history']=torch.stack(road_history).cpu().numpy()
if task.physics_roots:
 arrays['physics_root_xyzw']=torch.stack(task.physics_roots).cpu().numpy()
 arrays['physics_contact_mode']=torch.stack(task.physics_contact_modes).cpu().numpy()
 arrays['physics_valid']=torch.stack(task.physics_valid).cpu().numpy()
 arrays['physics_time']=(np.arange(len(task.physics_roots))+1)*env.plant.physics_dt
for name,values in dict(root_xyzw=rows,valid=valid,reward=rewards,stage=stages,phase=phases,observation=observations,action=actions,end_code=codes).items():
 arrays[name]=torch.stack(values).cpu().numpy()
np.savez_compressed(a.output/'trajectories.npz',**arrays,event_time=task.event_time.cpu().numpy(),event_root_xyzw=task.event_root.cpu().numpy(),event_names=np.asarray(task.event_names),road_violation=task.road_violation.cpu().numpy(),event_road_violation=task.event_road_violation.cpu().numpy())
summary=dict(cases=[dict(lane=i,success=bool(task.reached_goal[i]),skill='ground_only' if bool(task.ground_only[i]) else 'jump_then_ground',end_code=int(task.terminal_code[i]),handoff_s=float(task.handoff_time[i]) if torch.isfinite(task.handoff_time[i]) else None,entry_s=float(task.entry_time[i]) if torch.isfinite(task.entry_time[i]) else None,steps=int(arrays['valid'][:,i].sum()),road_violation=bool(task.road_violation[i])) for i in range(env.num_envs)],elapsed_s=time.monotonic()-start,physics_lane_steps=env.total_physics_steps,scope=cfg.get('data_role','Declared evaluation batch')+'; original physical thresholds; success is selected skill sequence plus task goal without prior original-road violation; jump routes require measured far-bank handback, ground-only routes do not; not old support hold')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
(a.output/'status.json').write_text(json.dumps(dict(stage='completed',**summary),indent=2))
os.sync();os._exit(0)
