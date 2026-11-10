"""Original H75 PPO and weights with the shared Isaac/Go2-randomized collector."""
import argparse,json,os,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--evaluate',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--eval-alpha',type=float,choices=[0,.25,.5,.75,1]);a=p.parse_args();s=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(Path(s['rsl_runtime']).resolve()))
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from sttw_control.go2_path_env import Go2PathEnv
from sttw_control.torch_path_training import make_algorithm
cfg=json.loads((Path(s['bundle'])/'config.json').read_text());torch.manual_seed(s['seed']);torch.set_num_threads(2)
s['evaluation_mode']=a.evaluate;s['trace_lanes']=s['num_envs']
started=time.time();status=dict(stage='initializing',pid=os.getpid(),completed_updates=0,updates=s['updates']);status_path=a.output/'status.json';status_path.write_text(json.dumps(status))
(a.output/'config.json').write_text(json.dumps(dict(settings=s,original_training=cfg['training_future_phase_C'],initialization=('checkpoint Actor/Critic/Adam resumed; environments and RNG reset' if a.resume else 'original H75 Actor/Critic/log_std; fresh original-method Adam')),indent=2))
env=Go2PathEnv(s,a.output)
(a.output/'randomization.json').write_text(json.dumps(env.rollout.randomization_properties))
if s.get('reset_mode')=='common_start':
 _,start_pose,start_speed=env.plant.observe()
 assert torch.allclose(start_pose[:,:2],torch.zeros_like(start_pose[:,:2]),atol=1e-6)
 assert torch.allclose(env.route.xy[:,0],torch.zeros_like(env.route.xy[:,0]),atol=1e-6)
 assert not env.rollout.task_tick.any()
 assert not env.rollout.lower.mask.any()
 assert (env.rollout.middle.mask.sum(-1)==1).all()
 (a.output/'common_start_check.json').write_text(json.dumps(dict(passed=True,environments=s['num_envs'],position_xy_max_m=float(start_pose[:,:2].abs().max()),reference_start_xy_max_m=float(env.route.xy[:,0].abs().max()),start_speed_range_m_s=[float(start_speed.min()),float(start_speed.max())],task_tick=0,lower_history='cleared',middle_history='one common-start frame',prefix_states_restored=False),indent=2))

if a.evaluate:env.rollout.alpha.copy_(torch.tensor([a.eval_alpha if a.eval_alpha is not None else s['alphas'][i%5] for i in range(s['num_envs'])],device='cuda'))
(a.output/'initial_cases.json').write_text(json.dumps(dict(route_ids=env.route.route_ids.cpu().tolist(),alpha=env.rollout.alpha.cpu().tolist(),terrain_ids=env.terrain_kind_id.cpu().tolist(),origins=env.plant.origin.cpu().tolist(),terrain_centers=env.terrain_center.cpu().tolist()),indent=2))
if a.evaluate or s.get('reset_mode')=='common_start':
 np.savez_compressed(a.output/'initial_state.npz',root=env.plant.roots.cpu().numpy(),dofs=env.plant.dofs.cpu().numpy(),pulse=env.force_world.cpu().numpy(),servo_gain=env.plant.servos.gain.cpu().numpy(),steer_offset=env.rollout.randomization.steer_offset.cpu().numpy())
def observation():
 return TensorDict({'policy':env.get_observations(),'critic':env.get_privileged_observations()},batch_size=[s['num_envs']])
obs=observation();algo=make_algorithm(obs,s['rollout_steps'],cfg,'cuda');original=torch.load(a.checkpoint or s['initial_policy'],map_location='cuda',weights_only=False);algo.policy.load_state_dict(original['policy'])
start_update=int(original['update']) if a.resume else 0
if a.resume:algo.optimizer.load_state_dict(original['optimizer'])
status.update(completed_updates=start_update);status_path.write_text(json.dumps(status))
writer=SummaryWriter(str(a.output/'tb'));writer.add_scalar('stage/started',1,0);writer.flush()
def save(update,name):
 torch.save(dict(policy=algo.policy.state_dict(),optimizer=algo.optimizer.state_dict(),update=update,config=cfg),a.output/name)
save(start_update,'model_initial.pt');history=[];rejections=0
if a.evaluate:
 frames=[];flags=[];rewards=[];domains=[];faults=[];poses=[];progresses=[];body_frames=[]
 if s.get("record_body_poses",False):
  (a.output/"contract.json").write_text(json.dumps(dict(bodies=env.plant.bodies,origin=env.plant.origin.cpu().tolist(),physics_dt=env.plant.physics_dt,settings=dict(s,alphas=env.rollout.alpha.cpu().tolist()),quaternion_order="xyzw",reference_routes="reference_routes.npz"),indent=2))
 with torch.no_grad():
  for step in range(500):
   action=algo.policy.act_inference(obs);_,_,reward,done,_=env.step(action);out=env.last_out
   if s.get("record_body_poses",False):body_frames.append(out["body_poses"].cpu().numpy())
   poses.append(env.plant.observe()[1].cpu().numpy());progresses.append(env.rollout.middle.projection['progress'].cpu().numpy());domains.append(env.last_terminal['domain'].cpu().numpy());faults.append(env.last_terminal['fault'].cpu().numpy());frames.append(out['lower_trace'].cpu().numpy());rewards.append(reward.cpu().numpy());flags.append(out['failure'].cpu().numpy());obs=observation()
 np.savez_compressed(a.output/'trajectory.npz',**({"body_poses":np.stack(body_frames)} if body_frames else {}),lower_trace=np.stack(frames),reward=np.stack(rewards),failure=np.stack(flags),pose=np.stack(poses),progress=np.stack(progresses),domain_exit=np.stack(domains),policy_fault=np.stack(faults),alpha=env.rollout.alpha.cpu().numpy(),route_ids=env.route.route_ids.cpu().numpy(),terrain=env.terrain_kind_id.cpu().numpy(),terrain_entered=env.terrain_entered.cpu().numpy(),terrain_contacted=env.terrain_contacted.cpu().numpy(),terrain_cleared=env.terrain_cleared.cpu().numpy())
 status.update(stage='evaluation_complete',failed_lanes=int(np.stack(flags).any(0).sum()),conditions=s['num_envs'],elapsed_s=time.time()-started);status_path.write_text(json.dumps(status,indent=2,default=lambda x:float(x)))
else:
 for update in range(start_update,s['updates']):
  begin=time.time();rs=[];failures=0;domains=0;faults=0;ended=0;contacted_episodes=0;cleared_episodes=0;alpha_stats={str(x):dict(samples=0,reward_sum=0.,early_terminations=0) for x in s['alphas']}
  with torch.no_grad():
   for step in range(s['rollout_steps']):
    sample_alpha=env.rollout.alpha.clone();action=algo.act(obs);_,_,reward,done,info=env.step(action);obs=observation();algo.process_env_step(obs,reward,done,{})
    rs.append(reward.mean().item());failures+=int(env.last_out['failure'].sum());ended+=int(done.sum());domains+=int(env.last_terminal['domain'].sum());faults+=int(env.last_terminal['fault'].sum());contacted_episodes+=int((done&env.last_exposure['contacted']).sum());cleared_episodes+=int((done&env.last_exposure['cleared']).sum())
    for alpha in s['alphas']:
     mask=sample_alpha==alpha;g=alpha_stats[str(alpha)];g['samples']+=int(mask.sum());g['reward_sum']+=float(reward[mask].sum());g['early_terminations']+=int(env.last_out['failure'][mask].sum())
   algo.compute_returns(obs)
  sample=time.time()-begin;begin=time.time();metrics=algo.update();opt=time.time()-begin
  row=dict(update=update+1,sampling_model_update=update,mean_reward=float(np.mean(rs)),early_terminations=failures,tracking_domain_signals=domains,policy_fault_signals=faults,episodes=ended,contacted_episodes=contacted_episodes,cleared_episodes=cleared_episodes,sample_seconds=sample,optimize_seconds=opt,optimizer=metrics,alpha_groups=alpha_stats)
  history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2,default=lambda x:float(x)));save(update+1,'checkpoint.pt')
  if (update+1)%s['save_interval']==0 or update+1==s['updates']:save(update+1,f'model_{update+1:04d}.pt')
  for key in ['surrogate','value','entropy','mean_kl','accepted_epochs']:writer.add_scalar('ppo/'+key,float(metrics[key]),update+1)
  writer.add_scalar('reward/mean_step',row['mean_reward'],update+1);writer.add_scalar('termination/early',failures,update+1);writer.add_scalar('time/sampling',sample,update+1);writer.flush()
  status.update(stage='training',completed_updates=update+1,elapsed_s=time.time()-started,last_update=row);status_path.write_text(json.dumps(status,indent=2,default=lambda x:float(x)));print(json.dumps(row,default=lambda x:float(x)),flush=True)
  if metrics['nonfinite_stop']:raise RuntimeError('Original PPO nonfinite stop; checkpoint retained')
  rejections=rejections+1 if metrics['hard_kl_stop'] else 0
  if metrics['hard_kl_stop']:
   algo.optimizer.param_groups[0]['lr']=max(cfg['ppo_acceptance']['actor_lr_floor'],algo.optimizer.param_groups[0]['lr']/2);algo.hard_kl_stop=False
  if rejections>=cfg['ppo_acceptance']['hard_reject_consecutive_stop']:raise RuntimeError('Original PPO three-consecutive-hard-rejection stop')
 status.update(stage='training_complete');status_path.write_text(json.dumps(status,indent=2,default=lambda x:float(x)))
writer.close();app.close(wait_for_replicator=False,skip_cleanup=True)
