"""Persistent GPU PhysX/Torch PPO terrain finetuning with Go2 sampling scale."""
import argparse,copy,hashlib,json,os,shutil,subprocess,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=json.loads(a.config.read_text());a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
if s.get('backend','gym')=='lab':
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
else:
    from isaacgym import gymapi
sys.path.insert(0,s['go2_rsl_rl'])
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sttw_control.torch_path_policy import TorchPathPolicy,TrainablePathActor
from sttw_control.isaac_path_rollout import IsaacPathRollout
from rsl_rl.algorithms import PPO
from sttw_control.go2_ppo_adapter import H75ActorCritic,ActiveLaneStorage
torch.manual_seed(s['seed']);torch.set_num_threads(2)
previous=Path(s['resume_training']);previous_status=json.loads((previous/'status.json').read_text());source=Path(previous_status['final_policy']);identity=json.loads((source/'identity.json').read_text());cfg=json.loads((source/'config.json').read_text());checkpoint=torch.load(previous/'checkpoint.pt',map_location='cuda',weights_only=False)
actor=TrainablePathActor(TorchPathPolicy(source/'actor.npz',identity['files']['actor.npz'],cfg,1,'cuda'));critic=torch.nn.Sequential(torch.nn.Linear(352,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,1)).cuda();actor.load_state_dict(checkpoint['actor']);critic.load_state_dict(checkpoint['critic']);actor.log_std.data.fill_(np.log(s['initial_std']))
model=H75ActorCritic(actor,critic)
ppo=PPO(model,num_learning_epochs=s['epochs'],num_mini_batches=s['minibatches'],clip_param=s['clip'],gamma=s['gamma'],lam=s['lam'],value_loss_coef=s['value_loss_coef'],entropy_coef=s['entropy'],learning_rate=s['lr'],max_grad_norm=s['max_grad_norm'],use_clipped_value_loss=s['use_clipped_value_loss'],schedule=s['schedule'],desired_kl=s['desired_kl'],device='cuda:0',sym_loss=False)
optimizer=ppo.optimizer;optimizer.load_state_dict(checkpoint['optimizer']);lr=s['lr']
for group in optimizer.param_groups:group['lr']=lr
ppo.storage=ActiveLaneStorage(s['num_envs'],s['rollout_steps'],[351],[352],[2],device='cuda:0')
writer=SummaryWriter(str(a.output/'tb'));writer.add_scalar('stage/started',1,0);writer.flush();status=dict(stage='initializing',pid=os.getpid(),completed_updates=0,accepted_updates=0,valid_transitions=0,physics_steps=0,started=time.time());(a.output/'status.json').write_text(json.dumps(status,indent=2))
contract=dict(s,parent_bundle=str(source),parent_checkpoint_sha256=hashlib.sha256((previous/'checkpoint.pt').read_bytes()).hexdigest(),initialization='Actor/Critic/Adam continued; explicit Go2 std=1 and fresh seed',actor_hidden=cfg['network']['hidden'],critic_hidden=[512,256,128],physics_dt=.0002,lower_dt=.005,policy_dt=.02,max_physics_steps=s['updates']*s['rollout_steps']*s['num_envs']*100,max_policy_transitions=s['updates']*s['rollout_steps']*s['num_envs'],reward='unchanged frozen v3; failure discount remains .997, PPO discount .99',contact_monitor='GPU net contact force; not identical to old CPU individual contact query',reset='synchronous five-second episodes; failed lanes excluded until reset',go2_exceptions=['STTW network and action semantics retained','STTW physics/control timing retained','quadruped symmetry permutation not applicable','inactive-lane minibatch mask retained; upstream PPO replaces custom rollback/std clamp'])
(a.output/'contract.json').write_text(json.dumps(contract,indent=2));snapshot=a.output/'source';snapshot.mkdir()
for file in ['learning/cli/train_isaac_gpu.py','learning/src/sttw_control/isaac_vector_env.py','learning/src/sttw_control/isaac_path_rollout.py','learning/src/sttw_control/torch_path_reward.py','learning/src/sttw_control/isaaclab_vector_env.py','learning/src/sttw_control/go2_ppo_adapter.py',str(Path(s['go2_rsl_rl'])/'rsl_rl/algorithms/ppo.py'),str(Path(s['go2_rsl_rl'])/'rsl_rl/storage/rollout_storage.py'),str(a.config)]:shutil.copy2(file,snapshot/Path(file).name)
def save_policy(index):
    dest=a.output/('policy_%03d'%index);dest.mkdir();np.savez(dest/'actor.npz',**actor.export_arrays())
    for name in ['config.json','route.npz']:shutil.copy2(source/name,dest/name)
    meta=copy.deepcopy(identity);meta['files']['actor.npz']=hashlib.sha256((dest/'actor.npz').read_bytes()).hexdigest();meta['gpu_stage_update']=index;(dest/'identity.json').write_text(json.dumps(meta,indent=2));return dest
current=save_policy(0);plant=None
if s.get('backend','gym')=='lab':
    from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
    plant=IsaacLabVectorPlant(s['asset'],list(s['terrains'].values()),[i%len(s['terrains']) for i in range(s['num_envs'])],a.output)
env=IsaacPathRollout(source,s,actor,plant=plant);torch.cuda.synchronize();status['initialization_seconds']=time.time()-status['started'];status['stage']='training';history=[];best=-float('inf')
for update in range(s['updates']):
    start=time.time();rollout=[];traces=[];episode_rows=[]
    with torch.no_grad():
        for step in range(s['rollout_steps']):
            obs,co=env.observation();value=critic(co).flatten();distribution=actor.distribution(obs);out=env.step(actor.log_std.exp());next_obs,next_co=env.observation();next_value=critic(next_co).flatten()
            rollout.append(dict(obs=obs,critic=co,action=out['action'],reward=out['reward'],value=value,next_value=next_value,old_log=distribution.log_prob(out['action']).sum(-1),old_mean=distribution.loc,old_std=distribution.scale,done=out['done'],valid=out['valid'],failure=out['failure']));traces.append(out['lower_trace']);episode_rows.append([out['episode'],out['episode_tick']])
        data={k:torch.stack([r[k] for r in rollout]) for k in rollout[0]}
        mask=data['valid'];ppo.storage.valid=mask
        for t in range(s['rollout_steps']):
            tr=ppo.transition
            tr.observations=data['obs'][t];tr.critic_observations=data['critic'][t];tr.actions=data['action'][t]
            tr.values=data['value'][t,:,None];tr.actions_log_prob=data['old_log'][t];tr.action_mean=data['old_mean'][t];tr.action_sigma=data['old_std'][t]
            ppo.process_env_step(data['reward'][t],data['done'][t]|~mask[t],{})
        ppo.compute_returns(next_co)
    torch.cuda.synchronize();sample_seconds=time.time()-start;valid_count=int(mask.sum());start=time.time();losses=[];kl=0.;accepted=False
    if valid_count:
        value_loss,policy_loss,_,entropy=ppo.update();losses=[[policy_loss,value_loss,entropy]];accepted=True
        with torch.no_grad():kl=float(torch.distributions.kl_divergence(torch.distributions.Normal(data['old_mean'][mask],data['old_std'][mask]),actor.distribution(data['obs'][mask])).sum(-1).mean())
        lr=ppo.learning_rate
    else:ppo.storage.clear()
    torch.cuda.synchronize();optimize_seconds=time.time()-start;mean_reward=float(data['reward'][mask].mean()) if valid_count else None
    row=dict(update=update,scored_policy=str(current),valid_transitions=valid_count,failures=int(data['failure'].sum()),mean_step_reward=mean_reward,kl=kl,accepted=bool(accepted),lr=lr,loss_mean=np.mean(losses,axis=0).tolist() if losses else None,sample_seconds=sample_seconds,optimize_seconds=optimize_seconds,episode_first=episode_rows[0],episode_last=episode_rows[-1])
    groups=[]
    for terrain in range(3):
        for alpha in s['alphas']:
            lanes=(env.terrain==terrain)&(env.alpha==alpha);selected=mask[:,lanes];r=data['reward'][:,lanes];groups.append(dict(terrain=terrain,alpha=alpha,count=int(selected.sum()),reward_sum=float(r[selected].sum()),failures=int(data['failure'][:,lanes].sum())))
    row['groups']=groups;history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2))
    if mean_reward is not None and mean_reward>best:
        best=mean_reward;(a.output/'best_training.json').write_text(json.dumps(dict(policy=str(current),mean_step_reward=best,selection='pre-update stochastic rollout reward; varying episode phases; not fixed DEV best'),indent=2))
    np.savez_compressed(a.output/('rollout_%03d.npz'%update),reward=data['reward'].cpu().numpy(),valid=mask.cpu().numpy(),done=data['done'].cpu().numpy(),failure=data['failure'].cpu().numpy(),lower_trace=torch.stack(traces).cpu().numpy(),episode_steps=np.array(episode_rows),alpha=env.alpha.cpu().numpy(),terrain=env.terrain.cpu().numpy())
    for name,value in [('ppo/kl',kl),('ppo/accepted',int(accepted)),('ppo/lr',lr),('sampling/valid_steps',valid_count),('sampling/failures',row['failures']),('time/sampling',sample_seconds),('time/optimization',optimize_seconds)]:writer.add_scalar(name,value,update)
    if mean_reward is not None:writer.add_scalar('reward/mean_step',mean_reward,update)
    if losses:writer.add_scalar('loss/policy',row['loss_mean'][0],update);writer.add_scalar('loss/value',row['loss_mean'][1],update)
    writer.flush();current=save_policy(update+1);torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),update=update+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),a.output/'checkpoint.pt')
    status.update(completed_updates=update+1,accepted_updates=status['accepted_updates']+int(accepted),valid_transitions=status['valid_transitions']+valid_count,physics_steps=(update+1)*s['rollout_steps']*s['num_envs']*100,last_update=row,final_policy=str(current));(a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='groups'}),flush=True)
status.update(stage='training_complete',finished=time.time(),final_policy_scored=False);(a.output/'status.json').write_text(json.dumps(status,indent=2));writer.close();env.plant.close();subprocess.run(['notify-send','STTW GPU训练阶段结束','150轮阶段结束，后续评估待完成'],check=False)

if s.get("backend","gym")=="lab":app.close()
