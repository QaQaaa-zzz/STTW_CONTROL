"""Bounded H75 weights-only PPO adaptation using recorded Isaac transitions.

Sequential CPU physics is an engineering collector, not a scalable training backend.
The original JAX reward is reconstructed before advantages and optimization.
"""
import argparse, copy, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sttw_control.torch_path_policy import TorchPathPolicy, TrainablePathActor

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
root=Path.cwd();base=root/'runs/isaac_h75_curve_bundle';identity=json.loads((base/'identity.json').read_text());cfg=json.loads((base/'config.json').read_text())
seed=20261105;torch.manual_seed(seed);torch.set_num_threads(2)
actor=TrainablePathActor(TorchPathPolicy(base/'actor.npz',identity['files']['actor.npz'],cfg,1,'cuda'))
critic=torch.nn.Sequential(torch.nn.Linear(352,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,1)).cuda()
optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1e-5)
writer=SummaryWriter(str(a.output/'tb'));writer.add_scalar('stage/started',1,0);writer.flush()
env=os.environ.copy();env['PATH']='/home/qy/miniconda3/envs/go2_isaacgym/bin:'+env['PATH'];env['LD_LIBRARY_PATH']='/home/qy/miniconda3/envs/go2_isaacgym/lib';env['PYTHONPATH']=str(root/'learning/src')
reward_env=env.copy();reward_env['JAX_PLATFORMS']='cpu';reward_env.pop('LD_LIBRARY_PATH',None)
contract=dict(initialization='H75 weights_only; fresh critic/Adam/std/RNG',alphas=[0,.25,.5,.75,1],terrain_schedule=['slope','rough_slope'],updates=2,max_physics_steps=250000,max_policy_transitions=2500,episode_s=5,physics_dt=.0002,lower_dt=.005,middle_dt=.02,lr=1e-5,epochs=5,minibatches=4,clip=.2,entropy=.01,gamma=.997,lam=.95,max_grad_norm=1,hard_kl=.02,std_bounds=[.05,.5],role='engineering on-policy adaptation DEV; not convergence or terrain competence',collector='CPU sequential; existing JAX vmap cannot run PhysX; CUDA PPO',parent=identity)
(a.output/'contract.json').write_text(json.dumps(contract,indent=2))
status=dict(stage='training',completed_updates=0,accepted_updates=0,physics_steps=0,policy_transitions=0,pid=os.getpid(),started=time.time())
def save_status():
    (a.output/'status.json').write_text(json.dumps(status,indent=2))
def bundle(index):
    dest=a.output/('policy_%03d'%index);dest.mkdir()
    for name in ('config.json','route.npz'):shutil.copy2(base/name,dest/name)
    np.savez(dest/'actor.npz',**actor.export_arrays());meta=copy.deepcopy(identity)
    meta['files']['actor.npz']=hashlib.sha256((dest/'actor.npz').read_bytes()).hexdigest()
    meta['initialization_parent']=meta.pop('source_actor');meta['training_update']=index
    (dest/'identity.json').write_text(json.dumps(meta,indent=2));return dest
save_status();current=bundle(0);history=[]
for update,terrain in enumerate(contract['terrain_schedule']):
    collected=[];receipts=[];start=time.time();std=actor.log_std.exp().detach().cpu().tolist()
    for ai,alpha in enumerate(contract['alphas']):
        receipt=a.output/('u%03d_alpha%s.json'%(update,alpha));tile=root/'runs/isaac_training_terrain'/terrain
        digest=json.loads((tile/'identity.json').read_text())['sha256']
        command=[sys.executable,'learning/cli/isaac_asset_probe.py','--asset',str(root/'runs/isaac_urdf_drive/vehicle.urdf'),'--output',str(receipt),'--steps','25000','--ground','trimesh','--terrain-mesh',str(tile/'terrain.npz'),'--terrain-sha256',digest,'--controller-config',str(root/'runs/isaac_signals_parity/dynamic.json'),'--lower-actor',str(root/'runs/local300_torch_portable_parity/actor.npz'),'--lower-sha256','517f41fe825282ac4d4260a9757802e2258c644a1b235d0c328cd12e7b45f125','--path-bundle',str(current),'--alpha',str(alpha),'--contact-offset','.01','--rest-offset','0','--policy-std',str(std[0]),str(std[1]),'--seed',str(seed+update*5+ai)]
        status.update(current_update=update,alpha=alpha,terrain=terrain,receipt=str(receipt));save_status()
        with receipt.with_suffix('.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with receipt.with_suffix('.reward.log').open('w') as log:subprocess.run(['/home/qy/miniconda3/envs/mjp/bin/python','learning/cli/reconstruct_isaac_reward.py','--receipt',str(receipt),'--bundle',str(current)],env=reward_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=dict(np.load(receipt.with_suffix('.rewards.npz')));r=json.loads(receipt.read_text());status['physics_steps']+=r['physics_steps'];status['policy_transitions']+=len(data['reward']);save_status()
        collected.append(data);receipts.append(dict(alpha=alpha,termination=r['termination'],steps=r['physics_steps'],reward=float(data['reward'].sum())))
    sample_seconds=time.time()-start
    obs=torch.tensor(np.concatenate([d['observation'] for d in collected]),device='cuda',dtype=torch.float32)
    co=torch.tensor(np.concatenate([d['critic'] for d in collected]),device='cuda',dtype=torch.float32)
    action=torch.tensor(np.concatenate([d['latent'] for d in collected]),device='cuda',dtype=torch.float32)
    with torch.no_grad():
        old=actor.distribution(obs);old_mean=old.loc.clone();old_std=old.scale.clone();old_log=old.log_prob(action).sum(-1);values=critic(co).flatten();advantages=[];offset=0
        for d in collected:
            rewards=torch.tensor(d['reward'],device='cuda',dtype=torch.float32);v=values[offset:offset+len(rewards)];adv=torch.zeros_like(v);carry=0.
            # Declared finite-horizon episode: budget end and physical failure both terminate.
            for t in reversed(range(len(v))):
                next_value=v[t+1] if t+1<len(v) else 0.
                delta=rewards[t]+.997*next_value-v[t];carry=delta+.997*.95*carry;adv[t]=carry
            advantages.append(adv);offset+=len(v)
        advantages=torch.cat(advantages);returns=advantages+values;advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    before=copy.deepcopy((actor.state_dict(),critic.state_dict(),optimizer.state_dict()));start=time.time();losses=[]
    for epoch in range(5):
        for ids in torch.randperm(len(obs),device='cuda').tensor_split(4):
            dist=actor.distribution(obs[ids]);ratio=(dist.log_prob(action[ids]).sum(-1)-old_log[ids]).exp();policy=-torch.minimum(ratio*advantages[ids],ratio.clamp(.8,1.2)*advantages[ids]).mean();value=(critic(co[ids]).flatten()-returns[ids]).square().mean();entropy=dist.entropy().sum(-1).mean();loss=policy+.5*value-.01*entropy
            if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss; preserve collected evidence')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.);optimizer.step()
            with torch.no_grad():actor.log_std.clamp_(np.log(.05),np.log(.5))
            losses.append([float(policy),float(value),float(entropy)])
        with torch.no_grad():kl=float(torch.distributions.kl_divergence(torch.distributions.Normal(old_mean,old_std),actor.distribution(obs)).sum(-1).mean())
        if kl>.02:break
    accepted=np.isfinite(kl) and kl<=.02
    if not accepted:
        actor.load_state_dict(before[0]);critic.load_state_dict(before[1]);optimizer.load_state_dict(before[2])
    status['completed_updates']+=1;status['accepted_updates']+=int(accepted)
    row=dict(update=update,scored_policy=str(current),episodes=receipts,mean_step_reward=float(np.concatenate([d['reward'] for d in collected]).mean()),kl=kl,accepted=bool(accepted),loss_mean=np.mean(losses,axis=0).tolist(),sample_seconds=sample_seconds,optimize_seconds=time.time()-start)
    history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2))
    for name,val in [('reward/mean_step',row['mean_step_reward']),('ppo/kl',kl),('ppo/accepted',int(accepted)),('loss/policy',row['loss_mean'][0]),('loss/value',row['loss_mean'][1]),('sampling/failure_fraction',sum(r['termination']!='budget_end' for r in receipts)/5)]:writer.add_scalar(name,val,update)
    writer.flush();current=bundle(update+1);torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),update=update+1),a.output/'checkpoint.pt');save_status()
status.update(stage='training_complete',finished=time.time(),final_policy=str(current),final_policy_scored=False);save_status();writer.close()
subprocess.run(['notify-send','STTW 坡地训练阶段结束','有界 PPO 阶段完成，评估尚未完成'],check=False)
print(json.dumps(status))
