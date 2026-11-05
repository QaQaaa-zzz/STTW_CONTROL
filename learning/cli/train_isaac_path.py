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
p.add_argument('--config',type=Path,required=True)
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
root=Path.cwd();settings=json.loads(a.config.read_text());previous=root/settings['resume_training'];prior_status=json.loads((previous/'status.json').read_text())
if prior_status['stage']!='training_complete' or settings['updates']<=0 or settings['episode_s']!=5 or settings['alphas']!=[0,.25,.5,.75,1]:raise ValueError('invalid bounded continuation configuration')
base=Path(prior_status['final_policy']);identity=json.loads((base/'identity.json').read_text());cfg=json.loads((base/'config.json').read_text())
seed=settings['seed'];torch.manual_seed(seed);torch.set_num_threads(2)
actor=TrainablePathActor(TorchPathPolicy(base/'actor.npz',identity['files']['actor.npz'],cfg,1,'cuda'))
critic=torch.nn.Sequential(torch.nn.Linear(352,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,1)).cuda()
optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=settings['lr'])
checkpoint=torch.load(previous/'checkpoint.pt',map_location='cuda');actor.load_state_dict(checkpoint['actor']);critic.load_state_dict(checkpoint['critic']);optimizer.load_state_dict(checkpoint['optimizer'])
for group in optimizer.param_groups:group['lr']=settings['lr']
writer=SummaryWriter(str(a.output/'tb'));writer.add_scalar('stage/started',1,0);writer.flush()
env=os.environ.copy();env['PATH']='/home/qy/miniconda3/envs/go2_isaacgym/bin:'+env['PATH'];env['LD_LIBRARY_PATH']='/home/qy/miniconda3/envs/go2_isaacgym/lib';env['PYTHONPATH']=str(root/'learning/src')
reward_env=env.copy();reward_env['JAX_PLATFORMS']='cpu';reward_env.pop('LD_LIBRARY_PATH',None)
contract=dict(settings,initialization='Actor/Critic/Adam continuation; fresh declared RNG because previous checkpoint has no RNG state',max_physics_steps=settings['updates']*len(settings['terrains'])*len(settings['alphas'])*25000,max_policy_transitions=settings['updates']*len(settings['terrains'])*len(settings['alphas'])*250,physics_dt=.0002,lower_dt=.005,middle_dt=.02,role='bounded mixed terrain continuation DEV; no convergence claim',collector='CPU sequential PhysX; CUDA PPO',parent=identity,parent_checkpoint_sha256=hashlib.sha256((previous/'checkpoint.pt').read_bytes()).hexdigest())
(a.output/'contract.json').write_text(json.dumps(contract,indent=2))
status=dict(stage='training',completed_updates=0,accepted_updates=0,physics_steps=0,policy_transitions=0,pid=os.getpid(),started=time.time())
def save_status():
    (a.output/'status.json').write_text(json.dumps(status,indent=2))
def bundle(index):
    dest=a.output/('policy_%03d'%index);dest.mkdir()
    for name in ('config.json','route.npz'):shutil.copy2(base/name,dest/name)
    np.savez(dest/'actor.npz',**actor.export_arrays());meta=copy.deepcopy(identity)
    meta['files']['actor.npz']=hashlib.sha256((dest/'actor.npz').read_bytes()).hexdigest()
    meta['initialization_parent']=meta.pop('source_actor',meta.get('initialization_parent'));meta['training_update']=index
    (dest/'identity.json').write_text(json.dumps(meta,indent=2));return dest
save_status();current=bundle(0);history=[]
for update in range(contract['updates']):
    collected=[];receipts=[];start=time.time();std=actor.log_std.exp().detach().cpu().tolist()
    for ai,(terrain,alpha) in enumerate((terrain,alpha) for terrain in contract['terrains'] for alpha in contract['alphas']):
        receipt=a.output/('u%03d_%s_alpha%s.json'%(update,terrain,alpha));tile=root/contract['terrains'][terrain]
        digest=json.loads((tile/'identity.json').read_text())['sha256']
        command=[sys.executable,'learning/cli/isaac_asset_probe.py','--asset',str(root/'runs/isaac_urdf_drive/vehicle.urdf'),'--output',str(receipt),'--steps','25000','--ground','trimesh','--terrain-mesh',str(tile/'terrain.npz'),'--terrain-sha256',digest,'--controller-config',str(root/'runs/isaac_signals_parity/dynamic.json'),'--lower-actor',str(root/'runs/local300_torch_portable_parity/actor.npz'),'--lower-sha256','517f41fe825282ac4d4260a9757802e2258c644a1b235d0c328cd12e7b45f125','--path-bundle',str(current),'--alpha',str(alpha),'--contact-offset','.01','--rest-offset','0','--policy-std',str(std[0]),str(std[1]),'--seed',str(seed+update*len(contract['terrains'])*len(contract['alphas'])+ai)]
        status.update(current_update=update,alpha=alpha,terrain=terrain,receipt=str(receipt));save_status()
        with receipt.with_suffix('.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with receipt.with_suffix('.reward.log').open('w') as log:subprocess.run(['/home/qy/miniconda3/envs/mjp/bin/python','learning/cli/reconstruct_isaac_reward.py','--receipt',str(receipt),'--bundle',str(current)],env=reward_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=dict(np.load(receipt.with_suffix('.rewards.npz')));r=json.loads(receipt.read_text());status['physics_steps']+=r['physics_steps'];status['policy_transitions']+=len(data['reward']);save_status()
        collected.append(data);receipts.append(dict(terrain=terrain,receipt=str(receipt),alpha=alpha,termination=r['termination'],steps=r['physics_steps'],reward=float(data['reward'].sum())))
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
                delta=rewards[t]+contract['gamma']*next_value-v[t];carry=delta+contract['gamma']*contract['lam']*carry;adv[t]=carry
            advantages.append(adv);offset+=len(v)
        advantages=torch.cat(advantages);returns=advantages+values;advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    before=copy.deepcopy((actor.state_dict(),critic.state_dict(),optimizer.state_dict()));start=time.time();losses=[]
    for epoch in range(contract['epochs']):
        for ids in torch.randperm(len(obs),device='cuda').tensor_split(contract['minibatches']):
            dist=actor.distribution(obs[ids]);ratio=(dist.log_prob(action[ids]).sum(-1)-old_log[ids]).exp();policy=-torch.minimum(ratio*advantages[ids],ratio.clamp(1-contract['clip'],1+contract['clip'])*advantages[ids]).mean();value=(critic(co[ids]).flatten()-returns[ids]).square().mean();entropy=dist.entropy().sum(-1).mean();loss=policy+.5*value-contract['entropy']*entropy
            if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss; preserve collected evidence')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),contract['max_grad_norm'],error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():actor.log_std.clamp_(np.log(contract['std_bounds'][0]),np.log(contract['std_bounds'][1]))
            losses.append([float(policy),float(value),float(entropy)])
        with torch.no_grad():kl=float(torch.distributions.kl_divergence(torch.distributions.Normal(old_mean,old_std),actor.distribution(obs)).sum(-1).mean())
        if kl>contract['hard_kl']:break
    accepted=np.isfinite(kl) and kl<=contract['hard_kl']
    if not accepted:
        actor.load_state_dict(before[0]);critic.load_state_dict(before[1]);optimizer.load_state_dict(before[2])
    status['completed_updates']+=1;status['accepted_updates']+=int(accepted)
    row=dict(update=update,scored_policy=str(current),episodes=receipts,mean_step_reward=float(np.concatenate([d['reward'] for d in collected]).mean()),kl=kl,accepted=bool(accepted),loss_mean=np.mean(losses,axis=0).tolist(),sample_seconds=sample_seconds,optimize_seconds=time.time()-start)
    history.append(row);best=max(history,key=lambda item:item['mean_step_reward']);(a.output/'best_training.json').write_text(json.dumps(dict(policy=best['scored_policy'],mean_step_reward=best['mean_step_reward'],selection='pre-update sampled reward within this mixed-terrain stage; not fixed evaluation best'),indent=2));(a.output/'history.json').write_text(json.dumps(history,indent=2))
    for name,val in [('reward/mean_step',row['mean_step_reward']),('ppo/kl',kl),('ppo/accepted',int(accepted)),('loss/policy',row['loss_mean'][0]),('loss/value',row['loss_mean'][1]),('sampling/failure_fraction',sum(r['termination']!='budget_end' for r in receipts)/len(receipts))]:writer.add_scalar(name,val,update)
    writer.flush();current=bundle(update+1);torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),update=update+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),a.output/'checkpoint.pt');save_status()
status.update(stage='training_complete',finished=time.time(),final_policy=str(current),final_policy_scored=False);save_status();writer.close()
subprocess.run(['notify-send','STTW 坡地训练阶段结束','有界 PPO 阶段完成，评估尚未完成'],check=False)
print(json.dumps(status))
