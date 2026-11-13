"""Native Run3 weights + existing Go2 PPO on the real Playground trench."""
import argparse,json,os,sys,time,shutil
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check-steps',type=int,default=0);p.add_argument('--num-envs',type=int);a=p.parse_args()
cfg=json.loads(a.config.read_text())
if a.num_envs is not None:cfg['num_envs']=a.num_envs
if cfg['num_envs']<=0 or cfg['updates']<1:raise ValueError('Positive environment and update counts required')
a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();status=dict(stage='initializing',pid=os.getpid(),completed_updates=0,target_updates=cfg['updates'],integration_only=bool(a.check_steps))
def save_status():
    status['elapsed_s']=time.perf_counter()-start
    (a.output/'status.json').write_text(json.dumps(status,indent=2))
def failure(kind,error,trace):
    status.update(stage='failed',error=str(error));save_status();sys.__excepthook__(kind,error,trace)
sys.excepthook=failure
save_status();(a.output/'config.json').write_text(json.dumps(cfg,indent=2))
source=a.output/'source';source.mkdir()
for name in [Path(__file__),Path('learning/src/sttw_control/playground_rb_training.py'),Path('learning/src/sttw_control/phase_u_actor_critic.py')]:shutil.copy2(name,source/name.name)
sys.path.insert(0,str(Path(cfg['go2_checkout'])/'rsl_rl'))
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.algorithms import PPO
from sttw_control.phase_u_actor_critic import PhaseUActorCritic
from sttw_control.go2_training_config import spring_jump_recipe
from sttw_control.pendulum_skill import PendulumSkill
from sttw_control.playground_rb_training import PlaygroundRBEnv
from sttw_control.runtime import configure_compilation_cache
from evaluate_playground_handoff import transplant
configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
skill=PendulumSkill(Path(cfg['registry']),cfg['skill'])
model=transplant(skill.bundle,cfg['terrain'],a.output,1,cfg['ground_physics'])
env=PlaygroundRBEnv(model,skill,cfg)
actor=PhaseUActorCritic(76,106,4,cfg['bundle']).to('cuda:0')
recipe=spring_jump_recipe(cfg['go2_checkout']);algorithm=recipe['train']['algorithm'];algorithm.update(cfg['ppo_overrides'])
(a.output/'go2_recipe.json').write_text(json.dumps(recipe,indent=2))
(a.output/'randomization.json').write_text(json.dumps(dict(config=cfg['randomization'],actual='MuJoCo GPU startup properties; upstream Isaac Lab DelayBuffer;5ms sensor/action delay; noise and offsets only in observations;4s velocity pushes; raw clean reward',physics_dt_s=.005,actor_dt_s=.02),indent=2))
if a.check_steps:
    env.length.fill_(env.max_episode_length-1)
    with torch.inference_mode():
        for _ in range(a.check_steps):
            obs,critic,reward,done,info=env.step(actor.act_inference(env.obs))
    torch.cuda.synchronize()
    finite=bool(torch.isfinite(obs).all()&torch.isfinite(critic).all()&torch.isfinite(reward).all())
    status.update(stage='integration_complete',finite=finite,steps=a.check_steps,physics_steps=a.check_steps*4*cfg['num_envs'],optimizer_updates=0,stats=env.stats.tolist());save_status()
    if not finite:raise RuntimeError('Nonfinite integration output')
    print(json.dumps(status));sys.exit(0)
ppo=PPO(actor,device='cuda:0',**algorithm)
ppo.init_storage(cfg['num_envs'],cfg['steps_per_update'],[76],[106],[4])
writer=SummaryWriter(str(a.output/'training'));training=a.output/'training';training.mkdir(exist_ok=True)
def checkpoint(path,iteration,info):
    torch.save(dict(model_state_dict=actor.state_dict(),optimizer_state_dict=ppo.optimizer.state_dict(),iter=iteration,infos=info),path)
checkpoint(training/'model_initial.pt',0,dict(initialization='Run3 actor/critic/normalizer weights only; fresh Go2 PPO optimizer'))
best=-float('inf');obs,critic=env.obs,env.critic
for update in range(cfg['updates']):
    tick=time.perf_counter();env.stats.zero_()
    with torch.inference_mode():
        for _ in range(cfg['steps_per_update']):
            action=ppo.act(obs,critic)
            obs,critic,reward,done,info=env.step(action)
            ppo.process_env_step(reward,done,info)
        ppo.compute_returns(critic)
    torch.cuda.synchronize();sample_s=time.perf_counter()-tick;score=float(ppo.storage.rewards.mean())
    if score>best:
        best=score;checkpoint(training/'model_step_best.pt',update,dict(update=update,mean_step_reward=score,scope='sampled pre-update training reward; not fixed-panel qualification'))
        (a.output/'best_step_model.json').write_text(json.dumps(dict(update=update,mean_step_reward=score,checkpoint='training/model_step_best.pt'),indent=2))
    tick=time.perf_counter();loss=ppo.update();torch.cuda.synchronize();opt_s=time.perf_counter()-tick
    with torch.inference_mode():
        actor.update_distribution(ppo.storage.observations[0])
        old_mu,old_sigma=ppo.storage.mu[0],ppo.storage.sigma[0]
        kl=torch.sum(torch.log(actor.action_std/old_sigma)+(old_sigma.square()+(old_mu-actor.action_mean).square())/(2*actor.action_std.square())-.5,-1).mean()
    record=dict(update=update+1,scored_policy_update=update,mean_step_reward=score,sample_s=sample_s,optimize_s=opt_s,loss_value=loss[0],loss_surrogate=loss[1],entropy=loss[3],kl_first_rollout_step=float(kl),learning_rate=ppo.learning_rate,terminal_counts=env.stats.tolist())
    with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    for key,value in record.items():
        if isinstance(value,(int,float)):writer.add_scalar(key,value,update+1)
    writer.flush();status.update(stage='training',completed_updates=update+1,last=record,estimated_remaining_s=(sample_s+opt_s)*(cfg['updates']-update-1));save_status();print(json.dumps(record),flush=True)
    if not np.isfinite([score,*loss,float(kl)]).all():raise RuntimeError('Nonfinite training statistics')
    if (update+1)%cfg['save_interval']==0:checkpoint(training/f'model_{update+1}.pt',update+1,record)
checkpoint(training/f'model_{cfg["updates"]}.pt',cfg['updates'],dict(final_update_unscored=True))
status.update(stage='training_complete');save_status();writer.close()
