"""Train the original Phase U networks with the actual upstream Go2 PPO runner."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(Path(cfg['go2_checkout'])/'rsl_rl'))
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device=cfg['device']).app
import torch
from sttw_control.phase_u_env import PhaseUEnv
from sttw_control.phase_u_actor_critic import PhaseUActorCritic
from sttw_control.go2_training_config import spring_jump_recipe
path=Path(cfg['go2_checkout'])/'rsl_rl/rsl_rl/runners/on_policy_runner.py'
spec=importlib.util.spec_from_file_location('phase_u_go2_runner',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.PhaseUActorCritic=PhaseUActorCritic
recipe=spring_jump_recipe(cfg['go2_checkout']);train=recipe['train'];train['policy']={'bundle':cfg['bundle']};train['runner'].update(policy_class_name='PhaseUActorCritic',max_iterations=cfg['updates'],save_interval=cfg['save_interval'])
train['algorithm'].update(cfg.get('ppo_overrides',{}))
torch.manual_seed(cfg['seed']);torch.set_num_threads(2);start=time.time()
status=dict(stage='initializing',pid=os.getpid(),completed_updates=0,started=start)
(a.output/'config.json').write_text(json.dumps(dict(task=cfg,go2_recipe=recipe),indent=2));(a.output/'status.json').write_text(json.dumps(status,indent=2))
class RecordedRunner(module.OnPolicyRunner):
    def log(self,locs,width=80,pad=35):
        super().log(locs,width,pad)
        status.update(stage='training',completed_updates=locs['it']+1,stage_completed_updates=locs['it']+1-status.get('start_update',0),sample_seconds=locs['collection_time'],optimize_seconds=locs['learn_time'],elapsed_s=time.time()-start)
        (a.output/'status.json').write_text(json.dumps(status,indent=2))
env=PhaseUEnv(cfg,a.output)
(a.output/'randomized_properties.json').write_text(json.dumps(env.properties));(a.output/'plant.json').write_text(json.dumps(env.plant.metadata))
runner=RecordedRunner(env,train,log_dir=str(a.output/'training'),device=cfg['device']);(a.output/'training').mkdir(exist_ok=True)
initialization={'initialization':'original Phase U actor/critic/frozen normalizer; fresh optimizer'}
if cfg.get('resume_checkpoint'):
    runner.load(cfg['resume_checkpoint'],load_optimizer=True)
    runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]['lr']
    initialization.update(initialization='checkpoint model and optimizer resume; fresh simulation and rollout state',resume_checkpoint=cfg['resume_checkpoint'],start_update=runner.current_learning_iteration,learning_rate=runner.alg.learning_rate)
elif cfg.get('initial_checkpoint'):
    checkpoint=torch.load(cfg['initial_checkpoint'],map_location=cfg['device'],weights_only=False)
    runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    initialization.update(initial_checkpoint=cfg['initial_checkpoint'],parent_selection=checkpoint.get('infos'),initialization='selected prior actor/critic/normalizer weights; fresh optimizer and rollout state')
status.update(start_update=runner.current_learning_iteration,completed_updates=runner.current_learning_iteration,target_update=runner.current_learning_iteration+cfg['updates'])
(a.output/'status.json').write_text(json.dumps(status,indent=2))
runner.save(str(a.output/'training/model_initial.pt'),infos=initialization)
upstream_update=runner.alg.update;selection=dict(update=runner.current_learning_iteration,best=-float('inf'))
def recorded_update():
    score=float(runner.alg.storage.rewards.mean());update=selection['update']
    row=dict(update=update,mean_step_reward=score,samples=cfg['num_envs']*train['runner']['num_steps_per_env'],reward_scaling=env.task['ppo']['reward_scaling'])
    with (a.output/'training_rewards.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    runner.writer.add_scalar('Train/mean_step_reward_pre_update',score,update)
    if score>selection['best']:
        selection['best']=score
        torch.save(dict(model_state_dict=runner.alg.actor_critic.state_dict(),optimizer_state_dict=runner.alg.optimizer.state_dict(),iter=update,infos=row),a.output/'training/model_step_best.pt')
        (a.output/'best_step_model.json').write_text(json.dumps(dict(**row,checkpoint='training/model_step_best.pt',scope='best sampled pre-update policy in this run; not fixed-case qualification'),indent=2))
    result=upstream_update();selection['update']+=1
    return result
runner.alg.update=recorded_update
runner.learn(cfg['updates'],init_at_random_ep_len=False)
status.update(stage='training_complete',elapsed_s=time.time()-start,total_physics_steps=env.total_physics_steps)
(a.output/'status.json').write_text(json.dumps(status,indent=2));runner.writer.close();env.plant.close();app.close(wait_for_replicator=False,skip_cleanup=True)
