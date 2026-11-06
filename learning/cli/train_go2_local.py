"""Train the STTW lower task using Go2's native runner, network and PPO."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import importlib.util

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
s = json.loads(a.config.read_text())
a.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(Path(s['go2_checkout'])/'rsl_rl'))
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device=s['device']).app
import torch
# Import the actual upstream runner without importing unrelated AMP runners
# from the package initializer (those require quadruped motion/pybullet tools).
runner_path = Path(s['go2_checkout'])/'rsl_rl/rsl_rl/runners/on_policy_runner.py'
runner_spec = importlib.util.spec_from_file_location('sttw_upstream_go2_runner', runner_path)
runner_module = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner_module)
OnPolicyRunner = runner_module.OnPolicyRunner
from sttw_control.go2_training_config import spring_jump_recipe
from sttw_control.go2_local_env import Go2LocalEnv

torch.manual_seed(s['seed'])
torch.set_num_threads(2)
recipe = spring_jump_recipe(s['go2_checkout'])
train = recipe['train']
train['algorithm'].update(s.get('ppo_overrides', {}))
train['policy'].update(s.get('policy_overrides', {}))
train['runner'].update(max_iterations=s['updates'], save_interval=s['save_interval'], experiment_name='sttw_local')
(a.output/'config.json').write_text(json.dumps(dict(task=s, upstream=recipe), indent=2))
started = time.time()
status = dict(stage='initializing', pid=os.getpid(), completed_updates=0, started=started)
(a.output/'status.json').write_text(json.dumps(status, indent=2))


class RecordedRunner(OnPolicyRunner):
    """Add a status receipt; sampling, optimization and checkpointing stay upstream."""
    best_episode_reward = -float('inf')

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        status.update(stage='training', completed_updates=locs['it']+1,
                      sample_seconds=locs['collection_time'], optimize_seconds=locs['learn_time'],
                      elapsed_s=time.time()-started)
        (a.output/'status.json').write_text(json.dumps(status, indent=2))
        if locs['ep_infos']:
            score = float(torch.stack([x['reward'].reshape(()) for x in locs['ep_infos']]).mean())
            row = dict(completed_updates=locs['it']+1, episode_reward=score,
                       completed_episode_batches=len(locs['ep_infos']),
                       scope='candidate at completed-episode reward peak; episodes span policy updates')
            with (a.output/'training_episode_rewards.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if score > self.best_episode_reward:
                self.best_episode_reward = score
                torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),
                                optimizer_state_dict=self.alg.optimizer.state_dict(),
                                iter=locs['it']+1, infos=row), a.output/'training/model_best.pt')
                (a.output/'best_model.json').write_text(json.dumps(dict(**row,
                    checkpoint='training/model_best.pt'), indent=2))



env = Go2LocalEnv(s, a.output)
(a.output/'randomized_properties.json').write_text(json.dumps(env.properties))
(a.output/'plant.json').write_text(json.dumps(env.plant.metadata))
runner = RecordedRunner(env, train, log_dir=str(a.output/'training'), device=s['device'])
if s.get('zero_actor_output_init', False):
    torch.nn.init.zeros_(runner.alg.actor_critic.actor[-1].weight)
    torch.nn.init.zeros_(runner.alg.actor_critic.actor[-1].bias)
if s.get('initialize_from_local_actor', False):
    layers = [layer for layer in runner.alg.actor_critic.actor.modules()
              if isinstance(layer, torch.nn.Linear)]
    assert len(layers) == len(env.history.weights), 'native actor must match frozen Local300 layers'
    with torch.no_grad():
        for layer, (weight, bias) in zip(layers, env.history.weights):
            layer.weight.copy_(weight.T)
            layer.bias.copy_(bias)
(a.output/'training').mkdir(exist_ok=True)
runner.save(str(a.output/'training/model_initial.pt'),
            infos=dict(completed_updates=0, initialization=s.get('initialization','fresh actor/critic/optimizer')))
# Observe the native PPO update boundary to score the model that actually
# collected this rollout. Sampling, losses and optimizer remain upstream.
upstream_update = runner.alg.update
selection = dict(completed_updates=0, best_reward=-float('inf'))
def recorded_update():
    mean_reward = float(runner.alg.storage.rewards.mean())
    update = selection['completed_updates']
    row = dict(completed_updates=update, mean_step_reward=mean_reward,
               samples=s['num_envs']*train['runner']['num_steps_per_env'])
    with (a.output/'training_rewards.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    runner.writer.add_scalar('Train/mean_step_reward_pre_update', mean_reward, update)
    if mean_reward > selection['best_reward']:
        selection['best_reward'] = mean_reward
        torch.save(dict(model_state_dict=runner.alg.actor_critic.state_dict(),
                        optimizer_state_dict=runner.alg.optimizer.state_dict(),
                        iter=update, infos=row), a.output/'training/model_step_best.pt')
        (a.output/'best_step_model.json').write_text(json.dumps(dict(**row,
            checkpoint='training/model_step_best.pt',
            scope='best sampled pre-update model in this run; not fixed-case qualification'), indent=2))
    result = upstream_update()
    selection['completed_updates'] += 1
    return result
runner.alg.update = recorded_update
runner.learn(s['updates'], init_at_random_ep_len=False)
status.update(stage='training_complete', elapsed_s=time.time()-started,total_physics_steps=env.total_steps*env.decimation*env.num_envs)
(a.output/'status.json').write_text(json.dumps(status, indent=2))
runner.writer.close()
os.sync();os._exit(0)
