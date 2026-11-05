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
train['runner'].update(max_iterations=s['updates'], save_interval=s['save_interval'], experiment_name='sttw_local')
(a.output/'config.json').write_text(json.dumps(dict(task=s, upstream=recipe), indent=2))
started = time.time()
status = dict(stage='initializing', pid=os.getpid(), completed_updates=0, started=started)
(a.output/'status.json').write_text(json.dumps(status, indent=2))


class RecordedRunner(OnPolicyRunner):
    """Add a status receipt; sampling, optimization and checkpointing stay upstream."""
    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        status.update(stage='training', completed_updates=locs['it']+1,
                      sample_seconds=locs['collection_time'], optimize_seconds=locs['learn_time'],
                      elapsed_s=time.time()-started)
        (a.output/'status.json').write_text(json.dumps(status, indent=2))


env = Go2LocalEnv(s, a.output)
(a.output/'randomized_properties.json').write_text(json.dumps(env.properties))
runner = RecordedRunner(env, train, log_dir=str(a.output/'training'), device=s['device'])
runner.learn(s['updates'], init_at_random_ep_len=False)
status.update(stage='training_complete', elapsed_s=time.time()-started)
(a.output/'status.json').write_text(json.dumps(status, indent=2))
runner.writer.close()
env.plant.close()
app.close(wait_for_replicator=False)
