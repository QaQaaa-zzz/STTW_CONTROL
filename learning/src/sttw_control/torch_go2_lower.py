"""Native Go2 policy behind the existing causal Local300 lower interface."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from .torch_local_lower import TorchLocalLower


class TorchGo2Lower(TorchLocalLower):
    def __init__(self, training_run, checkpoint, count, device='cuda:0'):
        training_run, checkpoint = Path(training_run), Path(checkpoint)
        config = json.loads((training_run/'config.json').read_text())
        task = config['task']
        super().__init__(task['history_actor'], task['history_actor_sha256'], count, device)
        sys.path.insert(0, str(Path(task['go2_checkout'])/'rsl_rl'))
        from rsl_rl.modules import ActorCritic
        self.model = ActorCritic(210, 211 if task.get('task_spec') else 210, 2,
                                **config['upstream']['train']['policy']).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model.load_state_dict(saved['model_state_dict'])
        self.model.eval()
        self.identity = dict(training_run=str(training_run.resolve()),
            checkpoint=str(checkpoint.resolve()),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            policy=config['upstream']['train']['policy'],
            role='frozen lower candidate; selection does not imply qualification')

    @torch.no_grad()
    def prepare(self, measurement, true_speed, governed, previous_reference,
                previous_command, controller):
        obs = self.observation(measurement, true_speed, governed,
                               previous_reference, previous_command, controller)
        action = torch.tanh(self.model.act_inference(obs))
        return action, obs, torch.isfinite(obs).all(-1) & torch.isfinite(action).all(-1)
