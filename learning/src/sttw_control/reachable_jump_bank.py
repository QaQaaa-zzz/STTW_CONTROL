"""Recorded pre-jump states for optional Phase U reset sampling."""
import numpy as np
import torch


class ReachableJumpBank:
    def __init__(self, path, joint_names, device):
        with np.load(path) as data:
            order = [data['joint_names'].tolist().index(name) for name in joint_names]
            self.values = {
                key: torch.as_tensor(data[key].copy(), device=device)
                for key in ('root_pose', 'root_velocity', 'joint_position',
                            'joint_velocity', 'observation', 'last_action')
            }
        for key in ('joint_position', 'joint_velocity'):
            self.values[key] = self.values[key][:, order]
        self.count = len(self.values['root_pose'])
        self.device = device

    def sample(self, count):
        indices = torch.randint(self.count, (count,), device=self.device)
        return {key: value[indices] for key, value in self.values.items()}
