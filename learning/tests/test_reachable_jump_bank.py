import numpy as np
import torch
from sttw_control.reachable_jump_bank import ReachableJumpBank


def test_bank_remaps_joints_and_preserves_recorded_history(tmp_path):
    values = dict(root_pose=np.array([[2., .1, .2, 1., 0., 0., 0.]], dtype=np.float32),
                  root_velocity=np.arange(6, dtype=np.float32)[None],
                  joint_position=np.array([[.3, .8]], dtype=np.float32),
                  joint_velocity=np.array([[1., 2.]], dtype=np.float32),
                  observation=np.arange(76, dtype=np.float32)[None],
                  last_action=np.array([[.1, .2, .3, .4]], dtype=np.float32))
    path = tmp_path / 'entries.npz'
    np.savez(path, **values, joint_names=['knee', 'hip'])
    bank = ReachableJumpBank(path, ['hip', 'knee'], 'cpu')
    batch = bank.sample(5)
    for key in ('root_pose', 'root_velocity', 'observation', 'last_action'):
        torch.testing.assert_close(batch[key], torch.tensor(values[key]).expand(5, -1))
    torch.testing.assert_close(batch['joint_position'], torch.tensor([[.8, .3]]).expand(5, -1))
    torch.testing.assert_close(batch['joint_velocity'], torch.tensor([[2., 1.]]).expand(5, -1))
    assert bank.sample(0)['observation'].shape == (0, 76)
