"""One stochastic reference coordinate in the original speed/steer interface."""
import torch.nn.functional as F

CHANNEL_INDEX = {'speed':0, 'steer':1}


def expand_reference_action(actions, index):
    """Pad the executed action without altering the tensor stored by PPO."""
    return actions if index is None else F.pad(actions, (index, 1-index))
