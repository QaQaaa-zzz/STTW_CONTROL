"""Optional bounded heading feedback for the straight platform approach.

Only grounded steering is corrected. The frozen actor, arm and rear-wheel
outputs remain unchanged; this is a controller change, not an RL improvement.
"""
import torch


def correct_grounded_jump_heading(target, pose, contact, wheel_ids, config):
    yaw=pose[:,2]-config.get('reference_yaw_rad',0.)
    yaw=torch.atan2(torch.sin(yaw),torch.cos(yaw))
    gain=torch.as_tensor(config['yaw_gain'],device=target.device,dtype=target.dtype)
    correction=(-gain*yaw).clamp(-config['max_correction_rad'],config['max_correction_rad'])
    grounded=(contact[:,wheel_ids,2]>1.).all(-1)
    corrected=target.clone()
    corrected[:,0]=(target[:,0]+torch.where(grounded,correction,0.)).clamp(*config['steering_limits_rad'])
    return corrected


def heading_features(root):
    """Straight platform-frame localization: y, sin(yaw), cos(yaw)-1, world vy."""
    x,y,z,w=root[:,3:7].unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return torch.stack((root[:,1],torch.sin(yaw),torch.cos(yaw)-1,root[:,8]),-1)


def expand_actor_state(state):
    """Preserve the 76-input parent exactly at initialization; add four zero columns."""
    state=dict(state)
    state['actor.0.weight']=torch.nn.functional.pad(state['actor.0.weight'],(0,4))
    state['mean']=torch.cat((state['mean'],state['mean'].new_zeros(4)))
    state['std']=torch.cat((state['std'],state['std'].new_ones(4)))
    return state
