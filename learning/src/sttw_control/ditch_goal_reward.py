"""Goal-directed shaping for same-height ditch traversal; weights are STTW design choices."""
import torch


def ditch_goal_components(state, config):
    dx=config['goal_x_m']-state['x'];dy=config['goal_y_m']-state['y']
    distance=torch.sqrt(dx.square()+dy.square())
    direction_x=dx/distance.clamp_min(1e-6);direction_y=dy/distance.clamp_min(1e-6)
    desired=config['target_speed_m_s']
    velocity_error=(state['forward_velocity']-desired*direction_x).square()+(state['lateral_velocity']-desired*direction_y).square()
    return dict(ditch_goal_position=config['position_weight']*torch.exp(-distance.square()/config['position_scale_m']**2),
                ditch_goal_velocity=config['velocity_weight']*torch.exp(-velocity_error/config['velocity_scale_m_s']**2))
