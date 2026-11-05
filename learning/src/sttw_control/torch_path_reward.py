"""GPU translation of frozen path_persistent_preference_v3, checked against JAX."""
import torch


def costs(*,alpha,chi,ev,ey,eh,peak_roll,roll_rate,speed,offsets,offset_rate,config):
    c=config['path_reward']
    if c['reward_version']!='path_persistent_preference_v3':raise ValueError('GPU path reward requires frozen v3')
    def huber(x):return torch.where(x.abs()<=1,x.square(),2*x.abs()-1)
    def positive(x):return x.clamp_min(0)
    def scale(name):return torch.as_tensor(c[name],device=offsets.device,dtype=offsets.dtype)
    channels=dict(speed0=(10-9*chi)*huber(ev/c['speed_scale_m_s']),speed1=10*huber(ev/c['speed_scale_m_s']),
        path=10*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad'])),
        primary0=c['primary_weight']*chi*huber(positive(ey.abs()-c['primary_path_band_m'])/c['primary_path_scale_m']),
        primary1=c['primary_weight']*chi*huber(positive(ev.abs()-c['primary_speed_band_m_s'])/c['primary_speed_scale_m_s']))
    bounds={k:v.clamp_max(c['component_caps']['speed' if k.startswith('speed') else 'primary' if k.startswith('primary') else 'path']) for k,v in channels.items()}
    factor=1-(1-c['path_secondary_ratio'])*alpha
    raw=dict(speed=(1-alpha)*channels['speed0']+alpha*channels['speed1'],path=factor*channels['path'],primary=(1-alpha)*channels['primary0']+alpha*channels['primary1'],
        roll=c['roll_weight']*huber(positive(peak_roll-c['roll_start_rad'])/c['roll_scale_rad']),
        working_roll=c['working_roll_weight']*huber(positive(peak_roll-c['working_roll_start_rad'])/c['working_roll_scale_rad']),
        roll_rate=c['roll_rate_weight']*huber(positive(roll_rate.abs()-c['roll_rate_start_rad_s'])/c['roll_rate_scale_rad_s']),
        overspeed=c['overspeed_weight']*huber(positive(ev-c['overspeed_band_m_s'])/c['overspeed_scale_m_s']),
        low_speed=c['low_speed_weight']*huber(positive(c['low_speed_start_m_s']-speed)/c['low_speed_scale_m_s']),
        offset_magnitude=c['offset_magnitude_weight']*((offsets/scale('offset_magnitude_scales'))**2).sum(-1),
        offset_rate=c['offset_rate_weight']*((offset_rate/scale('offset_rate_scales'))**2).sum(-1))
    effective={k:v.clamp_max(c['component_caps'][k]) for k,v in raw.items()}
    effective.update(speed=(1-alpha)*bounds['speed0']+alpha*bounds['speed1'],path=factor*bounds['path'],primary=(1-alpha)*bounds['primary0']+alpha*bounds['primary1'])
    return dict(raw=raw,effective=effective,reward=-c['scale']*config['timing']['lower_dt_s']*sum(effective.values()))


def failure_reward(remaining_intervals,config):
    c=config['path_reward'];gamma=config['training_future_phase_C']['gamma']
    return -c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining_intervals)/(1-gamma)
