"""Specified phase-C geometric costs. No training is launched by this module."""
import jax.numpy as j

def huber(x):
 a=j.abs(x);return j.where(a<=1,x*x,2*a-1)

def costs(*,alpha,chi,ev,ey,eh,peak_roll,roll_rate,speed,offsets,offset_rate,config):
 c=config['path_reward'];positive=lambda x:j.maximum(x,0)
 raw=dict(speed=(10-9*chi*(1-alpha))*huber(ev/c['speed_scale_m_s']),
 path=(10-9*chi*alpha)*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad'])),
 primary=c['primary_weight']*chi*((1-alpha)*huber(positive(j.abs(ey)-c['primary_path_band_m'])/c['primary_path_scale_m'])+alpha*huber(positive(j.abs(ev)-c['primary_speed_band_m_s'])/c['primary_speed_scale_m_s'])),
 roll=c['roll_weight']*huber(positive(peak_roll-c['roll_start_rad'])/c['roll_scale_rad']),
 working_roll=c['working_roll_weight']*huber(positive(peak_roll-c['working_roll_start_rad'])/c['working_roll_scale_rad']),
 roll_rate=c['roll_rate_weight']*huber(positive(j.abs(roll_rate)-c['roll_rate_start_rad_s'])/c['roll_rate_scale_rad_s']),
 overspeed=c['overspeed_weight']*huber(positive(ev-c['overspeed_band_m_s'])/c['overspeed_scale_m_s']),
 low_speed=c['low_speed_weight']*huber(positive(c['low_speed_start_m_s']-speed)/c['low_speed_scale_m_s']),
 offset_magnitude=c['offset_magnitude_weight']*j.sum((offsets/j.array(c['offset_magnitude_scales']))**2),
 offset_rate=c['offset_rate_weight']*j.sum((offset_rate/j.array(c['offset_rate_scales']))**2))
 eff={k:j.minimum(v,c['component_caps'][k]) for k,v in raw.items()}
 return dict(raw=raw,effective=eff,capped={k:v>c['component_caps'][k] for k,v in raw.items()},reward=-c['scale']*config['timing']['lower_dt_s']*sum(eff.values()))

def failure_reward(remaining_intervals,config):
 c=config['path_reward'];gamma=config['training_future_phase_C']['gamma']
 return -c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining_intervals)/(1-gamma)
