"""Specified phase-C geometric costs. No training is launched by this module."""
import jax.numpy as j

def huber(x):
 a=j.abs(x);return j.where(a<=1,x*x,2*a-1)

def reward_version(config):
 """An absent version preserves the frozen historical reward identity."""
 version=config['path_reward'].get('reward_version')
 if 'reward_version' in config['path_reward'] and version!='path_safety_preference_v1':
  raise ValueError(f'unknown path reward version: {version!r}')
 return version

def costs(*,alpha,chi,ev,ey,eh,peak_roll,roll_rate,speed,offsets,offset_rate,config,roll_angle=None):
 version=reward_version(config)
 if version=='path_safety_preference_v1' and roll_angle is None:
  raise ValueError('signed roll_angle is required for path_safety_preference_v1')
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
 diagnostic={k:j.zeros_like(raw['roll']) for k in ('predictive_roll','working_exposure','corridor')}
 if version=='path_safety_preference_v1':
  rho=j.maximum(j.abs(roll_angle),j.abs(roll_angle+c['risk_prediction_s']*roll_rate))
  diagnostic=dict(predictive_roll=c['predictive_roll_weight']*huber(positive(rho-c['predictive_roll_start_rad'])/c['predictive_roll_scale_rad']),
   working_exposure=c['working_exposure_weight']*j.clip((peak_roll-.30)/c['working_exposure_width_rad'],0,1),
   corridor=c['corridor_weight']*huber(positive(j.abs(ey)-c['corridor_band_m'])/c['corridor_scale_m']))
  raw['roll']+=diagnostic['predictive_roll']
  raw['working_roll']+=diagnostic['working_exposure']
  raw['path']+=diagnostic['corridor']
 eff={k:j.minimum(v,c['component_caps'][k]) for k,v in raw.items()}
 return dict(raw=raw,effective=eff,capped={k:v>c['component_caps'][k] for k,v in raw.items()},diagnostic_subcosts=diagnostic,reward=-c['scale']*config['timing']['lower_dt_s']*sum(eff.values()))

def failure_reward(remaining_intervals,config):
 reward_version(config)
 c=config['path_reward'];gamma=config['training_future_phase_C']['gamma']
 return -c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining_intervals)/(1-gamma)
