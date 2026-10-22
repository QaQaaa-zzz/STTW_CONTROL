"""Specified phase-C geometric costs. No training is launched by this module."""
import jax.numpy as j

def huber(x):
 a=j.abs(x);return j.where(a<=1,x*x,2*a-1)

def reward_version(config):
 """An absent version preserves the frozen historical reward identity."""
 version=config['path_reward'].get('reward_version')
 if 'reward_version' in config['path_reward'] and version not in ('path_safety_preference_v1','path_safety_preference_v2','path_persistent_preference_v3'):
  raise ValueError(f'unknown path reward version: {version!r}')
 if version=='path_safety_preference_v2':
  import math
  try:multiplier=float(config['path_reward']['primary_speed_multiplier'])
  except (KeyError,TypeError,ValueError) as e:raise ValueError('v2 requires finite positive primary speed multiplier') from e
  if not math.isfinite(multiplier) or multiplier<=0:raise ValueError('v2 requires finite positive primary speed multiplier')
 if version=='path_persistent_preference_v3' and not 0<float(config['path_reward']['path_secondary_ratio'])<=1:
  raise ValueError('path_secondary_ratio must be in (0,1]')
 return version

def costs(*,alpha,chi,ev,ey,eh,peak_roll,roll_rate,speed,offsets,offset_rate,config,roll_angle=None):
 version=reward_version(config)
 if version in ('path_safety_preference_v1','path_safety_preference_v2') and roll_angle is None:
  raise ValueError('signed roll_angle is required for path_safety_preference_v1')
 multiplier=config['path_reward']['primary_speed_multiplier'] if version=='path_safety_preference_v2' else 1.
 c=config['path_reward'];positive=lambda x:j.maximum(x,0)
 raw=dict(speed=(10-9*chi*(1-alpha))*huber(ev/c['speed_scale_m_s']),
 path=(10-9*chi*alpha)*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad'])),
 primary=c['primary_weight']*chi*((1-alpha)*huber(positive(j.abs(ey)-c['primary_path_band_m'])/c['primary_path_scale_m'])+multiplier*alpha*huber(positive(j.abs(ev)-c['primary_speed_band_m_s'])/c['primary_speed_scale_m_s'])),
 roll=c['roll_weight']*huber(positive(peak_roll-c['roll_start_rad'])/c['roll_scale_rad']),
 working_roll=c['working_roll_weight']*huber(positive(peak_roll-c['working_roll_start_rad'])/c['working_roll_scale_rad']),
 roll_rate=c['roll_rate_weight']*huber(positive(j.abs(roll_rate)-c['roll_rate_start_rad_s'])/c['roll_rate_scale_rad_s']),
 overspeed=c['overspeed_weight']*huber(positive(ev-c['overspeed_band_m_s'])/c['overspeed_scale_m_s']),
 low_speed=c['low_speed_weight']*huber(positive(c['low_speed_start_m_s']-speed)/c['low_speed_scale_m_s']),
 offset_magnitude=c['offset_magnitude_weight']*j.sum((offsets/j.array(c['offset_magnitude_scales']))**2),
 offset_rate=c['offset_rate_weight']*j.sum((offset_rate/j.array(c['offset_rate_scales']))**2))
 diagnostic={k:j.zeros_like(raw['roll']) for k in ('predictive_roll','working_exposure','corridor')}
 if version in ('path_safety_preference_v1','path_safety_preference_v2'):
  rho=j.maximum(j.abs(roll_angle),j.abs(roll_angle+c['risk_prediction_s']*roll_rate))
  diagnostic=dict(predictive_roll=c['predictive_roll_weight']*huber(positive(rho-c['predictive_roll_start_rad'])/c['predictive_roll_scale_rad']),
   working_exposure=c['working_exposure_weight']*j.clip((peak_roll-.30)/c['working_exposure_width_rad'],0,1),
   corridor=c['corridor_weight']*huber(positive(j.abs(ey)-c['corridor_band_m'])/c['corridor_scale_m']))
  raw['roll']+=diagnostic['predictive_roll']
  raw['working_roll']+=diagnostic['working_exposure']
  raw['path']+=diagnostic['corridor']
 eff={k:j.minimum(v,c['component_caps'][k]) for k,v in raw.items()}
 capped={k:v>c['component_caps'][k] for k,v in raw.items()};extra={}
 if version=='path_persistent_preference_v3':
  channels=dict(speed0=(10-9*chi)*huber(ev/c['speed_scale_m_s']),speed1=10*huber(ev/c['speed_scale_m_s']),
   path=10*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad'])),
   primary0=c['primary_weight']*chi*huber(positive(j.abs(ey)-c['primary_path_band_m'])/c['primary_path_scale_m']),
   primary1=c['primary_weight']*chi*huber(positive(j.abs(ev)-c['primary_speed_band_m_s'])/c['primary_speed_scale_m_s']))
  limits={k:c['component_caps']['speed' if k.startswith('speed') else 'primary' if k.startswith('primary') else 'path'] for k in channels}
  bounded={k:j.minimum(v,limits[k]) for k,v in channels.items()};flags={k:v>limits[k] for k,v in channels.items()}
  factor=1-(1-c['path_secondary_ratio'])*alpha
  raw.update(speed=(1-alpha)*channels['speed0']+alpha*channels['speed1'],path=factor*channels['path'],primary=(1-alpha)*channels['primary0']+alpha*channels['primary1'])
  eff.update(speed=(1-alpha)*bounded['speed0']+alpha*bounded['speed1'],path=factor*bounded['path'],primary=(1-alpha)*bounded['primary0']+alpha*bounded['primary1'])
  capped.update(speed=((alpha<1)&flags['speed0'])|((alpha>0)&flags['speed1']),path=flags['path'],primary=((alpha<1)&flags['primary0'])|((alpha>0)&flags['primary1']))
  extra=dict(endpoint_raw=channels,endpoint_effective=bounded,endpoint_capped=flags)
 return dict(raw=raw,effective=eff,capped=capped,**extra,diagnostic_subcosts=diagnostic,reward=-c['scale']*config['timing']['lower_dt_s']*sum(eff.values()))

def failure_reward(remaining_intervals,config):
 reward_version(config)
 c=config['path_reward'];gamma=config['training_future_phase_C']['gamma']
 return -c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining_intervals)/(1-gamma)
