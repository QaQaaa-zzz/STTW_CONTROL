"""Specified phase-C geometric costs. No training is launched by this module."""
import jax.numpy as j
from .tolerance_shaping import tolerance_enabled

BASE_COST_KEYS=('speed','path','primary','roll','working_roll','roll_rate','overspeed','low_speed','offset_magnitude','offset_rate')

def configured_cost_keys(config):
 return BASE_COST_KEYS+(('tolerance',) if tolerance_enabled(config) else ())

def huber(x):
 a=j.abs(x);return j.where(a<=1,x*x,2*a-1)

def costs(*,alpha,chi,ev,ey,eh,peak_roll,roll_rate,speed,offsets,offset_rate,config,tolerance_cost=None):
 c=config['path_reward'];version=c.get('reward_version');positive=lambda x:j.maximum(x,0)
 if version not in (None,'path_persistent_preference_v3'):raise ValueError('unsupported reward version')
 if version=='path_persistent_preference_v3' and not 0<float(c['path_secondary_ratio'])<=1:raise ValueError('invalid secondary path ratio')
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
 if tolerance_enabled(config):
  if tolerance_cost is None:raise ValueError('enabled tolerance requires computed cost')
  raw['tolerance']=tolerance_cost
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
 return dict(raw=raw,effective=eff,capped=capped,**extra,reward=-c['scale']*config['timing']['lower_dt_s']*sum(eff.values()))

def failure_reward(remaining_intervals,config):
 c=config['path_reward'];gamma=config['training_future_phase_C']['gamma']
 return -c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining_intervals)/(1-gamma)
