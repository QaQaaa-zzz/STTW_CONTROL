import copy,json
from pathlib import Path
from dataclasses import asdict
import jax.numpy as j
import numpy as np
import pytest
from sttw_control.path_command_reward import costs,failure_reward,configured_cost_keys
from sttw_control.tolerance_shaping import validate_tolerance_config,measurement_reserve,model_reserve
from sttw_control.tolerance_state_features import unstable_features
from sttw_control.controller import ControllerConfig,initial_controller

BASE=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Coverage_V2.json').read_text())
def configured(weight=10.):
 c=copy.deepcopy(BASE);c['path_reward']['tolerance']=dict(method='dynamic',weight=weight,gamma=.815560907125473,effort_budget=.030839085298575167,duration=.1,target_force=6.)
 if weight:
  c['path_reward']['component_caps']['tolerance']=weight;c['path_reward']['failure_cost_rate_bound']+=weight
 return c

def reward(c,extra=None):
 return costs(alpha=0.,chi=.5,ev=.2,ey=.1,eh=.03,peak_roll=.2,roll_rate=.3,speed=2.3,offsets=j.zeros(2),offset_rate=j.zeros(2),config=c,tolerance_cost=extra)

def test_disabled_cost_preserves_exact_reward_and_schema():
 a=reward(BASE);b=reward(configured(0),j.nan)
 assert float(a['reward'])==float(b['reward']);assert configured_cost_keys(BASE)==configured_cost_keys(configured(0))

def test_new_component_and_failure_tail_are_consistent():
 c=configured();validate_tolerance_config(c);a=reward(BASE);b=reward(c,7.)
 np.testing.assert_allclose(float(b['reward']-a['reward']),-.1*.005*7,rtol=1e-5)
 assert len(configured_cost_keys(c))==11 and b['effective']['tolerance']==7
 gamma=c['training_future_phase_C']['gamma'];expected=-.1*.02*10*(1-gamma**20)/(1-gamma)
 np.testing.assert_allclose(float(failure_reward(20,c)-failure_reward(20,BASE)),expected,rtol=1e-5)
 c['path_reward']['failure_cost_rate_bound']-=10
 with pytest.raises(ValueError,match='failure'):validate_tolerance_config(c)

def test_live_measurement_uses_controller_wheel_proxy_and_eso():
 c=configured();cc=ControllerConfig();state=initial_controller(cc).replace(disturbance=j.array(.1));m=j.array([.1,.2,.03,.04,.05,23.,22.])
 x=measurement_reserve(m,state,cc,remaining=2.,config=c)
 f=unstable_features(speed_proxy=2.3,steer=.03,roll=.1,roll_rate=.2,disturbance=.1,controller=asdict(cc))
 spec=c['path_reward']['tolerance'];ref=model_reserve(z=f['z_u'],b=f['b_u'],lam=f['lambda_u'],remaining=2.,**{k:v for k,v in spec.items() if k!='method'})
 for key in ['plus_N','minus_N','cost_rate']:np.testing.assert_allclose(x[key],ref[key],rtol=2e-5)
 x=measurement_reserve(m,state,cc,remaining=.05,config=c);assert not bool(x['enabled']) and float(x['cost_rate'])==0 and bool(x['valid'])

def test_optional_statistics_component_is_reported():
 from test_path_coverage_statistics import fixture
 from sttw_control.path_command_statistics import step_statistics,summarize_statistics
 logs,route,c=fixture();c['path_reward']=configured()['path_reward']
 for key in ['raw_components','effective_components','component_capped']:logs[key]['tolerance']=j.array([1.,2.,3.,j.nan])
 s=step_statistics(logs,route,c);assert s['cost_effective_sum'].shape==(11,)
 report=summarize_statistics(s,c);assert 'tolerance' in str(report)
