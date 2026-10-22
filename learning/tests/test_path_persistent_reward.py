import json
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as j
from sttw_control.path_command_reward import costs,reward_version
ROOT=Path(__file__).parents[1]
def config():
 c=json.loads((ROOT/'configs/STTW_Path_HighDemand_Shared.json').read_text())
 c['path_reward'].update(reward_version='path_persistent_preference_v3',path_secondary_ratio=.1)
 return c

def test_persistent_path_preference_and_endpoint_caps():
 c=config();assert reward_version(c)=='path_persistent_preference_v3'
 def run(a,ey=.3):return costs(alpha=a,chi=0.,ev=0.,ey=ey,eh=0.,peak_roll=0.,roll_rate=0.,speed=2.6,offsets=j.zeros(2),offset_rate=j.zeros(2),config=c)
 assert float(run(0)['effective']['path'])==50.
 assert np.isclose(float(run(1)['effective']['path']),5.)
 assert np.isclose(float(run(1,2.)['effective']['path']),20.)
 assert np.isclose(float(run(.5,2.)['effective']['path']),110.)
 assert bool(run(1,2.)['endpoint_capped']['path'])

def test_alpha0_and_common_costs_preserve_current_reward():
 c=config();old=json.loads((ROOT/'configs/STTW_Path_HighDemand_Shared.json').read_text())
 args=dict(chi=.8,ev=-.3,ey=.15,eh=.12,peak_roll=.34,roll_rate=1.2,speed=2.3,offsets=j.array([-.3,.1]),offset_rate=j.array([-.5,.2]))
 p=costs(alpha=0,config=c,**args);q=costs(alpha=0,config=old,**args);v=costs(alpha=1,config=c,**args);mid=costs(alpha=.37,config=c,**args)
 for key in p['effective']:
  np.testing.assert_allclose(p['effective'][key],q['effective'][key],rtol=1e-6)
  np.testing.assert_allclose(mid['effective'][key],.63*p['effective'][key]+.37*v['effective'][key],rtol=1e-6)
 for key in ('roll','working_roll','roll_rate','overspeed','low_speed','offset_magnitude','offset_rate'):
  assert p['effective'][key]==v['effective'][key]

def test_v3_primary_caps_endpoints_before_mixing():
 c=config();args=dict(chi=1.,ev=3.,ey=0.,eh=0.,peak_roll=0.,roll_rate=0.,speed=5.6,offsets=j.zeros(2),offset_rate=j.zeros(2),config=c)
 assert np.isclose(float(costs(alpha=.5,**args)['effective']['primary']),200.)


def test_independent_numpy_v3_reconstruction_matches_verified_candidate():
 from sttw_control.path_command_reporting import reward_series
 c=config();base=Path(c.get('teacher_run','/home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/runs/path_coverage_v2_resume_20261011'))
 if not base.exists():pytest.skip('local frozen teacher trajectories unavailable')
 d0=dict(np.load(base/'alpha0/final_best/left90_R2_2.6.npz'));d1=dict(np.load(base/'alpha1/final_best/left90_R2_2.6.npz'))
 a,_=reward_series(d0,1.,c);b,_=reward_series(d1,1.,c)
 assert np.isclose(b.sum()-a.sum(),10.34610008886307,atol=1e-7)
