import json
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as j
from sttw_control import geometric_path as gp
from sttw_control import path_reference as pr
from sttw_control import path_command_policy as pp
CFG=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Feedback_V1.json').read_text())

def test_fixed_geometry_projection_and_sign():
 p=gp.build_path('straight',np.zeros(3),CFG,lateral_offset=0.)
 original=np.array(p.xy)
 s=gp.project(p,j.array([1.,.05]),j.array(1.),j.array(0.),CFG)
 assert float(s.cross_track)>0
 target=pr.pursuit(p,s.progress,j.array([1.,.05,0.]),j.array(2.),j.array(2.3),CFG)
 assert float(target.command[1])<0
 other=pr.pursuit(p,s.progress,j.array([1.,-.05,0.]),j.array(2.),j.array(2.3),CFG)
 np.testing.assert_allclose(target.command[1],-other.command[1],atol=1e-6)
 s2=gp.project(p,j.array([60.,0.]),s.progress,j.array(.01),CFG)
 assert float(s2.progress)<=float(s.progress)+.30001
 np.testing.assert_array_equal(p.xy,original)

def test_turn_angle_chord_and_domain():
 p=gp.build_path('left90_R2',np.zeros(3),CFG)
 assert abs(float(p.heading[-1])-np.pi/2)<1e-5
 assert abs(float(p.curvature.max())-.5)<1e-6
 r=pr.pursuit(p,j.array(4.),j.array([3.5,0.,0.]),j.array(2.),j.array(2.),CFG)
 x,y=np.asarray(r.preview_body)
 assert float(r.kappa)==pytest.approx(2*y/(x*x+y*y),rel=1e-6)
 r=pr.pursuit(p,j.array(0.),j.array([0.,0.,np.pi]),j.array(2.),j.array(2.),CFG)
 assert bool(r.invalid)

def test_filter_zero_slew_and_antiwindup():
 zero=j.zeros(2);raw=j.array([2.3,0.])
 f,o,g,flags=pp.correction_tick(zero,zero,zero,raw,CFG)
 np.testing.assert_array_equal(g,raw)
 f,o,g,_=pp.correction_tick(zero,zero,j.array([-1.,1.]),raw,CFG)
 target=np.tanh([-1.,1.])*[1.,.2]
 np.testing.assert_allclose(f,(1-np.exp(-.005/np.array([.1,.08])))*target,rtol=1e-5)
 assert float(o[0])==pytest.approx(-.005,abs=2e-7)
 f,o,g,flags=pp.correction_tick(j.array([.2,.2]),j.array([.2,.2]),j.ones(2),j.array([3.,.35]),CFG)
 np.testing.assert_array_equal(f,o);assert bool(flags['reference_clipped'])

def test_new_schema_rejects_old_and_nonfinite():
 with pytest.raises(ValueError):pp.validate_actor_input(j.zeros(345))
 pp.validate_actor_input(j.zeros(351))
 obs,fault=pp.assemble_observation(j.zeros((16,20)),j.zeros(16),j.zeros(15),j.ones(20),CFG)
 assert obs.shape==(351,) and not bool(fault)
 obs,fault=pp.assemble_observation(j.zeros((16,20)),j.zeros(16),j.zeros(15).at[1].set(j.nan),j.ones(20),CFG)
 assert bool(fault) and bool(j.isnan(obs).any())

def test_path_reward_caps_and_finite_horizon():
 from sttw_control.path_command_reward import costs,failure_reward
 c=costs(alpha=0.,chi=1.,ev=.2,ey=.3,eh=.1,peak_roll=.31,roll_rate=1.,speed=2.,offsets=j.zeros(2),offset_rate=j.zeros(2),config=CFG)
 assert float(c['raw']['path'])>float(c['raw']['speed'])
 assert sum(CFG['path_reward']['component_caps'].values())==pytest.approx(5080.2)
 assert float(failure_reward(1,CFG))==pytest.approx(-15.1604,abs=1e-4)

def test_fresh_actor_zero_and_legacy_params_rejected():
 import jax
 actor=pp.PathCommandActor();x=j.zeros(351)
 params=actor.init(jax.random.PRNGKey(1),x)
 np.testing.assert_array_equal(actor.apply(params,x),np.zeros(2))
 critic=pp.PathCommandCritic();assert critic.apply(critic.init(jax.random.PRNGKey(2),j.zeros(352)),j.zeros(352)).shape==(1,)
 from sttw_control.direct_command_policy import DirectCommandActor
 old=DirectCommandActor().init(jax.random.PRNGKey(3),j.zeros(345))
 with pytest.raises(Exception):actor.apply(old,x)


def test_nominal_speed_has_no_curvature_regulation_and_slew():
 p=gp.build_path('left90_R2',np.zeros(3),CFG)
 r=pr.pursuit(p,j.array(4.),j.array([3.5,0.,0.]),j.array(2.),j.array(2.6),CFG)
 assert float(r.command[0])==pytest.approx(2.6)
 raw,rates=pr.publish(j.array([2.3,0.]),r.command,CFG)
 np.testing.assert_allclose(raw[0],2.3025,atol=2e-7)
 assert abs(float(rates[1]))<=.30001

def test_nonfinite_lower_output_cannot_enter_actuator_composition():
 from sttw_control.path_command_env import guard_lower_action
 np.testing.assert_array_equal(guard_lower_action(j.array([j.nan,j.inf]),j.bool_(False)),np.zeros(2))
 action=j.array([.2,-.3]);np.testing.assert_array_equal(guard_lower_action(action,j.bool_(True)),action)
