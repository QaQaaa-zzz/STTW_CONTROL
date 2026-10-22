import numpy as np
import pytest
from sttw_control.tolerance_state_features import unstable_features

C=dict(mass=5.4,cg_height=.2,cg_forward=.164,wheelbase=.408,trail=.024,gravity=9.8,caster=np.deg2rad(25),minimum_speed=.5)

def test_equilibrium_has_zero_unstable_coordinate_and_rate_changes_sign():
 x=unstable_features(speed_proxy=2.6,steer=.1,roll=0,roll_rate=0,disturbance=0,controller=C)
 phi=-x['a2']*.1/x['a4']
 z=unstable_features(speed_proxy=2.6,steer=.1,roll=phi,roll_rate=0,disturbance=0,controller=C)
 assert abs(z['z_u'])<1e-12
 for rate in [-.3,.3]:
  y=unstable_features(speed_proxy=2.6,steer=.1,roll=phi,roll_rate=rate,disturbance=0,controller=C)
  assert y['z_u']==pytest.approx(rate)

def test_eso_constant_equilibrium_and_no_force_units_claim():
 x=unstable_features(speed_proxy=2.6,steer=0,roll=0,roll_rate=0,disturbance=2,controller=C)
 y=unstable_features(speed_proxy=2.6,steer=0,roll=-2/x['a4'],roll_rate=0,disturbance=2,controller=C)
 assert abs(y['z_u'])<1e-12
 assert 'margin_N' not in y

def test_nonfinite_rejected():
 with pytest.raises(ValueError):unstable_features(speed_proxy=2.6,steer=0,roll=float('nan'),roll_rate=0,disturbance=0,controller=C)
