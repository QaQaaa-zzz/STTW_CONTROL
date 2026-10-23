import numpy as np
import pytest
from sttw_control.preference_switching import scheduled_alpha,validate_schedule

def test_bidirectional_step_boundary_and_fixed():
 for a in (0.,1.):
  assert float(scheduled_alpha(a,299,.005,1.5,.5,'step'))==a
  assert float(scheduled_alpha(a,300,.005,1.5,.5,'step'))==1-a
  assert float(scheduled_alpha(a,500,.005,1.5,.5,'fixed'))==a

def test_ramp_preserves_endpoints_and_midpoint():
 for a in (0.,1.):
  np.testing.assert_allclose([scheduled_alpha(a,t,.005,1.5,.5,'ramp') for t in (200,300,350,400,500)],[a,a,.5,1-a,1-a])

def test_schedule_rejects_invalid_contract():
 validate_schedule(dict(switch_s=1.5,ramp_s=.5,modes={'a':'step'}),10,.02)
 for t,r,m in [(0,.5,'step'),(9.8,.5,'step'),(1.51,.5,'step'),(1.5,.5,'unknown')]:
  with pytest.raises(ValueError):validate_schedule(dict(switch_s=t,ramp_s=r,modes={'a':m}),10,.02)
