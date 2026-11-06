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


def test_original_route_switch_survives_replanned_progress_reset():
 import torch
 from types import SimpleNamespace
 from sttw_control.isaac_path_rollout import IsaacPathRollout
 from sttw_control.torch_path_reference import TorchPathReference
 route=TorchPathReference.__new__(TorchPathReference)
 route.s=torch.tensor([0.,10.]);route.xy=torch.tensor([[0.,0.],[10.,0.]])
 route.heading=torch.zeros(2);route.curvature=torch.zeros(2);route.batched=False
 route.config={'path':{'projection_backward_m':1.,'projection_forward_floor_m':2.,'projection_forward_motion_multiplier':2.,'projection_forward_padding_m':1.}}
 env=IsaacPathRollout.__new__(IsaacPathRollout)
 env.switch={'progress_s_m':5.3};env.preference_route=route;env.preference_progress=torch.tensor([5.2])
 env.middle=SimpleNamespace(projection={'progress':torch.tensor([0.])})
 env.plant=SimpleNamespace(observe=lambda:(None,torch.tensor([[5.4,0.,0.]]),torch.tensor([1.5])))
 env.alive=torch.tensor([True]);env.switched=torch.tensor([False]);env.alpha=torch.tensor([1.])
 env.switch_target=torch.tensor([0.]);env.switch_mode=torch.tensor([0]);env.tick=100
 env.switch_tick=torch.tensor([-1]);env.policy=SimpleNamespace(execution_mode=torch.tensor([0]),filtered=torch.tensor([[.1,.2]]))
 before=env.policy.filtered.clone()
 env.update_preference()
 assert env.alpha.item()==0 and env.switch_tick.item()==400
 assert env.middle.projection['progress'].item()==0
 torch.testing.assert_close(env.policy.filtered,before)
 env.tick+=1;env.update_preference()
 assert env.switch_tick.item()==400
