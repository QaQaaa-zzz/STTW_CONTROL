"""Check handoff observer coordinates and isolation of nonreturning lanes."""
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from sttw_control.isaac_path_rollout import IsaacPathRollout


def test_measured_observer_initialization_is_lane_local():
    n=3
    controller=SimpleNamespace(eso=torch.full((n,4),9.),filtered_roll_rate=torch.full((n,),9.))
    def reset(ids):
        controller.eso[ids]=0
        controller.filtered_roll_rate[ids]=0
    controller.reset=reset
    measurement=torch.tensor([[.1,.2,.3,.4,.5,.6],[.7,.8,.9,1.,1.1,1.2],[1.3,1.4,1.5,1.6,1.7,1.8]])
    obj=SimpleNamespace(controller=controller,lower=SimpleNamespace(reset=Mock()),alive=torch.zeros(n,dtype=torch.bool),speed=torch.zeros(n),alpha=torch.zeros(n),plant=SimpleNamespace(observe=lambda:(measurement,torch.zeros(n,3),torch.ones(n))),measurement=torch.zeros_like(measurement),previous_reference=torch.zeros(n,2),previous_command=torch.zeros(n,2),route=SimpleNamespace(xy=torch.zeros(2,2),batched=False,s=torch.arange(2.)),middle=SimpleNamespace(reset_lanes=Mock(),projection={'progress':torch.zeros(n)}),n=n,task_tick=torch.ones(n),preference_progress=torch.zeros(n))
    IsaacPathRollout.resume_ground_lanes(obj,torch.tensor([0,2]),1.5,.5,observer_init_mask=torch.tensor([False,True]))
    torch.testing.assert_close(controller.eso[0],torch.zeros(4))
    torch.testing.assert_close(controller.eso[1],torch.full((4,),9.))
    torch.testing.assert_close(controller.eso[2],torch.tensor([1.5,1.3,1.4,0.]))
    torch.testing.assert_close(controller.filtered_roll_rate,torch.tensor([0.,9.,1.4]))
    assert not obj.alive[1]
    assert obj.middle.reset_lanes.call_count==1
