"""Moving publication must preserve controller memory and reject closed corridors."""
from types import SimpleNamespace
import numpy as np
import torch
from sttw_control.moving_ditch_planner import MovingDitchPlanner, route_distances


def test_route_admission_uses_segment_distance_and_jump_deadline():
    planner=object.__new__(MovingDitchPlanner)
    planner.roads=np.array([[[0.,0.],[10.,0.]],[[0.,2.],[10.,2.]]])
    planner.parameters=np.array([[0.,1.],[0.,0.]])
    planner.task=SimpleNamespace(road_half_width=.6,approach={'jump_start_x_m':9.})
    roots=np.array([[3.,.5],[3.,1.5],[9.1,0.]])
    np.testing.assert_allclose(route_distances(roots,planner.roads),[[.5,1.5],[1.5,.5],[0.,2.]])
    assert planner.admission(roots)[0].tolist()==[[True,False],[False,True],[False,False]]


def test_same_route_publication_does_not_reinitialize_ground_controller():
    # No plant/project interface: reinitializing an unchanged route would fail.
    planner=object.__new__(MovingDitchPlanner);planner.parameters=np.array([[0.,0.,0.,0.,0.,2.,.5,0.],[0.,0.,0.,0.,0.,1.8,0.,1.]])
    memory=torch.tensor([3.,4.]);route=SimpleNamespace(route_ids=torch.tensor([1,0]))
    planner.task=SimpleNamespace(ground=SimpleNamespace(route=route,controller=memory,speed=torch.tensor([2.,2.]),alpha=torch.tensor([.5,.5])),env=SimpleNamespace(device='cpu'),ground_only=torch.tensor([False,True]),phase=torch.tensor([0,3]))
    planner.publish([0],[1])
    assert planner.task.ground.controller is memory
    torch.testing.assert_close(planner.task.ground.speed,torch.tensor([1.8,2.]))
    torch.testing.assert_close(planner.task.ground.alpha,torch.tensor([0.,.5]))
    assert planner.task.phase.tolist()==[0,3]
    assert planner.task.ground_only.tolist()==[False,True]


def test_ground_event_baseline_does_not_extrapolate_landing_fall():
    from sttw_control.ditch_event_model import EventTransitionNetwork
    state=torch.zeros(3,30);state[:,2]=.15;state[:,7]=2.;state[:,9]=-2.
    dt=torch.full((3,),3.);modes=torch.tensor([0,2,4])
    corrected=EventTransitionNetwork.physical_baseline(SimpleNamespace(ground_height_hold=True),state,dt,modes)
    old=EventTransitionNetwork.physical_baseline(SimpleNamespace(ground_height_hold=False),state,dt,modes)
    torch.testing.assert_close(corrected[:,2],torch.tensor([.15,-5.85,.15]))
    torch.testing.assert_close(old[:,2],torch.full((3,),-5.85))
    torch.testing.assert_close(corrected[:,0],old[:,0])
    # This is the predictor prior, never a modification of physical state.
    torch.testing.assert_close(state[:,2],torch.full((3,),.15))
