from types import SimpleNamespace
import torch
from sttw_control.handoff_route_planner import HandoffRoutePlanner


def test_publication_waits_for_physical_clock_and_keeps_terminated_lanes():
    routes=torch.tensor([60,60]);ticks=torch.tensor([10,10]);finished=torch.tensor([False,False])
    task=SimpleNamespace(env=SimpleNamespace(dt=.02,episode_length_buf=ticks),finished=finished,
                         ground=SimpleNamespace(route=SimpleNamespace(route_ids=routes)))
    planner=HandoffRoutePlanner.__new__(HandoffRoutePlanner);planner.task=task
    a=dict(status='pending',selected_route=61);b=dict(status='pending',selected_route=62)
    planner.pending={0:(11,a),1:(11,b)};root=torch.zeros((2,13))
    planner.publish_due(root)
    assert routes.tolist()==[60,60] and len(planner.pending)==2
    ticks[:]=11;finished[1]=True;planner.publish_due(root)
    assert routes.tolist()==[61,60] and not planner.pending
    assert a['status']=='published' and a['publication_s']==.22
    assert b['status']=='terminated_before_publication'
