import torch
from sttw_control.contact_events import WheelClearanceEvents


def test_airborne_gap_is_not_clearance_and_dispatch_waits_for_decision():
    events=WheelClearanceEvents(2,'cpu',.0002,confirmation_s=.001)
    alive=torch.tensor([True,True]);beyond=torch.tensor([False,False]);ground=torch.tensor([True,True])
    events.update(torch.tensor([[True,False],[True,True]]),beyond,ground,alive)
    events.update(torch.tensor([[False,True],[False,False]]),beyond,ground,alive)
    for _ in range(8):events.update(torch.zeros(2,2,dtype=torch.bool),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),alive)
    assert not events.take_ready(alive).any()  # no support while airborne
    events.update(torch.zeros(2,2,dtype=torch.bool),torch.tensor([True,False]),ground,alive)
    assert events.confirmed.tolist()==[11,-1]
    assert not events.dispatched.any()
    assert events.take_ready(alive).tolist()==[True,False]
    assert not events.take_ready(alive).any()
    events.update(torch.zeros(2,2,dtype=torch.bool),torch.ones(2,dtype=torch.bool),ground,torch.tensor([True,False]))
    assert not events.take_ready(torch.tensor([True,False])).any()


def test_recontact_restarts_confirmation_interval():
    events=WheelClearanceEvents(1,'cpu',.0002,confirmation_s=.0006)
    yes=torch.tensor([True]);no_contact=torch.zeros(1,2,dtype=torch.bool)
    events.update(torch.ones(1,2,dtype=torch.bool),yes,yes,yes)
    for _ in range(2):events.update(no_contact,yes,yes,yes)
    assert events.confirmed.item()==-1
    events.update(torch.tensor([[False,True]]),yes,yes,yes)
    for _ in range(2):events.update(no_contact,yes,yes,yes)
    assert events.confirmed.item()==-1
    events.update(no_contact,yes,yes,yes)
    assert events.take_ready(yes).item()
