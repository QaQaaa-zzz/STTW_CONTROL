"""Regression for uninterrupted motion, handback-relative clocks and stale route publications."""
import json
from types import SimpleNamespace

import pytest
import torch

from sttw_control import chapter3_p0_physical as physical
from sttw_control.chapter3_continuous_exit import ExitProposal


class FakeSolver:
    def __init__(self,*unused,**options):
        self.calls=0
        self.reject_second=True

    def plan(self,q,velocity,previous,route,committed,sections,sides,remaining,*,rows,**options):
        self.calls+=1
        xy=route.xy[rows].clone()
        xy[:,8:,1]=.35
        fields=[getattr(route,k)[rows].clone() for k in ('heading','curvature')]
        admitted=torch.full((len(q),),not (self.reject_second and self.calls==3))
        return ExitProposal(xy,*fields,admitted,admitted,torch.ones(len(q)),
                            torch.full((len(q),),.12 if self.calls==2 else 2.))


@pytest.mark.parametrize('delayed',[False,True])
def test_fallback_after_expiry_and_logged_consumption(tmp_path,monkeypatch,delayed):
    monkeypatch.setattr(physical,'ContinuousExitPlanner',FakeSolver)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    q=torch.zeros(7,12);q[:,3]=1.
    s=torch.linspace(0,30,61)
    bank=torch.stack((s,torch.zeros_like(s)),-1).expand(6+18*len(s),-1,-1).clone()
    route=SimpleNamespace(s=s,xy=bank,heading=torch.zeros_like(bank[:,:,0]),curvature=torch.zeros_like(bank[:,:,0]),
                          route_ids=torch.full((7,),6+len(s)+1))
    choose=lambda *args:(args[7],torch.zeros(4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool))
    middle=SimpleNamespace(projection={'progress':torch.zeros(7)},follower={'lookahead':torch.ones(7),'command':torch.zeros(7,2)},
                           pose=q[:,:3].clone(),v_user=torch.full((7,),2.),lane_ticks=torch.zeros(7,dtype=torch.long))
    cfg=dict(group_size=7,deadline_ms=100,schedule_ticks=20,latency_ms=[11.5,40. if delayed else 11.5]+[11.5]*98)
    env=dict(q=q,v=torch.zeros(7,11),route=route,middle=middle,alive=torch.ones(7,dtype=torch.bool),
        handoff_time=torch.zeros(7),hold=torch.zeros(7),approach_x=torch.zeros(6),approach_ids=torch.zeros(7,dtype=torch.long),
        arc64=s,join=torch.ones(7,dtype=torch.long),nominal_continuation_speed=middle.v_user)
    p=physical.P0Branches(cfg,tmp_path,SimpleNamespace(choose=choose),route,[dict(section_x=12.7,side=1)]*7,q,env['v'])
    if delayed:p.models.reject_second=False
    for tick in range(1,34):
        middle.lane_ticks.fill_(tick+1)
        p.step(tick,env,torch.full((7,),tick==1),torch.zeros(7,9))
        p.actor_command(tick,env['alive'],torch.full((7,),2),middle,torch.zeros(7,4))
        middle.lane_ticks+=1
        p.pursuit_read(tick+1,env['alive'],torch.full((7,),2),middle)
        if tick==24:
            assert p.active_plan[3:].all()
            assert route.xy[route.route_ids[3],-1,1].item()>.3
    assert not p.active_plan.any()
    assert torch.equal(p.version,torch.tensor([0,0,0,2,2,2,2]))
    assert torch.count_nonzero(route.xy[route.route_ids[3:],-1,1])==0
    assert p.initial_uncovered_ticks[3:].tolist()==[3]*4
    assert (p.expired_uncovered_ticks[3:]>0).all()
    p.finish()
    log=[json.loads(s) for s in (tmp_path/'p0_events.jsonl').read_text().splitlines()]
    fall=[e for e in log if e['kind']=='expired_to_default']
    assert len(fall)==1
    assert fall[0]['tick']>fall[0]['valid_until'][0]
    assert [e for e in log if e['kind']=='publication'][-1]['published']==[False]*4
    if delayed: assert fall[0]['tick'] < [e for e in log if e['kind']=='publication'][-1]['tick']
    assert next(e['tick'] for e in log if e['kind']=='h75_actor_consumed')==7
    assert next(e['tick'] for e in log if e['kind']=='pursuit_consumed')==7


def test_default_rejoin_preserves_all_committed_samples_and_derivatives():
    arc=torch.linspace(0.,20.,201)
    xy=torch.stack((arc,torch.ones_like(arc)*.35),-1);straight=xy.clone();straight[:,1]=0
    route=SimpleNamespace(s=arc,xy=torch.stack((xy,straight)),
                          heading=torch.stack((torch.full_like(arc,.1),arc*0)),
                          curvature=torch.stack((torch.full_like(arc,.05),arc*0)))
    output=physical.rejoin_default(route,torch.tensor([0]),torch.tensor([1]),torch.tensor([2.251]))
    for key,f in zip(('xy','heading','curvature'),output):
        assert torch.equal(f[0,:24],getattr(route,key)[0,:24])
    assert torch.max(output[0][0,45:,1].abs())<1e-6
    assert (output[0][0,:,0].diff()>0).all()
