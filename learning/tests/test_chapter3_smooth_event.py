"""Test the actual CEM search with a controlled gate predicate and its delayed physical publisher."""
import json
from types import SimpleNamespace

import torch

from sttw_control import chapter3_p0_physical as physical
from sttw_control.chapter3_continuous_exit import ContinuousExitPlanner


def route_and_query(n=1):
    s = torch.linspace(0., 20., 201)
    xy = torch.stack((s, torch.zeros_like(s)), dim=1).expand(n, -1, -1).clone()
    route = SimpleNamespace(s=s, xy=xy, heading=torch.zeros_like(xy[:, :, 0]),
                            curvature=torch.zeros_like(xy[:, :, 0]), route_ids=torch.arange(n))
    query = SimpleNamespace(relative={'height': .16}, goal={'x_min_m': 16.5},
                            model=SimpleNamespace(model_kind='residual', alpha_conditioned=False))
    return route, query


def test_monitor_and_two_round_lateral_search(monkeypatch):
    route, query = route_and_query(4)
    planner = ContinuousExitPlanner(query, route, mode='smooth_event', fused=False)
    batches = []

    def predicted(q, velocity, prev, xy, heading, k, gate_x, side, remaining, gate, hold):
        batches.append(len(q))
        # This test fixes downstream success so that only positively bending candidates can pass
        # before crossing. The geometry, sampling, ranking and incumbent decisions are all native.
        peak_y = xy[:, :, 1].max(1).values
        passed = ((peak_y > .20) | (gate == 1)) & (gate != -1)
        return 16*passed.float()+peak_y, passed, torch.ones_like(side), remaining

    monkeypatch.setattr(planner, '_evaluate', predicted)
    q = torch.zeros(4, 12)
    q[:, 0] = 8.
    # Old high curvature behind the car must not invalidate a currently usable reference.
    route.curvature[:, 10] = 9.
    # This bend in the committed preview invalidates every possible continuation of the fourth lane.
    route.curvature[3, 90] = 2.
    v = torch.zeros(4, 9)
    result = planner.plan(q, v, v, route, 10., 12.7, 1., 3.,
                          gate_status=torch.tensor([0, 1, -1, 1]))
    assert batches == [1]*4 + [64, 64, 64, 64]
    assert result.searched.tolist() == [True, False, False, True]
    assert result.incumbent_admitted.tolist() == [False, True, False, False]
    assert result.admitted.tolist() == [True, True, False, False]
    for name in ('xy', 'heading', 'curvature'):
        actual, before = getattr(result, name), getattr(route, name)
        assert torch.equal(actual[0, :101], before[0, :101])
        assert torch.equal(actual[1:], before[1:])
    assert result.curvature[0, 79:].abs().amax() <= 1.
    assert result.xy[0, :, 1].max() > .20


def test_short_remaining_window_attenuates_steering_and_score_prefers_smooth(monkeypatch):
    route, query = route_and_query(2)
    planner = ContinuousExitPlanner(query, route, fused=False, mode='smooth_event')
    p = torch.tensor([[[.36,0.,0.]], [[.36,0.,0.]]])
    xy, _, k = planner._smooth_paths(route.xy, route.heading, route.curvature, torch.tensor([100,160]), p)
    assert xy[0, :, 1].max() > .35
    assert xy[1, :, 1].max() < .006
    assert xy[1, :, 1].max() > 0
    assert k[1].abs().max() <= 1.
    q = torch.zeros(2,12)
    q[:,0] = 8.
    v = torch.zeros(2,9)
    mono = torch.ones(2)
    planner._evaluate = lambda *a: (mono*16, mono.bool(), mono, mono*2)
    bends = torch.stack((route.s*0, torch.full_like(route.s,.8)))
    scores, ok, _, _ = planner._smooth_score(q,v,v,xy,route.heading,bends,mono,mono,mono,mono,mono)
    assert ok.all() and scores[0] > scores[1]


def test_monitor_renewal_keeps_same_route_and_single_arm_stops(tmp_path, monkeypatch):
    route, query = route_and_query(5)
    calls = []
    original = ContinuousExitPlanner

    def cpu_solver(query,route,**options):
        planner = original(query,route,**options,fused=False)
        def accept(q,*unused):
            calls.append(len(q))
            one = q.new_ones(len(q))
            return one*16,one.bool(),one,one*2
        planner._evaluate = accept
        return planner

    monkeypatch.setattr(physical, 'ContinuousExitPlanner', cpu_solver)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:None)
    q = torch.zeros(5,12); q[:,3]=1.
    v = torch.zeros(5,11)
    middle = SimpleNamespace(projection={'progress':torch.zeros(5)},v_user=torch.ones(5)*2.,
                             follower={'lookahead':torch.ones(5)})
    query.choose = lambda *a: (a[7],torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.bool))
    cfg = dict(group_size=5, deadline_ms=100, schedule_ticks=20, arm_schedule_ticks=[0,0,0,0,20],
               planner_mode='smooth_event', robots_per_chunk=1,
               smooth_latency_ms={'monitor':[5.],'search':[40.]})
    pub = physical.P0Branches(cfg,tmp_path,query,route,[dict(section_x=12.7,side=1)]*5,q,v)
    env = dict(q=q,v=v,route=route,middle=middle,alive=torch.ones(5,dtype=torch.bool),
               handoff_time=torch.zeros(5),hold=torch.zeros(5),approach_x=torch.zeros(1),
               approach_ids=torch.zeros(5,dtype=torch.long),arc64=route.s,join=torch.ones(5,dtype=torch.long),
               nominal_continuation_speed=middle.v_user)
    before = route.route_ids.clone()
    for tick in range(1,27):
        pub.step(tick,env,torch.full((5,),tick==1),torch.zeros(5,9))
    pub.finish()
    assert pub.queries.tolist() == [0,0,0,1,2]
    assert torch.equal(before,route.route_ids) and not pub.version.any()
    assert pub.ever_covered.tolist() == [False,False,False,True,True]
    records=[json.loads(line) for line in (tmp_path/'p0_events.jsonl').read_text().splitlines()]
    assert all(not any(r['searched']) for r in records if r['kind']=='query')
    publications=[r for r in records if r['kind']=='publication']
    assert len(publications)==2 and all(all(r['incumbent_renewed']) and not any(r['published']) for r in publications)
