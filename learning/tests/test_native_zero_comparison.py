"""Refuse mismatched native evidence before constructing a physical runtime."""
import json
import pytest
from sttw_control.native_zero_comparison import digest, prepare, run, save, verify


def test_incomplete_native_panel_is_not_replaced(tmp_path):
    source=tmp_path/'source';(source/'evaluation').mkdir(parents=True)
    save(source/'evaluation/status.json',dict(phase='running',completed_cases=4))
    output=tmp_path/'new'
    with pytest.raises(ValueError,match='completed native current8'):
        prepare(source,output)
    assert not output.exists()


@pytest.mark.parametrize('category',['input_sha256','native_source_sha256','runner_sha256'])
def test_changed_bound_file_stops_before_runtime(tmp_path,category):
    root=tmp_path/'run';root.mkdir();f=tmp_path/'bound.py';f.write_text('old')
    plan=dict(input_sha256={},native_source_sha256={},runner_sha256={})
    plan[category]={str(f):digest(f)};save(root/'plan.json',plan)
    save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'cached_records.json',[]);f.write_text('changed')
    with pytest.raises(ValueError,match='sealed input/source changed'):
        verify(root)


def test_modern_package_cannot_silently_supply_native_physics(tmp_path):
    root=tmp_path/'run';root.mkdir()
    save(root/'plan.json',dict(native_runtime_root=str(tmp_path/'native'),input_sha256={},native_source_sha256={},runner_sha256={}))
    save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'cached_records.json',[]);save(root/'status.json',dict(phase='prepared'))
    with pytest.raises(ValueError,match='modern sttw_control already imported'):
        run(root)
    assert not (root/'evaluation').exists()


def capability_protocol():
    return dict(experiment_id='native_capability_test',horizon_s=10.,alphas=[0.,.5,1.],expected_cases=16,maximum_lower_ticks=32000,
        contexts=[dict(route=f's{int(a)}_{int(d)}',angle_deg=a,radius_m=r,speed=v,direction=d)
                  for a,r,v in [(60.,2.2,2.6),(90.,1.8,2.8)] for d in [-1.,1.]])


def test_capability_inventory_has_twelve_native_h75_and_four_zero():
    from sttw_control.native_zero_comparison import capability_cases
    cases=capability_cases(capability_protocol())
    assert len(cases)==16
    for route in {c['route'] for c in cases}:
        rows=[c for c in cases if c['route']==route]
        assert [(c['policy'],c['alpha']) for c in rows]==[('H75',0.),('H75',.5),('H75',1.),('zero',0.)]
    assert all(c['force_N']==0 for c in cases)


@pytest.mark.parametrize('change',['speed','radius','alpha','horizon','duplicate'])
def test_capability_rejects_inventory_drift(change):
    from sttw_control.native_zero_comparison import capability_cases
    p=capability_protocol()
    if change=='speed':p['contexts'][0]['speed']=1.8
    elif change=='radius':p['contexts'][0]['radius_m']=1.1
    elif change=='alpha':p['alphas']=[0.,1.]
    elif change=='horizon':p['horizon_s']=16.
    else:p['contexts'][0]['route']=p['contexts'][1]['route']
    with pytest.raises(ValueError):capability_cases(p)


def test_capability_geometry_remains_protocol_defined():
    from sttw_control.native_zero_comparison import capability_cases
    p=capability_protocol()
    for c in p['contexts']:
        c['angle_deg']+=5.
        c['radius_m']+=.1
    assert len(capability_cases(p))==16


def test_channel_factorial_inventory_and_protocol_guards():
    from sttw_control.native_zero_comparison import channel_factorial_cases
    from pathlib import Path
    p=json.loads((Path(__file__).parents[1]/'configs/h75_channel_factorial.json').read_text())
    contexts=capability_protocol()['contexts']
    cases=channel_factorial_cases(p,contexts)
    assert len(cases)==8
    assert [(x['speed_alpha'],x['steer_alpha']) for x in cases]==[(.5,1.),(1.,.5)]*4
    assert all(x['alpha']==x['original_user_alpha']==1 and x['force_N']==0 for x in cases)
    for key,value in [('replanning',True),('horizon_s',16.),('channel_alpha_pairs',[[.5,.5],[1.,1.]])]:
        with pytest.raises(ValueError):channel_factorial_cases(dict(p,**{key:value}),contexts)


@pytest.mark.parametrize('speed_alpha,expected',[(.5,[5.,100.]),(1.,[10.,50.])])
def test_channel_factorial_same_features_two_queries_one_physics_step(speed_alpha,expected):
    import jax.numpy as j
    import numpy as np
    from typing import NamedTuple
    from sttw_control.native_zero_comparison import channel_factorial_decision
    class State(NamedTuple):
        alpha: object
        fault: object
        history: object
        def replace(self,**kw):return self._replace(**kw)
    state=State(j.array(1.),j.array(False),j.arange(20.))
    original=j.arange(351.)/351
    original=original.at[336].set(1.)
    queries=[];observed=[];steps=[]
    def observe(st,path,v,h):
        observed.append(st)
        assert st is state
        return original,None,j.array(True)
    def actor(obs):
        queries.append(np.asarray(obs))
        return j.array([10*obs[336],100*obs[336]])
    def physics(st,z,path,v,h,force):
        steps.append(st)
        assert bool(st.fault) and float(st.alpha)==1.
        np.testing.assert_array_equal(st.history,state.history)
        np.testing.assert_array_equal(force,j.zeros(5))
        return st,dict(active_tick=j.ones(4,dtype=bool),latent_z=j.broadcast_to(z,(4,2)))
    end,logs=channel_factorial_decision(state,None,2.6,speed_alpha,observe,actor,physics)
    assert len(observed)==len(steps)==1 and len(queries)==2
    mask=np.arange(351)!=336
    np.testing.assert_array_equal(queries[0][mask],queries[1][mask])
    np.testing.assert_array_equal(queries[0][mask],np.asarray(original)[mask])
    assert queries[0][336]==.5 and queries[1][336]==1.
    np.testing.assert_array_equal(logs['composed_latent'],np.tile(expected,(4,1)))
    np.testing.assert_array_equal(logs['latent_z'],logs['composed_latent'])
    np.testing.assert_array_equal(logs['queried_latent_alpha_half'],np.tile([5.,50.],(4,1)))
    np.testing.assert_array_equal(logs['queried_latent_alpha_one'],np.tile([10.,100.],(4,1)))
    np.testing.assert_array_equal(logs['channel_alpha_pair'],np.tile([speed_alpha,1.5-speed_alpha],(4,1)))
    np.testing.assert_array_equal(logs['original_user_alpha'],np.ones(4))
    assert float(end.alpha)==1.
