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
