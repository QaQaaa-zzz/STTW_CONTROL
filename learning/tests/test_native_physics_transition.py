"""Asset/identity preparation tests only; execute_preparation is never invoked."""
import copy
import json
from pathlib import Path
import pytest
from sttw_control.native_physics_transition import wheel_assets,prepare_transition,digest


def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def files(tmp_path):
    source=tmp_path/'old';native=source/'native_runtime/learning/src';code=native/'sttw_control/model.py';code.parent.mkdir(parents=True);code.write_text('# frozen source\n')
    for name in ['teleop_pref_governor_v1.json','fixed_endpoint_directional_alpha0.json']:put(native.parent/'configs'/name,{})
    upper=tmp_path/'upper';spec=upper/'alpha0/frozen_config_stage52.json';put(spec,dict(initialization=dict(initial_conditions_speed_roll=[[2.6,0.]]*8,preparation_seconds=3.5,preparation_zero_correction=True)))
    model={}
    for key in ['actor','checkpoint']:
        file=tmp_path/key;file.write_bytes(b'frozen checkpoint');model[key]=str(file);model[key+'_sha256']=digest(file)
    oldbank=tmp_path/'oldbank.pkl';oldbank.write_bytes(b'MUST NOT READ OR COPY')
    old=dict(native_runtime_root=str(native),native_expected_sources={str(code):digest(code)},models={'H75':model},initial_source=dict(candidate=model,upper_run=str(upper),prepared_bank=str(oldbank),prepared_bank_sha256=digest(oldbank),model={'old':'physics'},controller={},actuator={}),config={},contexts=[])
    put(source/'plan.json',old);put(source/'seal.json',dict(plan_sha256=digest(source/'plan.json')))
    xml=tmp_path/'newmodel/model.xml';xml.parent.mkdir();(xml.parent/'meshes').mkdir();(xml.parent/'meshes/wheel.stl').write_bytes(b'mesh')
    xml.write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh file="wheel.stl"/><mesh><plugin/></mesh></asset><worldbody><body name="frontwheel"><geom class="wheel" friction="0.8"/></body><body name="rearwheel"><geom class="wheel" friction="0.8"/></body></worldbody></mujoco>')
    cfg=tmp_path/'config.json';put(cfg,dict(source_xml=str(xml),expected_source_xml_sha256=digest(xml),expected_plant_spec_sha256=digest(spec),maximum_preparation_lower_ticks=5600,preparation_ticks=700,evaluation_bank_index=6,shared_jax_cache=str(tmp_path/'cache')))
    return source,tmp_path/'newrun',cfg,xml,oldbank


def test_isolated_copy_binds_new_assets_and_never_copies_old_prepared_bank(files):
    source,out,cfg,xml,oldbank=files;oldbytes=oldbank.read_bytes();before=digest(source/'plan.json')
    p=prepare_transition(source,out,cfg)
    assert not p['old_bank_reuse'] and p['maximum_preparation_lower_ticks']==5600
    assert 'prepared_bank' not in p['initial_source_pending'] and 'model' not in p['initial_source_pending']
    newxml=out/'native_runtime/mujoco_ros_ws20260313/model/scalebike_scene_matlab.xml'
    assert newxml.read_bytes()==xml.read_bytes() and not newxml.is_symlink()
    assert (newxml.parent/'meshes/wheel.stl').read_bytes()==b'mesh'
    assert not (out/'prepared_bank.pkl').exists() and not (out/'initial_source.json').exists()
    assert digest(source/'plan.json')==before and oldbank.read_bytes()==oldbytes
    with pytest.raises(ValueError,match='fresh'):prepare_transition(source,out,cfg)


def test_reject_model_or_frozen_source_drift_before_copy(files):
    source,out,cfg,xml,oldbank=files;xml.write_text(xml.read_text()+' ')
    with pytest.raises(ValueError,match='XML identity'):prepare_transition(source,out,cfg)
    assert not out.exists()


def test_wheel_friction_and_path_escape_are_explicit(files):
    source,out,cfg,xml,oldbank=files
    assets,bodies=wheel_assets(xml)
    assert assets==[Path('meshes/wheel.stl')] and bodies==['frontwheel','rearwheel']
    xml.write_text(xml.read_text().replace('friction="0.8"','friction="10"',1))
    with pytest.raises(ValueError,match='two explicit'):wheel_assets(xml)


def test_random_initial_bank_is_reproducible_stratified_and_separated(files):
    source,out,cfg,xml,oldbank=files
    spec=json.loads(cfg.read_text())
    spec.update(maximum_preparation_lower_ticks=11200,evaluation_bank_index=0,
        initial_randomization=dict(count=16,seed=930101,data_role='TRAIN',
                                   speed_m_s=[1.5,3.4],roll_rad=[-.06,.06]))
    put(cfg,spec)
    first=prepare_transition(source,out,cfg)
    second=prepare_transition(source,out.with_name('repeat'),cfg)
    values=first['initialization']['initial_conditions_speed_roll']
    assert values==second['initialization']['initial_conditions_speed_roll']
    assert first['mode']=='native_random_initial_preparation'
    assert len(values)==16 and len({int((v-1.5)/1.9*16) for v,r in values})==16
    assert all(1.5<=v<3.4 and -.06<=r<.06 for v,r in values)
    spec['initial_randomization'].update(seed=930102,data_role='DEV');put(cfg,spec)
    third=prepare_transition(source,out.with_name('dev'),cfg)
    assert values!=third['initialization']['initial_conditions_speed_roll']
    assert not (out/'prepared_bank.pkl').exists()
