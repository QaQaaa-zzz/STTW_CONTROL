"""CPU geometry/command contract tests, independent of simulator dynamics."""
import copy
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control import path_command_scenarios as scenarios
from sttw_control.geometric_path import build_path
ROOT=Path(__file__).parents[2]
BASE=json.loads((ROOT/'learning/configs/STTW_Path_Feedback_V1.json').read_text())
SPEC=json.loads((ROOT/'docs/plans/path_coverage_v2/path_coverage_iteration_v2.json').read_text())
def config():
    c=copy.deepcopy(BASE);c['training_sampler']=copy.deepcopy(SPEC['sampler'])
    c['training_sampler'].update(version='path_conflict_exposure_v1',route_seed=910501,speed_event_seed=911503)
    return c

def test_stage_boundaries_preserve_legacy_and_generate_persistent_bends():
    c=config();pose=j.zeros(3)
    for update,stage,count,low,high in [(40,1,1,2.4,2.6),(41,2,2,2.4,2.7),(80,2,2,2.4,2.7),(81,3,2,2.7,2.9),(200,3,2,2.7,2.9)]:
        route=scenarios.sample_training_route(1,102,0,pose,c,absolute_update=update)
        assert int(route.sampling_stage)==stage
        assert int(np.sum(route.path.turn_mask))==count
        assert low<=float(route.speed)<=high
        assert route.speed_values.shape==(3,)
        np.testing.assert_array_equal(route.speed_values,np.repeat(route.speed,3))
    old=copy.deepcopy(c);old['training_sampler']['version']='demand_exposure_v2';old['training_sampler']['route_seed']=910501
    new=scenarios.sample_training_route(1,102,0,pose,c,absolute_update=40)
    prior=scenarios.sample_training_route(1,102,0,pose,old)
    for key in ['s','xy','heading','curvature','turn_end','goal']:
        np.testing.assert_array_equal(getattr(new.path,key),getattr(prior.path,key))

def test_multibend_pairs_cover_exact_angles_and_bounded_geometry():
    c=config();pose=j.zeros(3)
    for eid,nturn,angle_bounds in [(102,2,(150,180)),(306,4,(80,100))]:
        a=scenarios.sample_training_route(1,eid,3,pose,c,absolute_update=81)
        b=scenarios.sample_training_route(55,eid+1,3,pose,c,absolute_update=81)
        p=a.path;mask=np.asarray(p.turn_mask,bool)
        assert mask.sum()==nturn and p.s.shape==(5001,)
        starts=np.asarray(p.turn_starts)[mask];ends=np.asarray(p.turn_ends)[mask]
        assert 2<=starts[0]<=5
        assert np.all((starts[1:]-ends[:-1]>=.5)&(starts[1:]-ends[:-1]<=1.00001))
        assert float(p.goal)==pytest.approx(float(ends[-1])+4,abs=2e-6)
        for start,end,direction in zip(starts,ends,np.asarray(p.turn_directions)[mask]):
            delta=np.interp(end,p.s,p.heading)-np.interp(start,p.s,p.heading)
            assert angle_bounds[0]-.05<=abs(np.rad2deg(delta))<=angle_bounds[1]+.05
            assert np.sign(delta)==direction
        np.testing.assert_array_equal(p.xy[:,0],b.path.xy[:,0])
        np.testing.assert_array_equal(p.xy[:,1],-b.path.xy[:,1])

def test_speed_events_switch_on_boundary_and_only_ordinary_after_stage1():
    c=config();pose=j.zeros(3)
    a=scenarios.sample_training_route(1,0,5,pose,c,absolute_update=41)
    times=np.asarray(a.speed_switch_times);values=np.asarray(a.speed_values)
    assert 1<=times[0]<=3 and 5<=times[1]<=8
    np.testing.assert_allclose(times/.02,np.round(times/.02),atol=2e-5)
    assert values[0]==values[2] and abs(values[0]-values[1])>=.3
    for t,want in [(0,values[0]),(times[0]-.001,values[0]),(times[0],values[1]),(times[1],values[2])]:
        assert float(scenarios.speed_request(a.speed_values,a.speed_switch_times,j.asarray(t)))==want
    b=scenarios.sample_training_route(1,0,5,pose,c,absolute_update=40)
    np.testing.assert_array_equal(b.speed_values,np.repeat(b.speed,3))

def test_explicit_cases_keep_old_six_and_add_mirrored_fixed_grid_routes():
    rows=scenarios.explicit_case_declarations(config())
    assert len(rows)==12
    assert [r['name'] for r in rows[:6]]==[f'{r}_{v:.1f}' for r in BASE['stage_C_evaluation']['route_ids'] for v in [2.0,2.6]]
    for left,right,expected in [('left180_R1p8','right180_R1p8',np.pi),('compact_LR','compact_RL',0)]:
        a=build_path(left,[0,0,0],BASE);b=build_path(right,[0,0,0],BASE)
        assert a.s.shape==(5001,)
        np.testing.assert_allclose(a.s,np.arange(5001)*.02,atol=5e-6)
        np.testing.assert_array_equal(a.xy[:,0],b.xy[:,0]);np.testing.assert_array_equal(a.xy[:,1],-b.xy[:,1])
        assert float(a.heading[-1])==pytest.approx(expected,abs=2e-6)
        assert float(a.goal)==pytest.approx(float(a.turn_end)+4,abs=2e-6)
    assert rows[-1]['speed_values']==[2.3,2.8,2.3]
    assert rows[-1]['speed_switch_times']==[1.,5.]


def test_conflict_sampler_jit_vmap_and_validation():
    c=config();scenarios.validate_training_sampler(c)
    batch=jax.jit(jax.vmap(lambda eid:scenarios.sample_training_route(1,eid,2,j.zeros(3),c,absolute_update=j.asarray(81))))(j.array([0,101,102,103,306,307]))
    np.testing.assert_array_equal(batch.sample_group,[0,0,1,1,2,2])
    np.testing.assert_array_equal(np.sum(batch.path.turn_mask[2:],axis=1),[2,2,4,4])
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(batch))
    for key,value in [('route_seed',7),('speed_event_seed',7),('groups',[])]:
        bad=copy.deepcopy(c);bad['training_sampler'][key]=value
        with pytest.raises(ValueError):scenarios.validate_training_sampler(bad)


def test_conflict_config_enriches_old_geometry_without_changing_arrays():
    for name,count in [('straight',0),('left90_R2',1),('right90_R2',1)]:
        old=build_path(name,[0,0,0],BASE);new=build_path(name,[0,0,0],config())
        for key in ['s','xy','heading','curvature','turn_end','goal']:
            np.testing.assert_array_equal(getattr(old,key),getattr(new,key))
        assert old.turn_mask is None
        assert new.turn_mask is not None
        assert new.turn_mask.shape==(4,) and np.sum(new.turn_mask)==count
