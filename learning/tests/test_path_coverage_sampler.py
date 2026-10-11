"""CPU-only reset distribution checks; no dynamics or training."""
import copy
import json
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control import path_command_scenarios as scenarios

ROOT=Path(__file__).parents[2]
BASE=json.loads((ROOT/'learning/configs/STTW_Path_Feedback_V1.json').read_text())
SPEC=json.loads((ROOT/'docs/plans/path_coverage_v2/path_coverage_iteration_v2.json').read_text())


def config():
    cfg=copy.deepcopy(BASE)
    cfg['training_sampler']=copy.deepcopy(SPEC['sampler'])
    return cfg


def test_legacy_dispatch_and_v2_retained_lanes_are_exact():
    cfg=config();pose=j.array([3.,4.,.2])
    for eid in [0,42,101]:
        for ep in [0,7]:
            for seed,c in [(17,BASE),(910201,cfg)]:
                actual=jax.jit(lambda:scenarios.sample_training_route(17,eid,ep,pose,c))()
                expected=jax.jit(lambda:scenarios.sample_route(scenarios.episode_key(seed,eid,ep),pose,BASE))()
                for x,y in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
                    np.testing.assert_array_equal(x,y)
    scenarios.validate_training_sampler(BASE)


def test_jit_vmap_group_counts_bounds_and_local_reflection():
    cfg=config();scenarios.validate_training_sampler(cfg)
    pose=j.array([3.,4.,.2]);before=np.asarray(pose).copy()
    batch=jax.jit(jax.vmap(lambda eid:scenarios.sample_training_route(19,eid,3,pose,cfg)))(j.arange(512))
    np.testing.assert_array_equal(np.bincount(np.asarray(batch.sample_group)),[102,204,206])
    assert batch.path.xy.shape==(512,5001,2)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(batch))
    np.testing.assert_array_equal(batch.family[102:306],1)
    np.testing.assert_array_equal(batch.family[306:],2)
    for values,low,high in [(batch.speed[102:],2.4,2.6),(batch.radius[102:],2.,2.2),
                            (np.abs(batch.angle[102:]),np.deg2rad(90),np.deg2rad(100))]:
        assert np.all((values>=low)&(values<=high))
    for field in ['speed','radius']:
        np.testing.assert_array_equal(getattr(batch,field)[102::2],getattr(batch,field)[103::2])
    for field in ['angle','lateral_offset','heading_offset']:
        np.testing.assert_array_equal(getattr(batch,field)[102::2],-getattr(batch,field)[103::2])
    assert np.any(np.asarray(batch.lateral_offset[102:])!=0)
    np.testing.assert_allclose(batch.path.curvature[102::2],-batch.path.curvature[103::2],atol=0)
    xy=np.asarray(batch.path.xy)-before[:2]
    c,s=np.cos(before[2]),np.sin(before[2])
    local=xy@np.array([[c,-s],[s,c]])
    np.testing.assert_allclose(local[102::2,:,0],local[103::2,:,0],atol=2e-5)
    np.testing.assert_allclose(local[102::2,:,1],-local[103::2,:,1],atol=2e-5)
    np.testing.assert_array_equal(pose,before)


def test_route_rng_is_endpoint_independent_but_uses_actual_episode_counter():
    cfg=config();pose=j.zeros(3)
    a=scenarios.sample_training_route(1,102,3,pose,cfg)
    b=scenarios.sample_training_route(999,102,3,pose,cfg)
    c=scenarios.sample_training_route(1,103,4,pose,cfg)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    assert float(a.radius)!=float(c.radius)


@pytest.mark.parametrize('section,key,value',[
    ('training_sampler','version','unknown'),('training_sampler','route_seed',1),
    ('training_sampler','groups',[]),('training_sampler','strong',{}),
    ('training_future_phase_C','num_envs',510),('training_future_phase_C','episode_s',10.),
    ('training_future_phase_C','initial_route_pose_offset_fraction',0.),
    ('path','spacing_m',.04),('path','straight_prefix_m',2.),
    ('path','goal_section_offset_after_turn_m',3.),('path','reanchor_during_rollout',True),
])
def test_v2_contract_rejects_unapproved_distribution_or_geometry(section,key,value):
    cfg=config();cfg[section][key]=value
    with pytest.raises(ValueError):scenarios.validate_training_sampler(cfg)
