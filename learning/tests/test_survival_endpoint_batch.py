"""Batching must preserve declared cases and use dynamic per-environment weights."""
import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control.survival_endpoint_batch import partition_cases, make_endpoint_chunk, simulation_alpha


def test_grouping_separates_geometry_mode_floor_and_actual_path_shapes():
    common=dict(panel='physical_deterministic',inference_mode='mean',speed_reference_floor_m_s=None)
    cases=[dict(common,case_id='a',route='left'),dict(common,case_id='b',route='right'),
           dict(common,case_id='c',route='left',speed_reference_floor_m_s=1.5),
           dict(common,case_id='d',route='left',inference_mode='zero'),
           dict(common,case_id='e',route='left',panel='nominal_deterministic')]
    shapes={('physical','left'):(20,),('physical','right'):(20,),('nominal','left'):(20,)}
    groups=partition_cases(cases,shapes)
    assert len(groups)==4 and groups[0]['case_indices']==[0,1]
    assert sorted(i for g in groups for i in g['case_indices'])==list(range(5))
    shapes[('physical','right')]=(21,)
    assert len(partition_cases(cases,shapes))==5  # no padded/changed geometry
    with pytest.raises(ValueError):partition_cases(cases+[cases[0]],shapes)


def test_dynamic_model_values_do_not_retrace_the_same_batch_kernel():
    traces=[]
    def step(c,p,std,path,speed,pulse):
        traces.append(True)
        return c+p*path+speed,(p,std)
    chunk=make_endpoint_chunk(step,steps_per_chunk=2)
    args=[j.zeros(3),j.array([1.,2.,3.]),j.zeros((3,2)),j.ones(3),j.zeros(3),j.zeros((3,5))]
    first=chunk(*args)[0];args[1]=j.array([4.,5.,6.]);second=chunk(*args)[0]
    np.testing.assert_array_equal(first,[2.,4.,6.]);np.testing.assert_array_equal(second,[8.,10.,12.])
    assert len(traces)==1


def test_zero_correction_batches_require_no_fabricated_parameter_tree():
    def step(c,p,std,path,speed,pulse):
        assert p is None
        return c+speed,j.zeros(2)
    chunk=make_endpoint_chunk(step,steps_per_chunk=2,zero_correction=True)
    end,_=chunk(j.zeros(3),None,j.zeros((3,2)),j.ones(3),j.array([1.,2.,3.]),j.zeros((3,5)))
    np.testing.assert_array_equal(end,[2.,4.,6.])
    assert simulation_alpha(dict(alpha=None,inference_mode='zero'))==0.
    with pytest.raises(ValueError):simulation_alpha(dict(alpha=None,inference_mode='mean'))
