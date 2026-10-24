"""A capture block must stop each lane at its first control-step crossing."""
import importlib.util
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np


def test_batched_capture_preserves_first_crossing_and_counts_only_active_steps():
    path=Path(__file__).resolve().parents[2]/'docs/evidence/tolerance_progress_20261022/alpha_physical_boundary/collect_prefixes.py'
    spec=importlib.util.spec_from_file_location('phase_collector',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert hasattr(module,'capture_first_crossing_block'), 'first-crossing scan block is missing'
    def advance(state,done):
        return state+j.where(done,0.,j.array([1.,2.,3.])),dict(active=~done)
    result=jax.jit(lambda state:module.capture_first_crossing_block(
        advance,state,j.zeros(3,dtype=bool),j.array(5.),lambda x:x,50))(j.zeros(3))
    state,done,previous,logs=result
    np.testing.assert_array_equal(state,[5.,6.,6.])
    np.testing.assert_array_equal(previous,[4.,4.,3.])
    np.testing.assert_array_equal(done,[True,True,True])
    np.testing.assert_array_equal(np.asarray(logs['active']).sum(axis=0),[5,3,2])


def test_capture_continues_across_blocks_without_advancing_completed_lanes():
    path=Path(__file__).resolve().parents[2]/'docs/evidence/tolerance_progress_20261022/alpha_physical_boundary/collect_prefixes.py'
    spec=importlib.util.spec_from_file_location('phase_collector',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert hasattr(module,'capture_first_crossing_block'), 'first-crossing scan block is missing'
    def advance(state,done):
        return state+j.where(done,0.,j.array([1.,3.])),dict(active=~done)
    state,done,_,_=module.capture_first_crossing_block(advance,j.zeros(2),j.zeros(2,dtype=bool),5.,lambda x:x,2)
    np.testing.assert_array_equal(state,[2.,6.])
    state,done,previous,logs=module.capture_first_crossing_block(advance,state,done,5.,lambda x:x,50)
    np.testing.assert_array_equal(state,[5.,6.])
    np.testing.assert_array_equal(np.asarray(logs['active']).sum(axis=0),[3,0])
    assert previous[0]==4.
