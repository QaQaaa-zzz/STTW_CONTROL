"""Complete-state capture, segment masks and bootstrap contract, no physics."""
import jax
import jax.numpy as j
import numpy as np
from flax import struct
from sttw_control.recovery_bank import select_complete_states


def test_complete_state_selection_keeps_clock_history_and_source_immutable():
    before={'tick':j.array([160,700]),'eso':j.arange(6.).reshape(2,3),
            'history':j.arange(24.).reshape(2,3,4)}
    old=jax.tree.map(lambda x:x-100,before)
    result=jax.jit(select_complete_states)(j.array([True,False]),old,before)
    for key in before:
        np.testing.assert_array_equal(result[key][0],before[key][0])
        np.testing.assert_array_equal(result[key][1],old[key][1])
    np.testing.assert_array_equal(before['tick'],[160,700])


def test_segment_timeout_is_truncation_not_physical_or_domain_terminal():
    from sttw_control.short_recovery import segment_flags
    flags=dict(done=j.array(False),physical_failure=j.array(False),
               tracking_domain_failure=j.array(False),engineering_fault=j.array(False),
               finite_task_end=j.array(False))
    a=segment_flags(flags,799,800);assert not bool(a['done'])
    b=jax.jit(segment_flags)(flags,800,800)
    assert bool(b['done']) and bool(b['segment_timeout'])
    assert not bool(b['finite_task_end']) and not bool(b['physical_failure'])
    for kind in ['physical_failure','tracking_domain_failure','engineering_fault']:
        c=segment_flags({**flags,'done':j.array(True),kind:j.array(True)},800,800)
        assert not bool(c['segment_timeout'])


def test_only_segment_timeouts_bootstrap_final_state_not_reset_value():
    import torch
    from sttw_control.short_recovery import bootstrap_segment_rewards
    r=torch.tensor([-1.,-2.,-3.]);v=torch.tensor([[10.],[20.],[30.]])
    x=bootstrap_segment_rewards(r,v,torch.tensor([True,False,False]),.997)
    np.testing.assert_allclose(x,[8.97,-2.,-3.],rtol=1e-6)
    np.testing.assert_array_equal(r,[-1.,-2.,-3.])
