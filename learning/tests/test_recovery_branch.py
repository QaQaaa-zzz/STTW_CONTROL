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
