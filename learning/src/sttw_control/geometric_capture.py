"""Batch physical prefixes while freezing each lane at its first crossing."""
import jax
import jax.numpy as j


def capture_first_crossing_block(advance,states,done,target,get_progress,length):
    """Advance control boundaries only; advance must mask already-done lanes.

    Returns the preserved complete states, done flags, progress immediately
    before each first crossing, and all logs. No state components are reset.
    """
    previous=get_progress(states)
    def step(carry,_):
        state,finished,before=carry
        old_progress=get_progress(state)
        next_state,log=advance(state,finished)
        next_finished=finished|(get_progress(next_state)>=target)
        before=j.where(~finished&next_finished,old_progress,before)
        return (next_state,next_finished,before),log
    carry,logs=jax.lax.scan(step,(states,done,previous),None,length=length)
    return *carry,logs
