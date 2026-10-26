"""Verify full-state clone/clock boundaries and that prefix freezing is inert."""
from types import SimpleNamespace
from flax import struct
import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control.same_state_reference_pause import clone_complete_carry,prefix_guard,pause_step,inventory,expected_mask

@struct.dataclass
class Physical:
    failed: object
    fifo: object
    eso: object
    rng: object

@struct.dataclass
class State:
    tick: object
    fault: object
    physical: object
    history: object
    filtered_offset: object

@struct.dataclass
class Carry:
    state: object
    gate: object
    clean_blocks: object
    relay: object


def state(tick=0):
    return Carry(State(j.int32(tick),j.bool_(False),Physical(j.bool_(False),j.array([2.,7.,11.]),j.array([.1,.2]),j.array([13,17],j.uint32)),j.arange(6).reshape(2,3),j.array([.04,-.12])),j.array([.3,.4]),j.int32(9),j.array([.2,.6]))


def test_clone_preserves_every_history_fifo_eso_rng_filter_gate_and_clock_leaf():
    originals=jax.tree.map(lambda *x:j.stack(x),state(836),state(864))
    before=jax.tree.map(np.asarray,originals)
    copied=clone_complete_carry(originals)
    for source,copy in zip(jax.tree.leaves(before),jax.tree.leaves(copied)):
        np.testing.assert_array_equal(copy,np.repeat(source,3,axis=0))
    # A branch-local change must not alter a sibling or the captured state.
    changed=copied.replace(state=copied.state.replace(filtered_offset=copied.state.filtered_offset.at[1,1].set(99.)))
    assert float(changed.state.filtered_offset[0,1])!=99
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(originals)):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(copied.state.tick,[836]*3+[864]*3)


def test_frozen_prefix_does_not_advance_any_part_of_state_or_count_physics():
    def advance(c,*args):
        return c.replace(state=c.state.replace(tick=c.state.tick+4,history=c.state.history+1)),dict(active_tick=j.ones(4,bool))
    guarded=jax.jit(prefix_guard(advance,dict(active_tick=j.zeros(4,bool))))
    captured=state(836);same,logs=guarded(captured,{},None,0.,None,0,j.int32(836))
    for a,b in zip(jax.tree.leaves(captured),jax.tree.leaves(same)):np.testing.assert_array_equal(a,b)
    assert not np.any(logs['active_tick'])
    prior=state(832);end,logs=guarded(prior,{},None,0.,None,0,j.int32(836))
    assert int(end.state.tick)==836 and np.all(logs['active_tick'])
    np.testing.assert_array_equal(end.state.physical.fifo,prior.state.physical.fifo)
    terminal=state(820).replace(state=state(820).state.replace(physical=state(820).state.physical.replace(failed=j.bool_(True))))
    end,logs=guarded(terminal,{},None,0.,None,0,j.int32(836))
    assert int(end.state.tick)==820 and not np.any(logs['active_tick'])


def test_pause_is_exactly_eighty_lower_ticks_and_does_not_reset_filter_or_history():
    def advance(c,p,*args):
        return c.replace(state=c.state.replace(tick=c.state.tick+4)),dict(source_keep_channels=j.tile(p['actor']['keep_channels'],(4,1)))
    step=jax.jit(pause_step(advance));fork=836
    p=dict(actor=dict(native=j.array([19.,23.]),keep_channels=j.array([True,True])),fork_task_tick=j.int32(fork),pause_mask=j.array([True,False]))
    for tick in [832,836,840,912,916,920]:
        original=state(tick);end,logs=step(original,p,None,0.,None,0)
        np.testing.assert_array_equal(logs['source_keep_channels'],expected_mask(np.arange(tick,tick+4),fork,[True,False]))
        for a,b in [(original.state.history,end.state.history),(original.state.filtered_offset,end.state.filtered_offset),(original.state.physical.rng,end.state.physical.rng),(original.gate,end.gate)]:np.testing.assert_array_equal(a,b)
        assert int(end.state.tick)==tick+4


def test_inventory_keeps_all_nominal_and_disturbed_endpoints_without_outcome_filtering():
    cases=[]
    for context in range(12):
        for alpha in [0.,1.]:
            for force in [0.,-16.,16.,-32.,32.]:
                cases.append(dict(case_id=f'c{context}/a{alpha}/{force}/goal_full_feedback',context=str(context),role='goal_full_feedback',alpha=alpha,force_N=force,source_keep_channels=[True,True],feedback_gain_rad=.1,goal_preview=True,causal_support=True,pulse_duration_s=.1,pulse_onset_task_tick=660))
    prefixes,suffixes=inventory(cases)
    assert len(prefixes)==120 and len(suffixes)==360
    assert all(c['fork_task_tick']==680 and c['pause_lower_ticks']==80 for c in suffixes)
    assert 'fork_task_tick' not in cases[0]
    with pytest.raises(ValueError):inventory(cases[:-1])
    bad=[dict(c) for c in cases];bad[0]['pulse_duration_s']=.14
    with pytest.raises(ValueError):inventory(bad)
