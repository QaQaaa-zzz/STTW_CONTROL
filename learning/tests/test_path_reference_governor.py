"""Reference authority, incomplete forecasts and full-carry prediction contracts."""
import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import struct
from sttw_control.path_reference_governor import candidates,choose,forecast_rows,make_preview_rollout
from sttw_control.path_command_policy import map_latent

CFG=dict(residual_interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2))


def test_candidates_are_physical_not_latent_interpolation_with_exact_endpoints():
    old=np.array([-20.,20.],np.float32);new=np.array([20.,-20.],np.float32)
    c=candidates(old,new,CFG)
    np.testing.assert_array_equal(c.latents[0],old);np.testing.assert_array_equal(c.latents[-1],new)
    # The midpoint is (-1+.25)/2, not tanh(0)=0 on the speed channel.
    np.testing.assert_allclose(c.physical_offsets[2],[-.375,0.],atol=2e-7)
    assert np.all(c.physical_offsets[:,0]>=-1) and np.all(c.physical_offsets[:,0]<=.25)
    assert np.max(abs(c.physical_offsets[:,1]))<=.2
    np.testing.assert_array_equal(map_latent(c.latents[-1],CFG),map_latent(new,CFG))


def test_selection_does_not_assume_monotone_feasibility_and_never_certifies_fallback():
    p=[dict(status=s) for s in ['contact','survived','contact','survived','contact']]
    d=choose(p);assert d.index==3 and d.predicted_admissible and not d.fallback_used
    d=choose([dict(status='contact')]*5)
    assert d.index==0 and d.fallback_used and not d.predicted_admissible
    p[2]['status']='unknown';d=choose(p)
    assert d.index is None and d.status=='prediction_error' and not d.fallback_used


def test_incomplete_and_faulted_forecasts_remain_unknown_even_if_no_contact():
    s=dict(observed_lower_ticks=np.array([200,150,100,0,200]),requested_lower_ticks=np.full(5,200),
        allocated_lower_slots=np.full(5,200),contact=np.array([0,1,0,0,0],bool),finite=np.array([1,1,1,1,0],bool),
        admissible=np.array([1,0,0,0,0],bool),original_user_cost=np.arange(5.))
    r=forecast_rows(s);assert [q['status'] for q in r]==['survived','contact','unknown','unknown','unknown']
    assert choose(r).status=='prediction_error'
    s['admissible'][2]=True
    with pytest.raises(ValueError):forecast_rows(s)


@struct.dataclass
class Physical:
    failed: object
    eso: object
    rng: object

@struct.dataclass
class State:
    tick: object
    fault: object
    physical: object
    history: object
    filtered: object
    alpha: object

@struct.dataclass
class Carry:
    state: object
    gate: object
    relay: object
    clean_blocks: object


def test_jitted_preview_preserves_complete_origin_and_never_accepts_an_event():
    origin=Carry(State(j.int32(1992),j.bool_(False),Physical(j.bool_(False),j.array([.2,.3]),j.array([17,23],j.uint32)),j.array([4.,5.]),j.array([.03,-.02]),j.float32(.7)),j.array([.4]),j.array([.6]),j.int32(11))
    def step(c,p,path,speed,pulse,mode):
        # Wrong pulse or actor mutation turns into a fault, observable below.
        wrong=j.any(pulse!=0)|(mode!=1)|j.any(p['actor']['native']!=j.array([.3,-.4]))
        active=j.repeat(c.state.tick<2000,4)
        end=c.replace(state=c.state.replace(tick=j.minimum(c.state.tick+4,2000),history=c.state.history+j.any(active),fault=c.state.fault|wrong))
        return end,dict(active_tick=active,contact_failure=j.zeros(4,bool),nonfinite=j.zeros(4,bool),lower_fault=j.zeros(4,bool),policy_fault=j.zeros(4,bool),scored_tick_reward=j.where(active,-1.,0.))
    preview=jax.jit(make_preview_rollout(step,upper_steps=3))
    end,s=preview(origin,j.array([.3,-.4]),None,j.float32(1.2))
    assert int(s['observed_lower_ticks'])==8 and int(s['requested_lower_ticks'])==8 and int(s['allocated_lower_slots'])==12
    assert bool(s['admissible']) and float(s['original_user_cost'])==8
    np.testing.assert_array_equal(end.state.history,origin.state.history+2)
    for a,b in [(end.state.physical.eso,origin.state.physical.eso),(end.state.physical.rng,origin.state.physical.rng),(end.state.filtered,origin.state.filtered),(end.gate,origin.gate),(end.relay,origin.relay),(end.state.alpha,origin.state.alpha)]:
        np.testing.assert_array_equal(a,b)
    assert int(origin.state.tick)==1992  # Original actual carry was not advanced.


def test_invalid_candidate_and_forecast_shapes_are_rejected():
    with pytest.raises(ValueError):candidates([0.,np.nan],[0.,0.],CFG)
    with pytest.raises(ValueError):candidates([0.],[0.,0.],CFG)
    with pytest.raises(ValueError):choose([dict(status='survived')]*4)
