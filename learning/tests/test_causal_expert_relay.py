import jax.numpy as j
import numpy as np
from sttw_control.causal_reference_baseline import initial_gate
from sttw_control.causal_expert_relay import relay_action

CONFIG={'residual_interface':dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.2)}


class Actor:
    def apply(self,weights,obs):
        return weights+j.stack([obs[336],obs[0]])


def obs(signal=-.03):
    return j.zeros(351).at[0].set(.4).at[334:336].set(1.).at[318].set(-signal).at[340].set(-1.5).at[345].set(.5)


def test_teacher_context_changes_only_its_query_and_user_history_stays_intact():
    observation=obs();original=np.asarray(observation).copy()
    p=dict(base=j.array([-.3,.2]),teacher=j.array([.1,-.4]),teacher_alpha=j.float32(1.))
    action,gate,record=relay_action(Actor(),Actor(),p,observation,j.zeros(2),j.int32(300),initial_gate(),CONFIG,j.bool_(False))
    np.testing.assert_allclose(action,[1.1,0.],atol=1e-7)
    np.testing.assert_array_equal(observation,original)
    assert float(observation[336])==0. and record['override_active'] and int(gate.start_tick)==300
    np.testing.assert_allclose(record['base_latent'],[-.3,.6],atol=1e-7)


def test_same_window_handback_and_no_second_detection():
    p=dict(base=j.array([-.3,.2]),teacher=j.array([.1,-.4]),teacher_alpha=j.float32(1.))
    _,gate,_=relay_action(Actor(),Actor(),p,obs(),j.zeros(2),j.int32(300),initial_gate(),CONFIG,j.bool_(False))
    for tick,active in [(376,True),(380,False),(800,False)]:
        action,end,record=relay_action(Actor(),Actor(),p,obs(),j.zeros(2),j.int32(tick),gate,CONFIG,j.bool_(False))
        assert bool(record['override_active'])==active and not record['detected']
        np.testing.assert_allclose(action,[1.1,0.] if active else [-.3,.6],atol=1e-7)
        assert int(end.start_tick)==300


def test_disabled_and_opposite_first_event_never_publish_teacher_action():
    p=dict(base=j.array([-.3,.2]),teacher=j.array([.1,-.4]),teacher_alpha=j.float32(1.))
    action,gate,record=relay_action(Actor(),Actor(),p,obs(),j.zeros(2),j.int32(4),initial_gate(),CONFIG,j.bool_(True))
    np.testing.assert_array_equal(action,j.zeros(2));assert not gate.fired and not record['override_active']
    action,gate,record=relay_action(Actor(),Actor(),p,obs(.03),j.zeros(2),j.int32(4),initial_gate(),CONFIG,j.bool_(False))
    assert gate.fired and not record['override_active']
    action,_,record=relay_action(Actor(),Actor(),p,obs(),j.zeros(2),j.int32(8),gate,CONFIG,j.bool_(False))
    assert not record['override_active'];np.testing.assert_allclose(action,[-.3,.6],atol=1e-7)
