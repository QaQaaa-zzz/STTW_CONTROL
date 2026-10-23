"""Exercise real policy_step aggregation with synthetic bounded tick costs."""
import copy
import jax.numpy as j
import numpy as np
import pytest
from test_path_terminal_contract import minimal_env, State, Physical
from test_tolerance_reward_integration import BASE, configured
from sttw_control.path_command_reward import failure_reward


@pytest.mark.parametrize('mode',['baseline','static','dynamic'])
def test_four_lower_cost_envelopes_match_one_upper_failure_bound(mode):
    cfg=copy.deepcopy(BASE) if mode=='baseline' else configured()
    if mode!='baseline':cfg['path_reward']['tolerance']['method']=mode
    env=minimal_env();env.config=cfg
    c=cfg['path_reward'];bound=sum(c['component_caps'].values())
    assert np.isclose(bound,c['failure_cost_rate_bound'])
    # Caps form a conservative envelope, not a jointly attainable vehicle state.
    # offset_rate is added by policy_step itself; every other cap is in tick.
    tick_rate=bound-c['component_caps']['offset_rate']
    def tick(s,z,path,speed,**kwargs):
        log={**env.zero_log,'active_tick':j.bool_(True),
             'tick_reward':j.asarray(-c['scale']*.005*tick_rate)}
        return s.replace(tick=s.tick+1,offsets=s.offsets+10.,domain_exit=s.trigger),log
    env.tick=tick
    state=State(Physical(j.bool_(False)),j.bool_(False),j.bool_(False),
                j.zeros(2),j.int32(0),j.bool_(False))
    end,log=env.policy_step(state,j.zeros(2),None,2.,20.)
    assert int(end.tick)==4 and int(log['active_tick'].sum())==4
    expected=-c['scale']*cfg['timing']['upper_dt_s']*bound
    np.testing.assert_allclose(log['scored_tick_reward'].sum(),expected,rtol=2e-6)
    # One remaining policy interval: failure includes exactly that same envelope
    # plus the explicit failure_extra, rather than charging only one lower tick.
    np.testing.assert_allclose(failure_reward(1,cfg),expected-c['failure_extra'],rtol=2e-6)
