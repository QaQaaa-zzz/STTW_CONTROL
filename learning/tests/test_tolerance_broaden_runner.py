import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.native_goal_run import tolerance_inventory
from sttw_control.tolerance_coordination import speedcap_latent,native_offsets


def test_broaden_inventory_complete_and_strict():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_tolerance_broaden120.json').read_text())
    rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==120
    groups={}
    for r in rows:groups.setdefault((r['bank_index'],r['route'],r['event']),[]).append(r)
    assert len(groups)==24 and all(len(v)==5 for v in groups.values())
    assert {r['onset_s'] for r in rows}=={2.2}
    p['bank_indices']=[2,5]
    with pytest.raises(ValueError):tolerance_inventory(p)


def test_speedcap_active_native_authority_and_steering_preserved():
    import jax
    import jax.numpy as j
    interface=dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.2,speed_reference_bounds_m_s=[1.5,3.5])
    z=j.array([.2,.73]);nominal=j.array([2.8,.2])
    capped,log=jax.jit(lambda z:speedcap_latent(z,nominal,j.array(1/1.8),interface))(z)
    assert capped[1]==z[1]
    assert log['active']
    np.testing.assert_allclose(log['reachable_target_m_s'],np.sqrt(3*1.8),atol=1e-6)
    assert abs(float(np.tanh(capped[0])))<1
    plain,straight=speedcap_latent(z,nominal,j.array(0.),interface)
    assert not straight['active'] and np.isinf(straight['cap_m_s'])
    np.testing.assert_allclose(plain,z,atol=1e-6)
    saturated,s=speedcap_latent(z,nominal,j.array(3.),interface)
    assert s['requested_target_m_s']<s['reachable_target_m_s']
    np.testing.assert_allclose(s['reachable_target_m_s'],1.800001,atol=1e-6)


def test_five_arm_dispatch_and_publication_permission():
    import jax
    import jax.numpy as j
    from sttw_control.native_goal_run import select_broaden_arm
    decision=dict(alpha=j.asarray(0.),latched=j.asarray(True))
    alpha,latch=jax.jit(jax.vmap(lambda mode:select_broaden_arm(mode,decision,True,.5,False)))(j.arange(5))
    np.testing.assert_array_equal(alpha,[0.,0.,.5,1.,1.])
    np.testing.assert_array_equal(latch,[True,True,False,False,False])
    alpha,latch=jax.jit(jax.vmap(lambda mode:select_broaden_arm(mode,decision,False,.5,True)))(j.arange(5))
    np.testing.assert_array_equal(alpha,[.5]*5)
    assert np.all(latch)
