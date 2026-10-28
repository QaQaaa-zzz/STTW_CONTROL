"""Frozen online panel identity and arm/publication mapping, no physics."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.native_goal_run import tolerance_inventory,select_tolerance_arm


def protocol():
    return json.loads((Path(__file__).parents[1]/'configs/h75_tolerance_online120.json').read_text())


def test_panel_is_twenty_paired_conditions_six_arms_and_exact_budget():
    p=protocol();rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==120
    assert len(rows)*p['horizon_s']/.005==p['maximum_lower_ticks']==240000
    groups={}
    for row in rows:groups.setdefault((row['bank_index'],row['route'],row['event']),[]).append(row)
    assert len(groups)==20
    for group in groups.values():
        assert [r['mode'] for r in group]==p['arms']
        assert len({(r['force_N'],r['onset_s'],r['duration_s'],r['initial_fingerprint'],r['path_key']) for r in group})==1
        assert {r['alpha'] for r in group}=={1.}
    assert {r['onset_s'] for r in rows if r['force_N']}=={1.6,3.4}


@pytest.mark.parametrize('key,value',[('bank_indices',[0,1]),('expected_cases',100),('maximum_lower_ticks',240001),('onsets_s',[2.2,3.4]),('optimizer_updates',1),('maximum_replans',2),('force_frame','body')])
def test_protocol_drift_rejected(key,value):
    p=copy.deepcopy(protocol());p[key]=value
    with pytest.raises(ValueError):tolerance_inventory(p)


def test_all_arm_choices_and_withheld_publication_preserve_sidecar():
    import jax
    import jax.numpy as j
    mode=j.arange(6)
    choose=jax.jit(jax.vmap(select_tolerance_arm,in_axes=(0,None,None,None,None,None,None)))
    decision=dict(alpha=j.array(0.),latched=j.array(True))
    alpha,latch=choose(mode,decision,.5,True,True,1.,False)
    np.testing.assert_array_equal(alpha,[0,0,0,.5,.5,1])
    np.testing.assert_array_equal(latch,[1,1,1,1,0,0])
    alpha,latch=choose(mode,decision,.5,False,False,.5,True)
    np.testing.assert_array_equal(alpha,np.full(6,.5))
    np.testing.assert_array_equal(latch,np.ones(6,bool))
