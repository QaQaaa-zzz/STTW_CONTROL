"""Frozen route-time panel contract; no simulation or model inference."""
import json
from pathlib import Path
import jax.numpy as j
import numpy as np
import pytest
from flax import struct
from sttw_control.native_goal_run import (five_alpha_online_inventory,deterministic_route_onsets,
    audit_five_alpha_online_initial,force_pulses,planning_chunk_steps)


def config():return json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online_disturbed_users.json').read_text())


def test_declared_deterministic_onsets_full_seventy_and_physical_pulse_clock():
    p=config();rows=five_alpha_online_inventory(p)
    expected={'straight_v2p6':2.12,'left70_R1p9':2.40,'right70_R1p9':1.80,'left130_R2p7':2.64,'right130_R2p7':1.96,'S_left_90_R1p8':2.66,'S_right_90_R1p8':2.28}
    assert 'onset_s' not in p and p['onset_by_route_s']==expected
    assert deterministic_route_onsets(list(reversed(expected)),20261029)==expected
    assert len(rows)==len({r['case_id'] for r in rows})==70
    assert sum(planning_chunk_steps(10.,10.))*4*len(rows)==140000
    for route,time in expected.items():
        rr=[r for r in rows if r['route']==route]
        assert len(rr)==10 and {r['onset_s'] for r in rr}=={time}
        assert {(r['user_alpha'],r['force_N']) for r in rr}=={(a,f) for a in [0,.25,.5,.75,1] for f in [-64,64]}
        assert all(r['alpha']==r['user_alpha']==r['query_alpha'] for r in rr)
    pulses=np.asarray(force_pulses(rows,3.5))
    np.testing.assert_allclose(pulses[:,0],[3.5+r['onset_s'] for r in rows])
    np.testing.assert_allclose(pulses[:,1]-pulses[:,0],.15)
    nominal=json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online_nominal_users_guard.json').read_text())
    for k in ['route_contexts','predictor_training','export_manifest_sha256','guard_fallback','alpha_dwell_s','candidate_alphas','hypothetical_duration_s','planning_frequency_hz']:
        assert p[k]==nominal[k]


@pytest.mark.parametrize('change',['seed','seed_bool','map','missing_route','algorithm','global_onset','grid_bool','users','budget','count','guard'])
def test_reject_unfrozen_or_malformed_panel(change):
    p=config()
    if change=='seed':p['onset_seed']+=1
    elif change=='seed_bool':p['onset_seed']=True
    elif change=='map':p['onset_by_route_s']['left70_R1p9']+=.02
    elif change=='missing_route':p['onset_by_route_s'].pop('left70_R1p9')
    elif change=='algorithm':p['onset_algorithm']='another PRNG'
    elif change=='global_onset':p['onset_s']=2.2
    elif change=='grid_bool':p['candidate_alphas'][0]=False
    elif change=='users':p['user_alphas']=[0,.5,1]
    elif change=='budget':p['maximum_lower_ticks']+=1
    elif change=='count':p['expected_cases']=69
    else:p['guard_fallback']=False
    with pytest.raises(ValueError):five_alpha_online_inventory(p)


def test_ten_lane_reset_groups_keep_full_history_and_explicit_users():
    @struct.dataclass
    class Physical:
        physical_tick: object
    @struct.dataclass
    class State:
        alpha: object
        tick: object
        physical: object
        history: object
    @struct.dataclass
    class Carry:
        state: object
        hidden: object
    rows=five_alpha_online_inventory(config());users=j.array([r['alpha'] for r in rows]);hidden=j.repeat(j.arange(7),10)
    initial=Carry(State(users,j.zeros(70,j.int32),Physical(j.full(70,700,j.int32)),hidden),hidden+10)
    receipt=audit_five_alpha_online_initial(initial,rows,700,panel_kind='disturbed_five_user')
    assert receipt['alpha_neutralized_only'] and receipt['initial_hold_until_tick']==0
    np.testing.assert_array_equal(initial.state.alpha,users)
    bad=initial.replace(hidden=initial.hidden.at[1].set(999))
    with pytest.raises(ValueError,match='neutralization'):audit_five_alpha_online_initial(bad,rows,700,panel_kind='disturbed_five_user')
