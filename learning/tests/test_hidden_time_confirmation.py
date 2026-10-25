import pytest
from sttw_control.hidden_time_confirmation import timing_cases


def test_route_matrix_pairs_times_directions_and_retains_one_actual_zero():
    rows=timing_cases('right90_R1p1',speed=1.5,world_direction=[.6,.8,0.])
    assert len(rows)==55 and len({r['case_id'] for r in rows})==55
    zero=[r for r in rows if r['role']=='zero_middle'];assert len(zero)==5
    assert all(r['alpha'] is None and r['state_alpha']==0. and r['zero_mode'] for r in zero)
    parent=[r for r in rows if r['role']=='parent'];assert len(parent)==10
    for alpha in [.5,.75]:
        r=[x for x in parent if x['alpha']==alpha]
        assert {(x['force_N'],x['pulse_onset_task_tick']) for x in r if x['force_N']}=={(f,t) for f in [-32.,32.] for t in [380,580]}
    assert all(r['profile']==5 for r in rows if r['role']=='causal_eso_allalpha')
    assert all(r['state_alpha']==r['alpha'] for r in rows if r['alpha'] is not None)
    assert all(r['reference_speed_floor_m_s']==1.5 for r in rows if r['role']=='floor')
    with pytest.raises(ValueError):timing_cases('unknown',speed=1.5,world_direction=[1,0,0])


def test_historical_source_can_be_verified_from_exact_archive_without_changing_live(tmp_path):
    from sttw_control.hidden_time_confirmation import historical_source_receipt
    from sttw_control.preference_best import digest
    live=tmp_path/'live.py';live.write_text('new comparator')
    archive=tmp_path/'archive';archive.mkdir();old=archive/'004_live.py';old.write_text('old training')
    oldsha=digest(old)
    receipt=historical_source_receipt({str(live):oldsha},[archive])
    assert receipt[0]['verified_path']==str(old) and receipt[0]['live_matches'] is False
    assert live.read_text()=='new comparator'
    old.write_text('corrupt')
    with pytest.raises(ValueError):historical_source_receipt({str(live):oldsha},[archive])
