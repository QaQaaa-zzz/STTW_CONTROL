from sttw_control.geometric_reacquisition_run import extension_cases


def test_cases_reference_complete_states_without_changing_clock_alpha_or_floor():
    states=[dict(source_case_id='floor0',source_task_tick=2000,physical_tick=2700,user_state_alpha=0.,original_case_alpha=0.,reference_speed_floor_m_s=1.5,requested_speed_m_s=1.5),dict(source_case_id='zero',source_task_tick=2000,physical_tick=2700,user_state_alpha=0.,original_case_alpha=None,reference_speed_floor_m_s=1.,requested_speed_m_s=1.5)]
    rows=extension_cases(states)
    assert len(rows)==6 and len({r['case_id'] for r in rows})==6
    assert {r['role'] for r in rows}=={'live_zero','geo_fixed','geo_adaptive'}
    assert all(r['source_task_tick']==2000 and r['physical_tick']==2700 and r['alpha']==0. for r in rows)
    assert all(r['suffix_force_N']==0 and r['horizon_s']==22 and r['suffix_s']==12 for r in rows)
    assert all(r['reference_speed_floor_m_s']==states[r['source_state_index']]['reference_speed_floor_m_s'] for r in rows)
