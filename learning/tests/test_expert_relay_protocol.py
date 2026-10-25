from sttw_control.expert_relay_protocol import relay_cases


def test_two_fixed_teachers_share_same_five_events_and_user_alpha_zero():
    events=[dict(event_id='zero',force_N=0.),*[dict(event_id=str(i),force_N=32.) for i in range(4)]]
    templates=[dict(r,alpha=0.,state_alpha=0.,profile=0,source_batch_slot=6) for r in events]
    rows=relay_cases(templates)
    assert len(rows)==10 and len({r['case_id'] for r in rows})==10
    assert {r['role'] for r in rows}=={'relay_dual0','relay_dual1'}
    assert all(r['alpha']==r['state_alpha']==0. and r['source_batch_slot']==6 for r in rows)
    assert {r['teacher_alpha_context'] for r in rows}=={0.,1.}
    for role in ['relay_dual0','relay_dual1']:
        assert {r['event_id'] for r in rows if r['role']==role}=={r['event_id'] for r in templates}
    assert all('teacher_alpha_context' not in r for r in templates)
