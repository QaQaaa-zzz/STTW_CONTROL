from sttw_control.sequence_disturbance_protocol import probe_cases


def test_only_two_nominally_selected_processes_get_same_four_hidden_events():
    templates=[dict(event_id=f't{t}_{force}',force_N=force,alpha=a,state_alpha=a,source_batch_slot=6+int(a>0),pulse_onset_task_tick=t) for a in [0.,.5] for t in [420,520] for force in [-32.,32.]]
    rows=probe_cases(templates)
    assert len(rows)==16 and len({x['case_id'] for x in rows})==16
    assert {x['role'] for x in rows}=={'sequence_26','constant_slow'}
    assert {x['stage'] for x in rows}=={'sequence_probe'}
    assert all(x['force_N']!=0 and x['alpha']==x['state_alpha'] for x in rows)
    assert all('role' not in x for x in templates)
    for role in ['sequence_26','constant_slow']:
        assert {(x['alpha'],x['event_id']) for x in rows if x['role']==role}=={(x['alpha'],x['event_id']) for x in templates}
