from sttw_control.full_task_sequence_protocol import sequence_cases


def test_37_processes_run_separately_for_two_original_user_alphas():
    templates=[dict(case_id='parent/'+str(a),role='parent',actor_role='parent',alpha=a,state_alpha=a,source_batch_slot=6+i,stage='nominal',force_N=0.) for i,a in enumerate([0.,.5])]
    names=[f'process_{i}' for i in range(37)]
    cases=sequence_cases(templates,names)
    assert len(cases)==74 and len({r['case_id'] for r in cases})==74
    assert {r['stage'] for r in cases}=={'sequence_nominal'}
    assert all(r['alpha']==r['state_alpha'] and r['force_N']==0. and r['profile']==0 and not r['zero_mode'] for r in cases)
    assert all(r['source_batch_slot']==6+int(r['alpha']>.1) for r in cases)
    assert len({r['actor_role'] for r in cases})==37
    assert all('sequence_index' not in r for r in templates)
