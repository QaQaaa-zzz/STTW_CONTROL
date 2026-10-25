from sttw_control.middle_lineage_evaluation import historical_cases


def test_history_uses_both_fixed_endpoints_and_reuses_only_exact_parent_zero_alpha():
    events=[dict(event_id='zero',force_N=0.),
            *[dict(event_id=f'{t}/{f}',force_N=f) for t in [420,520] for f in [-32.,32.]]]
    templates=[dict(r,role='parent',actor_role='parent',alpha=0.,state_alpha=0.,
                    source_batch_slot=6,profile=0,zero_mode=False) for r in events]
    rows=historical_cases(templates)
    assert len(rows)==35 and len({r['case_id'] for r in rows})==35
    assert {(r['role'],r['alpha']) for r in rows}=={
        ('dual',0.),('dual',1.),('bc',0.),('bc',1.),('v3',0.),('v3',1.),('parent',1.)}
    assert {r['actor_role'] for r in rows if r['role']=='dual'}=={'dual_a0','dual_a1'}
    assert all(r['state_alpha']==r['alpha'] and r['source_batch_slot']==6 for r in rows)
    assert all(r['stage']=='endpoints' and r['profile']==0 and not r['zero_mode'] for r in rows)
