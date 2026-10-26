import pytest

from sttw_control.middle_lineage_comparison import IDS, inventory


def test_dual_endpoint_selection_preserves_alpha_and_matched_events():
    original=[]
    for i in range(12):
        for alpha in [0.,1.]:
            original.append(dict(role='goal_full_feedback',case_id=f'c{i}/a{alpha:g}/F0/goal_full_feedback',
                context=f'c{i}',alpha=alpha,force_N=0.,source_keep_channels=[True,True]))
    cases=inventory(original)
    assert len(cases)==72
    for row in cases:
        assert row['force_N']==0.
        assert row['source_keep_channels']==[True,True]
        if row['role']=='lineage_dual':
            assert row['model_key']==IDS[0 if row['alpha']==0 else 1]
    assert {row['role'] for row in original}=={'goal_full_feedback'}
    with pytest.raises(ValueError,match='exact24'):
        inventory(original[:-1])
