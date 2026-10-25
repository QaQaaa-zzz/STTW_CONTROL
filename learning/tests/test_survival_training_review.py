"""Training telemetry is stratified evidence, never an endpoint success claim."""
import numpy as np
import pytest


def test_case_identity_is_joined_before_stratifying_not_guessed_from_row_order():
    from sttw_control.survival_training_review import join_collection,summarize_strata
    cases=[dict(arm='task_only',route='left',speed=1.8,alpha=0.,force_N=0.,replicate=0),
           dict(arm='task_only',route='right',speed=1.5,alpha=1.,force_N=-32.,replicate=0)]
    def row(i,outcome):return dict(case_index=i,alpha=cases[i]['alpha'],force_N=cases[i]['force_N'],outcome=outcome,
        hold_ticks=0,actor_decisions=10,physical_ticks=40,task_scored_return=-10.,mean_true_speed=1.5,max_task_path_error=.2,full_pulse_exposure=True)
    data=dict(all_physical_tails_complete=True,case_summaries={'task_only':[row(1,'contact'),row(0,'survived')]})
    joined=join_collection(cases,data,update=7)
    assert joined[0]['route']=='right' and joined[1]['route']=='left'
    stats=summarize_strata(joined,dt_s=.005)
    assert [s['survived'] for s in stats]==[0,1]
    data['case_summaries']['task_only'][0]['case_index']=0
    with pytest.raises(ValueError):join_collection(cases,data,update=7)


def test_unknown_and_missing_case_rows_cannot_be_silently_counted_as_training_failures():
    from sttw_control.survival_training_review import join_collection
    cases=[dict(arm='a',route='left',speed=1.,alpha=0.,force_N=0.,replicate=0)]
    data=dict(all_physical_tails_complete=True,case_summaries={'a':[]})
    with pytest.raises(ValueError):join_collection(cases,data,update=1)
    data['case_summaries']['a']=[dict(case_index=0,alpha=0.,force_N=0.,outcome='unknown')]
    with pytest.raises(ValueError):join_collection(cases,data,update=1)


def test_gradient_geometry_uses_real_gradients_without_populating_or_updating_parameters():
    import torch
    from sttw_control.survival_training_review import gradient_geometry
    p=torch.nn.Parameter(torch.tensor([2.,3.]))
    got=gradient_geometry(p[0],-p[0]+p[1],[p])
    assert got['task_norm']==pytest.approx(1.) and got['physical_norm']==pytest.approx(np.sqrt(2))
    assert got['cosine']==pytest.approx(-1/np.sqrt(2)) and got['combined_norm']==pytest.approx(1.)
    assert p.grad is None and torch.equal(p,torch.tensor([2.,3.]))
