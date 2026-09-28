import copy
import pytest
from sttw_control.tracking_diagnostics import comparison_task_identity

def test_endpoint_comparison_only_ignores_fixed_preference_labels():
    a={'priority':{'fixed_alpha':0.,'validation_alphas':[0.],'randomize_alpha':False},'observation':{'include_priority':False},'controller':{'fixed_roll_reference':.12}}
    b=copy.deepcopy(a);b['priority'].update(fixed_alpha=1.,validation_alphas=[1.])
    assert comparison_task_identity(a,independent=True)==comparison_task_identity(b,independent=True)
    assert comparison_task_identity(a)!=comparison_task_identity(b)
    b['controller']['fixed_roll_reference']=.13
    assert comparison_task_identity(a,independent=True)!=comparison_task_identity(b,independent=True)
    assert a['priority']['fixed_alpha']==0.

def test_independent_comparison_rejects_hidden_random_or_conditioned_alpha():
    for observation,random in [(True,False),(False,True)]:
        with pytest.raises(ValueError):
            comparison_task_identity({'priority':{'randomize_alpha':random},'observation':{'include_priority':observation}},independent=True)

def test_baseline_reuse_rejects_model_change_and_residual_source():
    import numpy as np
    from sttw_control.tracking_diagnostics import validate_baseline_reuse
    cfg={'priority':{'fixed_alpha':0.,'randomize_alpha':False},'observation':{'include_priority':False}}
    data={'config':cfg,'trace':{'effective_action':np.zeros((2,2))}}
    decl={'config':cfg,'seed':1,'model':{'xml':'a'},'controller':'baseline','policy':None}
    validate_baseline_reuse(data,decl,cfg,{'xml':'a'},1)
    with pytest.raises(ValueError):validate_baseline_reuse(data,decl,cfg,{'xml':'b'},1)
    with pytest.raises(ValueError):validate_baseline_reuse(data,{**decl,'controller':'residual'},cfg,{'xml':'a'},1)
    data['trace']['effective_action'][1,0]=.1
    with pytest.raises(ValueError):validate_baseline_reuse(data,decl,cfg,{'xml':'a'},1)
