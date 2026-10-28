import copy
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.tolerance_predictor_pilot import validate_user_mix_config,validate_user_mix_datasets,discordant_pairs,temporal_diagnostics,user_mix_selection


def config():return json.loads((Path(__file__).parents[1]/'configs/h75_tolerance_predictor_user_mix.json').read_text())


def datasets():
    result=[]
    for early,late,user,falls in [(3,2,4,15),(0,5,6,8)]:
        meta=[dict(case_id=f'early{early}/{i}',source_group_id=f'early{early}/{i//3}',bank_index=early,period='early',alpha=[0.,.5,1.][i%3],preparation_s=.2,force_N=64.) for i in range(72)]
        meta += [dict(case_id=f'oldlate{late}/{i}',source_group_id=f'oldlate{late}/{i//12}',bank_index=late,period='late',alpha=[0.,.5,1.][i%3],preparation_s=.2,force_N=64.) for i in range(24)]
        for route in ('S_left_90_R1p8','S_right_90_R1p8','S_left_60_R2p2','S_right_60_R2p2'):
            for origin,sourceuser in [('user0_last',0.),('userhalf_last',.5)]:
                group=f'bank{user}/{route}/{origin}/task6'
                for a in (0.,.5,1.):
                    for prep in (.2,.6):
                        for force in (-64.,64.):meta.append(dict(case_id=f'{group}/{a}/{prep}/{force}',source_group_id=group,bank_index=user,period='userlate',alpha=a,preparation_s=prep,force_N=force,source_user_alpha=sourceuser))
        y=np.ones(192,np.float32);y[:falls]=0
        result.append(dict(full=np.zeros((192,41),np.float32),y=y,meta=meta))
    return result


def test_budget_fixed_last_single_arm():
    c=config();validate_user_mix_config(c)
    assert c['budget']['maximum_optimizer_updates']==3000
    assert c['ranking']['arms']==['bce_rank'] and c['ranking']['selectors']==['last']
    c['ranking']['selectors']=['balanced_brier']
    with pytest.raises(ValueError):validate_user_mix_config(c)


def test_fixed_source_bank_split_preserves_early_failures_and_no_late_rank():
    train,dev=datasets();validate_user_mix_datasets(train,dev)
    assert all(train['meta'][i]['period']=='early' for pair in discordant_pairs(train['y'],train['meta']) for i in pair)
    dev['y'][0]=1
    with pytest.raises(ValueError):validate_user_mix_datasets(train,dev)


def test_source_user_restore_diagnostics_and_separate_period_scores():
    train,dev=datasets();p=np.ones(192)*.9
    users=[i for i,m in enumerate(dev['meta']) if m['period']=='userlate' and m['alpha']==m['source_user_alpha']]
    p[users]=.7
    d=temporal_diagnostics(dev['y'],p,dev['meta'])
    assert d['userlate_user_rows']==32 and d['userlate_user_rejected_085']==32
    score,parts=user_mix_selection(dev['y'],p,dev['meta'])
    assert set(parts)=={'early_brier','late_brier','userlate_brier'}
    assert np.isfinite(score)


def test_cannot_drop_or_duplicate_sourcegroup():
    train,dev=datasets();dev['meta'][-1]=copy.deepcopy(dev['meta'][-2])
    with pytest.raises(ValueError):validate_user_mix_datasets(train,dev)



def test_prior_group_weights_normalized_and_original_failure_prior_restored():
    from sttw_control.tolerance_predictor_pilot import user_mix_bce_weights
    train,_=datasets();weights=user_mix_bce_weights(train['meta'],{'early':.75,'late':.125,'userlate':.125})
    assert weights.sum()==pytest.approx(1.)
    assert weights[:72].sum()==pytest.approx(.75)
    assert weights[72:96].sum()==pytest.approx(.125)
    assert weights[96:].sum()==pytest.approx(.125)
    assert weights[train['y']==0].sum()==pytest.approx(15/96)
    assert np.unique(weights[:72]).size==np.unique(weights[72:96]).size==np.unique(weights[96:]).size==1


def test_raw_bce_unchanged_and_weighted_objective_matches_direct_sum():
    import torch
    from sttw_control.tolerance_predictor_pilot import bce_objective,user_mix_bce_weights
    train,_=datasets();logits=torch.linspace(-2,3,192,requires_grad=True);labels=torch.as_tensor(train['y'])
    raw,unweighted=bce_objective(logits,labels)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
    assert raw is unweighted
    torch.testing.assert_close(raw,expected,rtol=0,atol=0)
    weights=torch.as_tensor(user_mix_bce_weights(train['meta'],{'early':.75,'late':.125,'userlate':.125}))
    raw,weighted=bce_objective(logits,labels,weights)
    direct=sum(torch.nn.functional.binary_cross_entropy_with_logits(logits[a:b],labels[a:b])*total for a,b,total in [(0,72,.75),(72,96,.125),(96,192,.125)])
    torch.testing.assert_close(weighted,direct)
    weighted.backward()
    torch.testing.assert_close(logits.grad,(logits.detach().sigmoid()-labels)*weights)


def test_only_one_declared_prior_variant_allowed():
    c=json.loads((Path(__file__).parents[1]/'configs/h75_tolerance_predictor_user_mix_prior.json').read_text());validate_user_mix_config(c)
    c['bce_group_weights']['early']=.8
    with pytest.raises(ValueError):validate_user_mix_config(c)
