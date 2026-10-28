import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.native_goal_run import tolerance_inventory
from sttw_control.tolerance_coordination import fold_input_mask,candidate_features,infer_probabilities,candidate_probabilities,export_predictors

KEEP=[4]+list(range(34,41))


def test_exact60_frozen_conditions_allusers_only_new_model():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_speed_query_online60.json').read_text());rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==60
    assert p['maximum_lower_ticks']==120000 and p['comparison_total_records']==260 and p['cached_physical_cases']==200
    assert all(sum(r['alpha']==a for r in rows)==20 for a in (0.,.5,1.))
    assert {r['method'] for r in rows}=={'speed_query'}
    assert {r['bank_index'] for r in rows}=={1,7}
    p['allow_masked_export']=False
    with pytest.raises(ValueError):tolerance_inventory(p)


def test_mask_export_cannot_silently_opt_in(tmp_path):
    f=tmp_path/'masked.json';f.write_text(json.dumps(dict(format='sttw_frozen_speed_query_last_v1')))
    with pytest.raises(ValueError,match='explicit opt-in'):export_predictors(f,tmp_path/'q')


def test_three_seed_folded_jax12queries_match_masked_torch_and_ignore_removed_state():
    import torch
    import jax
    import jax.numpy as j
    from sttw_control.tolerance_predictor_pilot import make_model,predict
    params={};models=[]
    for seed in (0,1,2):
        torch.manual_seed(seed);model=make_model(41);models.append(model)
        state=model.state_dict()
        for layer,index in enumerate((0,2,4)):
            w=state[f'{index}.weight'].numpy().T;b=state[f'{index}.bias'].numpy()
            if seed==0:params[f'w{layer}']=np.empty((1,3)+w.shape,np.float32);params[f'b{layer}']=np.empty((1,3)+b.shape,np.float32)
            params[f'w{layer}'][0,seed]=w;params[f'b{layer}'][0,seed]=b
    original=params['w0'].copy();params['w0']=fold_input_mask(params['w0'],KEEP)
    np.testing.assert_array_equal(params['w0'][...,KEEP,:],original[...,KEEP,:])
    assert np.all(params['w0'][...,[i for i in range(41) if i not in KEEP],:]==0)
    rng=np.random.default_rng(67);obs=j.array(rng.normal(size=351),j.float32);latents=j.array(rng.normal(size=(3,2)),j.float32);interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    x=np.asarray(candidate_features(obs,latents,.6,interface)).reshape(12,41)
    expected=np.mean([predict(model,x,feature_mask_keep_indices=KEEP) for model in models],axis=0)
    actual=np.asarray(jax.jit(lambda x:infer_probabilities(params,x))(j.array(x)))[0]
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)
    actual4=np.asarray(candidate_probabilities(params,obs,latents,.6,interface))[0]
    np.testing.assert_allclose(actual4.min(-1),expected.reshape(3,4).min(-1),atol=2e-6,rtol=0)
    changed=obs.at[j.array([i for i in list(range(300,320))+list(range(337,351)) if i!=304])].add(100.)
    np.testing.assert_array_equal(actual4,np.asarray(candidate_probabilities(params,changed,latents,.6,interface))[0])


def test_native_initial_layout60three():
    f=Path(__file__).parents[1]/'cli/audit_native_goal_initial.py';spec=importlib.util.spec_from_file_location('_speedquery_initial_test',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=json.loads((Path(__file__).parents[1]/'configs/h75_speed_query_online60.json').read_text())
    assert m.confirmation_initial_layout(dict(mode=p['mode'],protocol=p))==(60,3)
