import numpy as np
import pytest
from sttw_control import tolerance_coordination as coordinator


@pytest.fixture
def interface():
    return dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2,speed_reference_bounds_m_s=[1.5,3.5],steer_reference_bound_rad=.35)


def test_native41_features_match_frozen_contract_for_all_hypotheses(interface):
    import jax.numpy as j
    from sttw_control.tolerance_predictor_pilot import feature_row
    obs=np.arange(351,dtype=np.float32)/100;obs[336]=1
    latent=np.array([[.2,-.3],[-.4,.5],[.6,-.7]],np.float32)
    actual=np.asarray(coordinator.candidate_features(j.asarray(obs),j.asarray(latent),.37,interface))
    cfg=dict(features=dict(full_indices=list(range(300,320))+list(range(337,351)),speed_indices=[304]))
    for a,alpha in enumerate((0.,.5,1.)):
        query=obs.copy();query[336]=alpha
        for h,(prep,force) in enumerate(((.2,-64),(.2,64),(.6,-64),(.6,64))):
            expected,_=feature_row(obs,query,latent[a],.37,dict(candidate_alpha=alpha,force_N=force,duration_s=.1,preparation_s=prep),interface,cfg)
            np.testing.assert_allclose(actual[a,h],expected,atol=2e-7)


def test_probability_averages_sigmoids_not_logits():
    params={}
    for layer,shape in enumerate(((41,16),(16,16),(16,1))):
        params[f'w{layer}']=np.zeros((2,3)+shape,np.float32)
        params[f'b{layer}']=np.zeros((2,3,shape[1]),np.float32)
    params['b2'][:,:]=np.array([[-5.],[0.],[1.]])
    p=np.asarray(coordinator.infer_probabilities(params,np.zeros((2,41),np.float32)))
    expected=np.mean(1/(1+np.exp(-np.array([-5.,0.,1.]))))
    np.testing.assert_allclose(p,expected,atol=1e-7)
    assert abs(expected-1/(1+np.exp(4/3)))>.1


def test_preview_covers_every_curvature_knot_and_endpoints():
    import jax.numpy as j
    s=j.array([0.,.01,.02,4.,6.,8.]);k=j.array([0.,9.,0.,0.,0.,0.])
    assert float(coordinator.preview_curvature(0.,s,k))==9
    # 6m preview endpoint interpolation can dominate interior knots.
    assert float(coordinator.preview_curvature(1.,j.array([0.,10.]),j.array([0.,10.])))==pytest.approx(7)


def test_user_priority_hysteresis_permission_and_degraded():
    import jax.numpy as j
    targets=j.array([[1.5,0.],[2.,0.],[2.5,0.]])
    args=(targets,.1,j.array([2.5,0.]),2.5)
    result=coordinator.choose_alpha(*args,j.array([.9,.9,.81]),True,False,1.)
    assert float(result['alpha'])==1 and not result['latched']
    result=coordinator.choose_alpha(*args,j.array([.9,.9,.81]),True,True,.5)
    assert float(result['alpha'])==.5 and result['latched']
    result=coordinator.choose_alpha(*args,j.array([.9,.9,.86]),True,True,.5)
    assert float(result['alpha'])==1 and not result['latched']
    result=coordinator.choose_alpha(*args,j.array([.1,.1,.1]),False,True,0.)
    assert float(result['alpha'])==0 and result['latched'] and result['publication_withheld']
    assert not result['degraded']
    result=coordinator.choose_alpha(*args,j.array([.1,.1,.1]),True,False,1.)
    assert float(result['alpha'])==.5 and result['degraded']


def test_task_guard_matches_geometry_cost_without_q_and_is_jittable():
    import jax
    import jax.numpy as j
    targets=j.array([[1.5,0.],[2.,0.],[3.,0.]])
    fn=jax.jit(lambda q:coordinator.choose_alpha(targets,.5,j.array([3.,0.]),3.,q,True,False,1.,use_q=False))
    low=fn(j.zeros(3));high=fn(j.ones(3))
    assert float(low['alpha'])==.5
    np.testing.assert_array_equal(low['task_mask'],[True,True,False])
    np.testing.assert_array_equal(low['alpha'],high['alpha'])
    expected=(np.array([1.5,2.,3.])-3.)**2+.05*(np.array([0.,.5,1.])-1)**2+.02*(np.array([0.,.5,1.])-1)**2
    np.testing.assert_allclose(low['proxy_cost'],expected,atol=1e-7)


def test_coordinate_clips_native_targets_before_geometry_check(interface):
    import jax
    import jax.numpy as j
    params={}
    for layer,shape in enumerate(((41,16),(16,16),(16,1))):
        params[f'w{layer}']=np.zeros((2,3)+shape,np.float32)
        params[f'b{layer}']=np.zeros((2,3,shape[1]),np.float32)
    params['b2'][:]=5
    fn=jax.jit(lambda latents:coordinator.coordinate(params,j.zeros(351),latents,0.,j.array([1.5,.34]),2.,0.,j.array([0.,10.]),j.array([0.,0.]),True,False,1.,interface))
    result=fn(j.array([[-10.,10.],[-10.,10.],[-10.,10.]]))
    np.testing.assert_allclose(result['targets'],np.tile([1.5,.35],(3,1)),atol=1e-7)
    assert np.asarray(result['target_clipped']).all()
    assert float(result['alpha'])==1


def test_export_loader_rejects_parameter_tamper(tmp_path):
    import json
    file=tmp_path/'parameters.npz';np.savez(file,w0=np.zeros((1,),np.float32))
    (tmp_path/'receipt.json').write_text(json.dumps(dict(files_sha256={'parameters.npz':coordinator.sha(file)})))
    np.testing.assert_array_equal(coordinator.load_export(tmp_path)['w0'],[0])
    np.savez(file,w0=np.ones((1,),np.float32))
    with pytest.raises(ValueError,match='export changed'):coordinator.load_export(tmp_path)


def test_no_permission_has_boolean_withheld_and_hold_state():
    import jax.numpy as j
    result=coordinator.choose_alpha(j.array([[2.,0.]]*3),0.,j.array([2.,0.]),2.,j.ones(3),False,False,.5)
    assert bool(result['publication_withheld']) and not bool(result['latched'])
    assert float(result['alpha'])==.5
    allowed=coordinator.choose_alpha(j.array([[2.,0.]]*3),0.,j.array([2.,0.]),2.,j.ones(3),True,False,.5)
    assert not bool(allowed['publication_withheld'])


def test_exact_thresholds_and_task_guard_equality():
    import jax.numpy as j
    targets=j.array([[1.,0.],[1.,0.],[1.,0.]])
    # Exact v²*k==3 is admitted; restore .85 and enter .8 use inclusive checks.
    result=coordinator.choose_alpha(targets,3.,j.array([1.,0.]),1.,j.array([.8,.8,.85]),True,True,.5)
    assert np.asarray(result['task_mask']).all() and float(result['alpha'])==1


def test_nonzero_float32_networks_match_torch_probabilities():
    import torch
    rng=np.random.default_rng(75)
    features=rng.uniform(-5,5,(128,41)).astype(np.float32);params={}
    for layer,shape in enumerate(((41,16),(16,16),(16,1))):
        params[f'w{layer}']=rng.normal(0,.7,(2,3)+shape).astype(np.float32)
        params[f'b{layer}']=rng.normal(0,.3,(2,3,shape[1])).astype(np.float32)
    expected=[]
    for arm in range(2):
        probabilities=[]
        for seed in range(3):
            x=torch.as_tensor(features)
            for layer in range(3):
                x=x@torch.as_tensor(params[f'w{layer}'][arm,seed])+torch.as_tensor(params[f'b{layer}'][arm,seed])
                if layer<2:x=torch.relu(x)
            probabilities.append(torch.sigmoid(x[:,0]).numpy())
        expected.append(np.mean(probabilities,axis=0))
    actual=np.asarray(coordinator.infer_probabilities(params,features))
    np.testing.assert_allclose(actual,np.asarray(expected),rtol=0,atol=2e-6)
