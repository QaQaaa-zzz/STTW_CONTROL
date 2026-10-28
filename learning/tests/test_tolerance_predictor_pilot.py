import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control import tolerance_predictor_pilot as pilot


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/h75_tolerance_predictor_pilot.json').read_text())


def test_features_use_native_latest_frame_no_useralpha_and_body_fixed_hypothetical_force():
    cfg=config();source=np.arange(351,dtype=np.float32)/100.;source[336]=1.
    query=source.copy();query[336]=.5
    case=dict(candidate_alpha=.5,force_N=64.,duration_s=.1,preparation_s=.2)
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    full,speed=pilot.feature_row(source,query,[.2,-.3],np.pi/2,case,interface,cfg)
    assert full.shape==(41,) and speed.shape==(8,)
    np.testing.assert_array_equal(full[:34],source[list(range(300,320))+list(range(337,351))])
    assert speed[0]==source[304]
    np.testing.assert_allclose(full[-7:],[.5,np.tanh(.2)*.25,np.tanh(-.3),1.,0.,1.,1/3],atol=1e-6)
    query[304]+=.01
    with pytest.raises(AssertionError):pilot.feature_row(source,query,[.2,-.3],0.,case,interface,cfg)


def test_native_candidate_target_asymmetric_scales_and_negative_speed():
    c=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    np.testing.assert_allclose(pilot.native_target([-1.,.5],c),[-np.tanh(1.),np.tanh(.5)*.2],atol=1e-7)


def test_single_class_proper_scores_and_ranking_denominator_are_truthful():
    meta=[dict(source_group_id='s',preparation_s=0.,force_N=64.)]*3
    r=pilot.metrics([1,1,1],[.8,.9,.7],meta)
    assert r['roc_auc'] is None and r['pr_auc'] is None
    assert r['discordant_pair_ranking']['pairs']==0
    r=pilot.metrics([0,1,1],[.8,.9,.7],meta)
    assert r['discordant_pair_ranking']==dict(discordant_groups=1,pairs=2,score_sum=1.,mean=.5)
    assert r['roc_auc']==.5 and r['confusion']['fp']==1


def test_auc_handles_ties_without_random_ranking():
    meta=[dict(source_group_id='s',preparation_s=0.,force_N=64.)]*2
    r=pilot.metrics([0,1],[.5,.5],meta)
    assert r['roc_auc']==.5
    assert r['discordant_pair_ranking']['mean']==.5


def test_fit_development_guard_and_eval_seal_guard_precede_label_open(tmp_path,monkeypatch):
    forbidden=tmp_path/'analysis_final'
    with pytest.raises(ValueError,match='development-only'):
        pilot.load_dataset(forbidden,'development',config(),audit_fn=lambda _:None)
    with pytest.raises(ValueError,match='sealed selection'):
        pilot.load_dataset(forbidden,'evaluation',config())
    def forbidden_load(*args,**kwargs):raise AssertionError('held-out labels opened before seal')
    monkeypatch.setattr(pilot,'load_dataset',forbidden_load)
    with pytest.raises(FileNotFoundError):pilot.evaluate(tmp_path,forbidden)


def test_two_model_architectures_same_hidden_width_and_logit_output():
    import torch
    for dim in (41,8):
        model=pilot.make_model(dim)
        assert [m.out_features for m in model if isinstance(m,torch.nn.Linear)]==[16,16,1]
        assert model(torch.zeros(3,dim)).shape==(3,1)
    pilot.validate_config(config())


def test_bad_frozen_features_rejected_without_reading_datasets():
    cfg=config();cfg['features']['full_indices'][-1]=336
    with pytest.raises(ValueError,match='frozen'):pilot.validate_config(cfg)


def test_six_checkpoint_seal_detects_tamper_before_label_release(tmp_path):
    cfg=tmp_path/'config.json';pilot.write_json(cfg,config())
    receipt=tmp_path/'data_receipt.json';pilot.write_json(receipt,{'train':'bank3','development':'bank0'})
    selected=[]
    for model in ('full_state_mlp','speed_state_ablation'):
        for seed in (0,1,2):
            file=tmp_path/f'{model}{seed}.pt';file.write_bytes(f'{model}:{seed}'.encode())
            selected.append(dict(model=model,seed=seed,file=str(file),sha256=pilot.digest(file)))
    manifest=dict(implementation_sha256=pilot.digest(pilot.__file__),selected=selected,config=str(cfg),config_sha256=pilot.digest(cfg),data_receipt_sha256=pilot.digest(receipt))
    pilot.write_json(tmp_path/'selection_manifest.json',manifest)
    pilot.write_json(tmp_path/'selection_seal.json',dict(selection_manifest_sha256=pilot.digest(tmp_path/'selection_manifest.json'),selected_sha256={r['file']:r['sha256'] for r in selected}))
    assert len(pilot.verify_selection(tmp_path)['selected'])==6
    Path(selected[0]['file']).write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='checkpoint changed'):pilot.verify_selection(tmp_path)


def test_feature_builder_ignores_outcome_and_future_trajectory_metadata():
    cfg=config();source=np.zeros(351,np.float32);source[336]=1.;query=source.copy();query[336]=0.
    case=dict(candidate_alpha=0.,force_N=-64.,duration_s=.1,preparation_s=.6)
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    expected=pilot.feature_row(source,query,[0.,0.],.3,case,interface,cfg)
    case.update(primary_label=False,future_velocity=100.,future_applied_offset=[999.,999.])
    actual=pilot.feature_row(source,query,[0.,0.],.3,case,interface,cfg)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('status',['invalid','invalid_fault','invalid_incomplete'])
def test_invalid_labels_remain_unknown_not_negative(status):
    assert pilot.resolved_label(dict(label_status=status,no_fall_through_window=False)) is None


def test_resolved_label_must_agree_with_status():
    assert pilot.resolved_label(dict(label_status='fall',no_fall_through_window=False))==0.
    assert pilot.resolved_label(dict(label_status='no_fall',no_fall_through_window=True))==1.
    with pytest.raises(ValueError,match='contradiction'):
        pilot.resolved_label(dict(label_status='fall',no_fall_through_window=True))


def ranking_config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/h75_tolerance_predictor_ranking.json').read_text())


def test_ranking_pairs_never_cross_source_preparation_or_force_and_gradient_orders_logits():
    import torch
    meta=[dict(source_group_id=s,preparation_s=p,force_N=f) for s,p,f in [('a',0,64),('a',0,64),('a',.6,64),('b',0,64),('a',0,-64)]]
    pairs=pilot.discordant_pairs(np.array([1,0,0,0,0]),meta)
    np.testing.assert_array_equal(pairs,[[0,1]])
    logits=torch.zeros(5,requires_grad=True);loss=pilot.ranking_loss(logits,pairs);loss.backward()
    assert logits.grad[0]<0 and logits.grad[1]>0
    assert torch.equal(logits.grad[2:],torch.zeros(3))
    assert pilot.ranking_loss(logits.detach()-.1*logits.grad,pairs)<loss.detach()


def test_ranking_zero_pairs_is_differentiable_zero():
    import torch
    logits=torch.tensor([.2,-.3],requires_grad=True)
    loss=pilot.ranking_loss(logits,np.empty((0,2),np.int64));loss.backward()
    assert loss.item()==0 and torch.equal(logits.grad,torch.zeros(2))


def test_ranking_selectors_share_tie_rules():
    key=pilot.selector_key
    assert key('rank',.2,.75,200)<key('rank',.1,.5,100)
    assert key('rank',.1,.75,200)<key('rank',.2,.75,100)
    assert key('rank',.1,.75,100)<key('rank',.1,.75,200)
    assert key('brier',.1,0.,100)<key('brier',.1,1.,200)
    assert key('rank',.1,None,100)>key('rank',.9,0.,100)


def test_ranking_dev_final_guard_before_any_data_access(tmp_path,monkeypatch):
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(ranking_config()))
    def forbidden(*args,**kwargs):raise AssertionError('dataset accessed')
    monkeypatch.setattr(pilot,'load_dataset',forbidden)
    with pytest.raises(ValueError,match='development-only'):
        pilot.ranking_fit(cfg,tmp_path/'train',tmp_path/'analysis_final',tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('field,value',[('updates_per_model_seed',1001),('maximum_total_optimizer_updates',6001),('full_batch',False),('seeds',[0,1,2,3]),('learning_rate',.002)])
def test_ranking_budget_and_optimization_cannot_be_extended(field,value):
    cfg=ranking_config();cfg['training'][field]=value
    with pytest.raises(ValueError,match='frozen'):pilot.validate_ranking_config(cfg)


def test_ranking_config_forbids_heldout_release_and_extra_arms():
    cfg=ranking_config();pilot.validate_ranking_config(cfg)
    cfg['data']['evaluation']='bank1'
    with pytest.raises(ValueError,match='ranking'):pilot.validate_ranking_config(cfg)
    cfg=ranking_config();cfg['ranking']['arms'].append('extra')
    with pytest.raises(ValueError,match='ranking'):pilot.validate_ranking_config(cfg)
