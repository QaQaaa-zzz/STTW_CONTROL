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
