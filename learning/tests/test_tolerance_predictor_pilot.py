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


def test_confirmation_identity_verification_precedes_all_label_access(tmp_path,monkeypatch):
    opened=[]
    monkeypatch.setattr(pilot,'load_dataset',lambda *a,**k:opened.append(True))
    with pytest.raises(FileNotFoundError):pilot.confirmation_evaluate(tmp_path/'missing_manifest',tmp_path/'analysis_confirmation',tmp_path/'output')
    assert opened==[]
    monkeypatch.undo()
    with pytest.raises(ValueError,match='explicit confirmation'):
        pilot.load_dataset(tmp_path/'analysis_final','new_confirmation',config(),confirmation=True)


def test_confirmation_exact_split_bank_inventory():
    declared=[dict(case_id=str(i),source_group_id=f's{i//18}',bank_index=2 if i<72 else 5,split='new_confirmation') for i in range(144)]
    records=[dict(case_id=c['case_id'],bank_index=c['bank_index']) for c in declared]
    pilot.validate_confirmation_inventory(declared,records)
    declared[0]['split']='development'
    with pytest.raises(ValueError,match='split/bank'):pilot.validate_confirmation_inventory(declared,records)
    declared[0]['split']='new_confirmation';records[0]['bank_index']=1
    with pytest.raises(ValueError,match='record bank'):pilot.validate_confirmation_inventory(declared,records)


def test_confirmation_reports_overall_bank_source_and_same_condition_ranking():
    meta=[dict(source_group_id=f's{i//2}',bank_index=2 if i<2 else 5,preparation_s=0.,force_N=64.) for i in range(4)]
    report=pilot.grouped_confirmation_metrics([1,0,1,0],[.9,.1,.2,.8],meta)
    assert report['overall']['n']==4
    assert report['overall']['discordant_pair_ranking']['pairs']==2
    assert set(report['per_bank'])=={'2','5'} and set(report['per_source'])=={'s0','s1'}
    assert report['per_source']['s0']['discordant_pair_ranking']['mean']==1
    assert report['per_source']['s1']['discordant_pair_ranking']['mean']==0


@pytest.mark.parametrize('tamper',['checkpoint','source','manifest'])
def test_confirmation_sealed_identity_tamper_before_label_access(tmp_path,monkeypatch,tamper):
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    snapshot=tmp_path/'source/tolerance_predictor_pilot.py';snapshot.parent.mkdir();snapshot.write_bytes(Path(pilot.__file__).read_bytes())
    candidates=[]
    for arm,selector in [('bce','rank'),('bce_rank','brier')]:
        for seed in (0,1,2):
            checkpoint=tmp_path/f'{arm}{seed}.pt';checkpoint.write_bytes(b'checkpoint')
            candidates.append(dict(arm=arm,selector=selector,seed=seed,dim=41,file=str(checkpoint),sha256=pilot.digest(checkpoint)))
    source=dict(selected=candidates,implementation_sha256=pilot.digest(snapshot),source_snapshot_sha256=pilot.digest(snapshot),config=str(cfg),config_sha256=pilot.digest(cfg))
    pilot.write_json(tmp_path/'selection_manifest.json',source)
    manifest=dict(candidate_models=candidates,ranking_source_manifest_sha256=pilot.digest(tmp_path/'selection_manifest.json'),bank_indices=[2,5],training_prior=57/72,new_optimizer_updates=0)
    path=tmp_path/'new_source_confirmation_manifest.json';pilot.write_json(path,manifest)
    pilot.verify_confirmation_candidates(path)
    if tamper=='checkpoint':Path(candidates[0]['file']).write_bytes(b'changed')
    elif tamper=='source':snapshot.write_bytes(b'changed')
    else:(tmp_path/'selection_manifest.json').write_text('{}')
    def forbidden(*args,**kwargs):raise AssertionError('new labels accessed')
    monkeypatch.setattr(pilot,'load_dataset',forbidden)
    with pytest.raises(ValueError,match='changed'):pilot.confirmation_evaluate(path,tmp_path/'analysis_confirmation',tmp_path/'output')


def test_confirmation_plan_rejects_old_banks_and_splits():
    plan=dict(mode='h75_samestate_bank144',protocol=dict(source_initialization='prepared_bank_all_prefixes',bank_indices=[2,5],source_groups=[dict(source_group_id=str(i),split='new_confirmation',bank_index=2 if i<4 else 5) for i in range(8)]))
    pilot.validate_confirmation_plan(plan)
    plan['protocol']['bank_indices']=[0,1]
    with pytest.raises(ValueError,match='plan identity'):pilot.validate_confirmation_plan(plan)


def temporal_config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/h75_tolerance_predictor_temporal_mix.json').read_text())


def temporal_fixture(early_bank,late_bank):
    meta=[];y=[]
    for source in range(4):
        for alpha in (0.,.5,1.):
            for preparation in (0.,.2,.6):
                for force in (-64.,64.):
                    meta.append(dict(case_id=f'bank{early_bank}/{len(meta)}',source_group_id=f'bank{early_bank}/early{source}',bank_index=early_bank,alpha=alpha,preparation_s=preparation,force_N=force,period='early'));y.append(float(alpha!=1.))
    for source in range(2):
        for alpha in (0.,.5,1.):
            for preparation in (.2,.6):
                for force in (-64.,64.):
                    meta.append(dict(case_id=f'bank{late_bank}/{len(meta)}',source_group_id=f'bank{late_bank}/late{source}',bank_index=late_bank,alpha=alpha,preparation_s=preparation,force_N=force,period='late'));y.append(1.)
    return dict(full=np.zeros((96,41),np.float32),y=np.asarray(y,np.float32),meta=meta,inputs={},excluded_invalid=[],audit={})


def test_temporal_balanced_selector_is_not_96_sample_mean_and_ties_are_early():
    data=temporal_fixture(0,5);y=data['y'];p=y.copy();p[:72]=1.-p[:72]
    score,parts=pilot.temporal_selection(y,p,data['meta'])
    assert score==.5 and parts==dict(early_brier=1.,late_brier=0.)
    assert np.mean((p-y)**2)==.75
    assert pilot.selector_key('brier',score,None,100)<pilot.selector_key('brier',score,None,200)


def test_temporal_ancestor_split_and_only_early_ranking_pairs():
    train=temporal_fixture(3,2);dev=temporal_fixture(0,5);pilot.validate_temporal_datasets(train,dev)
    pairs=pilot.discordant_pairs(train['y'],train['meta'])
    assert len(pairs)>0 and all(train['meta'][i]['period']=='early' for pair in pairs for i in pair)
    train['meta'][-1]['bank_index']=5
    with pytest.raises(ValueError,match='ancestor bank'):pilot.validate_temporal_datasets(train,dev)
    train=temporal_fixture(3,2);train['y'][-1]=0
    with pytest.raises(ValueError,match='late positive'):pilot.validate_temporal_datasets(train,dev)


@pytest.mark.parametrize('mutation',[lambda c:c['ranking'].update(selectors=['rank']),lambda c:c['training'].update(updates_per_model_seed=1001),lambda c:c['temporal_mix'].update(train_late_bank=5),lambda c:c['budget'].update(automatic_extensions=True)])
def test_temporal_frozen_recipe_no_search_or_implicit_extension(mutation):
    cfg=temporal_config();pilot.validate_temporal_config(cfg);mutation(cfg)
    with pytest.raises(ValueError,match='frozen'):pilot.validate_temporal_config(cfg)


def test_temporal_prepare_declines_combined_development_before_data_access(tmp_path,monkeypatch):
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(temporal_config()))
    def forbidden(*args,**kwargs):raise AssertionError('data opened')
    monkeypatch.setattr(pilot,'load_dataset',forbidden)
    with pytest.raises(ValueError,match='development-only'):
        pilot.prepare_temporal_mix(cfg,tmp_path/'train',tmp_path/'analysis_final',tmp_path/'late',tmp_path/'q',tmp_path/'prepared')


def test_temporal_diagnostics_preserve_fall_admission_at_original_risk_gate():
    data=temporal_fixture(0,5);p=np.full(96,.9)
    diagnostics=pilot.temporal_diagnostics(data['y'],p,data['meta'])
    assert diagnostics['early_falls']==24 and diagnostics['early_falls_not_rejected_08']==24
    report=pilot.temporal_metrics(data,{'constant':p})['constant']
    assert report['early']['fall_not_rejected_at_08']==24
    assert report['early']['user_rejection']['0.8']['fall_but_not_rejected_rows']==16
    assert report['late']['proper_and_ranking']['discordant_pair_ranking']['pairs']==0
    assert report['late']['proper_and_ranking']['roc_auc'] is None


def test_temporal_prepare_creates_six_loaded_initial_scalar_runs(tmp_path,monkeypatch):
    from sttw_control import tolerance_coordination as coordinator
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(temporal_config()));q=tmp_path/'q';q.mkdir();(q/'receipt.json').write_text('{}')
    train=temporal_fixture(3,2);dev=temporal_fixture(0,5)
    def early(data):return dict(full=data['full'][:72],y=data['y'][:72],meta=[{k:v for k,v in m.items() if k!='period'} for m in data['meta'][:72]],inputs={},excluded_invalid=[],audit={})
    monkeypatch.setattr(coordinator,'load_export',lambda _:None)
    monkeypatch.setattr(pilot,'load_dataset',lambda path,role,cfg:early(train if role=='training' else dev))
    late=dict(full=np.concatenate([train['full'][72:],dev['full'][72:]]),y=np.ones(48,np.float32),meta=train['meta'][72:]+dev['meta'][72:])
    monkeypatch.setattr(pilot,'load_late_features',lambda _: (late,{}))
    output=tmp_path/'run/prepared';receipt=pilot.prepare_temporal_mix(cfg,tmp_path/'old_train',tmp_path/'analysis_development',tmp_path/'late',q,output)
    assert receipt['new_optimizer_updates']==0
    for arm in ('bce','bce_rank'):
        for seed in (0,1,2):
            events=EventAccumulator(str(tmp_path/'run/tensorboard'/arm/f'seed{seed}')).Reload()
            values=events.Scalars('train/optimizer_updates')
            assert len(values)==1 and values[0].step==0 and values[0].value==0
    with np.load(output/'training.npz') as z:assert z['full'].shape==(96,41)


def test_temporal_late_feature_receipt_seal_and_bank_metadata(tmp_path):
    train=temporal_fixture(3,2);dev=temporal_fixture(0,5)
    features=np.concatenate([train['full'][72:],dev['full'][72:]]);y=np.ones(48,np.float32)
    rows=[dict(m,label=1) for m in train['meta'][72:]+dev['meta'][72:]]
    np.savez(tmp_path/'features_predictions.npz',features=features,labels=y)
    pilot.write_json(tmp_path/'predictions.json',rows)
    pilot.write_json(tmp_path/'receipt.json',dict(source_task_s=6.,resolved_rows=48,excluded_invalid=[],new_optimizer_updates=0))
    pilot.write_json(tmp_path/'seal.json',{p.name:pilot.digest(p) for p in tmp_path.iterdir() if p.is_file()})
    data,_=pilot.load_late_features(tmp_path)
    assert data['full'].shape==(48,41) and sum(m['bank_index']==2 for m in data['meta'])==24
    rows[0]['bank_index']=1;pilot.write_json(tmp_path/'predictions.json',rows)
    with pytest.raises(ValueError,match='seal changed'):pilot.load_late_features(tmp_path)
    seal=json.loads((tmp_path/'seal.json').read_text());seal['predictions.json']=pilot.digest(tmp_path/'predictions.json');pilot.write_json(tmp_path/'seal.json',seal)
    with pytest.raises(ValueError,match='metadata mismatch'):pilot.load_late_features(tmp_path)


def test_temporal_fit_checks_frozen_preparation_before_optimizer(tmp_path,monkeypatch):
    cfg=tmp_path/'cfg.json';pilot.write_json(cfg,temporal_config());prepared=tmp_path/'prepared';prepared.mkdir()
    (prepared/'training.npz').write_bytes(b'corrupt')
    pilot.write_json(prepared/'seal.json',{'training.npz':'originalsha'})
    monkeypatch.setattr(pilot,'ranking_fit',lambda *a,**k:pytest.fail('optimizer entered'))
    with pytest.raises(ValueError,match='preparation changed'):pilot.temporal_mix_fit(cfg,prepared,tmp_path/'training')
