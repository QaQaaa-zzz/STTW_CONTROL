"""No fitting or dynamics; frozen schema and authenticated reader checks."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control import tolerance_predictor_pilot as m
ROOT=Path(__file__).resolve().parents[2]


def test_fixed_budget_config():
    cfg=json.loads((ROOT/'learning/configs/h75_tolerance_predictor_five_alpha.json').read_text())
    m.validate_five_alpha_fit_config(cfg)
    for field,value in [('updates_per_model_seed',1001),('fixed_last',False),('learning_rate',.01)]:
        other=copy.deepcopy(cfg);other['training'][field]=value
        with pytest.raises(ValueError):m.validate_five_alpha_fit_config(other)


def test_grouped_metrics_keep_singleclass_uncertainty():
    meta=[dict(source_group_id='source7',preparation_s=.2,force_N=-64.,alpha=a,lifecycle_role='post_arrival_survival') for a in [0.,.25,.5,.75,1.]]
    report=m.five_alpha_grouped_metrics(np.zeros(5),np.full(5,.3),meta)
    assert report['overall']['roc_auc'] is None
    assert len(report['groups']['alpha'])==5
    assert report['groups']['lifecycle_role']['post_arrival_survival']['n']==5
    assert len(m.discordant_pairs(np.zeros(5),meta))==0


@pytest.fixture
def dataset(tmp_path):
    protocol=json.loads((ROOT/'learning/configs/h75_five_alpha_tolerance_labels.json').read_text())
    cfg=json.loads((ROOT/'learning/configs/h75_tolerance_predictor_five_alpha.json').read_text())
    def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    model={'registry_id':'STTW_MIDDLE_FULLTAIL_PPO75_U77'}
    for key in ('actor','checkpoint'):
        f=tmp_path/key;f.write_text(key);model[key]=str(f);model[key+'_sha256']=m.digest(f)
    common=dict(mode='five_alpha_tolerance_labels',protocol=protocol,models={'H75':model},config={'residual_interface':dict(speed_positive_scale_m_s=1.,speed_negative_scale_m_s=1.,steer_scale_rad=.2)},input_sha256={},source_sha256={})
    all_rows=[];all_cases=[];features=[];roles=[];children=[]
    for t in [1.,6.]:
        child=tmp_path/f'task{t:g}';children.append(str(child));rows=[];cases=[];states=[];source_records=[]
        for g in [g for g in protocol['source_groups'] if g['source_task_s']==t]:
            start=len(cases)
            for a in protocol['candidate_alphas']:
                for prep in [.2,.6]:
                    for force in [-64.,64.]:
                        cid=f"source{g['source_index']}/a{a}/p{prep}/f{force}"
                        c=dict(g,case_id=cid,alpha=.5,user_alpha=.5,query_alpha=a,candidate_alpha=a,preparation_s=prep,force_N=force,duration_s=.15,onset_s=t+prep,primary_window_start_s=t,primary_window_end_s=t+prep+.15+3)
                        source=np.zeros(351,np.float32);source[336]=.5;query=source.copy();query[336]=a;latent=np.zeros(2,np.float32)
                        raw=dict(time=np.array([t]),active_tick=np.ones(1,bool),reference_target_permitted=np.ones(1,bool),query_observation=query[None],composed_latent=latent[None],pre_upper_yaw=np.zeros(1))
                        for k in ['policy_fault','lower_fault','nonfinite','continuation_geometry_fault','physical_failure']:raw[k]=np.zeros(1,bool)
                        raw['contact_failure']=np.ones(1,bool)
                        f=child/'evaluation'/f'raw{len(cases)}.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez(f,**raw)
                        role='training' if c['route'] in protocol['fitting_split']['training_routes'] else 'development'
                        r=dict(c,raw_file=str(f),raw_sha256=m.digest(f),fitting_role=role,primary=dict(label_status='fall',no_fall_through_window=False))
                        x,_=m.feature_row(source,query,latent,0.,c,common['config']['residual_interface'],cfg,expected_source_alpha=.5)
                        cases.append(c);rows.append(r);states.append(source);features.append(x);roles.append(role)
            source_records.append(dict(case_indices=list(range(start,len(cases))),all_twenty_siblings_exact=True))
        save(child/'inventory.json',cases);save(child/'analysis/results.json',rows)
        np.savez(child/'evaluation/source_features.npz',native_observation=np.array(states),source_world_yaw=np.zeros(140))
        save(child/'evaluation/source_records.json',source_records)
        save(child/'plan.json',common);sha=m.digest(child/'plan.json');save(child/'seal.json',dict(plan_sha256=sha));save(child/'analysis/receipt.json',dict(plan_sha256=sha))
        all_rows+=rows;all_cases+=cases
    parent=dict(common,label_parent=True,children=children);save(tmp_path/'plan.json',parent);sha=m.digest(tmp_path/'plan.json');save(tmp_path/'seal.json',dict(plan_sha256=sha));save(tmp_path/'analysis/receipt.json',dict(plan_sha256=sha));save(tmp_path/'inventory.json',all_cases);save(tmp_path/'analysis/results.json',all_rows)
    np.savez(tmp_path/'analysis/features_labels.npz',features=np.array(features),labels=np.zeros(280),case_ids=np.array([r['case_id'] for r in all_rows]),fitting_role=np.array(roles),valid=np.ones(280,bool))
    return tmp_path/'analysis',cfg


def test_authenticated_reader_and_raw_tamper(dataset):
    analysis,cfg=dataset;data=m.load_five_alpha_dataset(analysis,cfg)
    assert data['training']['full'].shape==(160,41)
    assert data['development']['full'].shape==(120,41)
    assert data['receipt']['source_groups']==14
    row=json.loads((analysis/'results.json').read_text())[0]
    Path(row['raw_file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw changed'):m.load_five_alpha_dataset(analysis,cfg)


def test_feature_cache_tamper_rejected(dataset):
    analysis,cfg=dataset;file=analysis/'features_labels.npz'
    with np.load(file) as z:cache=dict(z)
    cache['features'][0,0]=1.;np.savez(file,**cache)
    with pytest.raises(AssertionError):m.load_five_alpha_dataset(analysis,cfg)
