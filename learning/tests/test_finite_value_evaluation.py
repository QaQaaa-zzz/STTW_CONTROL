"""Fixed final64 models and complete matched evaluation cells, no best picking."""
import copy
import pytest
import numpy as np

from sttw_control.finite_value_evaluation import (
    evaluation_inventory, endpoint_records, evaluation_profile, paired_report,
)
from sttw_control.finite_value_schedule import make_evaluation_schedule


def endpoint_fixture():
    source = dict(actor='parent.pkl', actor_sha256='a', checkpoint='parent.pt', checkpoint_sha256='b')
    plan = dict(model=source, updates=64)
    models = []
    for role, mode, lam in [('free_GAE','reset_free',.99), ('finite_GAE','time_nonpositive',.99), ('free_MC','reset_free',1.)]:
        models.append(dict(arm=role, profile=0, actor=role+'.pkl', actor_sha256='x',
                           checkpoint=role+'.pt', checkpoint_sha256='y',
                           value_contract=dict(role=role, mode=mode, rate=.16, gae_lambda=lam,
                                               gamma=.997, horizon_upper_steps=500)))
    status = dict(phase='completed', completed_updates=64)
    pair = dict(completed_updates=64, models=models, plan_sha256='sealed')
    return plan,status,pair


def test_exact_final_endpoint_includes_common_parent_and_three_authenticated_roles():
    plan,status,pair = endpoint_fixture()
    rows = endpoint_records(plan,status,pair,plan_sha256='sealed')
    assert set(rows) == {'parent_plain','parent_causal','parent_support','free_GAE','finite_GAE','free_MC'}
    assert rows['parent_causal']['actor'] == rows['parent_plain']['actor'] == rows['parent_support']['actor']
    assert rows['free_MC']['actor'] == 'free_MC.pkl'


@pytest.mark.parametrize('fault', ['running', 'partial', 'wrong_pair', 'duplicate', 'lambda', 'prior'])
def test_intermediate_mismatched_or_misdeclared_endpoints_rejected(fault):
    plan,status,pair = endpoint_fixture()
    if fault == 'running': status['phase']='running'
    if fault == 'partial': status['completed_updates']=63
    if fault == 'wrong_pair': pair['plan_sha256']='other'
    if fault == 'duplicate': pair['models'][2]=copy.deepcopy(pair['models'][0])
    if fault == 'lambda': pair['models'][2]['value_contract']['gae_lambda']=.99
    if fault == 'prior': pair['models'][1]['value_contract']['rate']=.17
    with pytest.raises(ValueError): endpoint_records(plan,status,pair,plan_sha256='sealed')


def test_inventory_preserves_sealed_events_and_exact_reference_modes():
    schedule = make_evaluation_schedule([dict(context='left',speed=1.8),dict(context='right',speed=1.5)],seed=23)
    inv = evaluation_inventory(schedule)
    assert len(inv)==300 and len({c['case_id'] for c in inv})==300
    assert len({(c['context'],c['alpha'],c['event_id']) for c in inv})==50
    for c,original in zip(inv,schedule):
        assert all(c[k]==v for k,v in original.items())
        assert c['requested_speed_m_s']==original['speed']
        assert c['mode_index']=={'parent_causal':1,'parent_support':2}.get(c['role'],0)


def test_incomplete_or_duplicate_inventory_is_not_silently_evaluated():
    schedule = make_evaluation_schedule([dict(context='left',speed=1.8),dict(context='right',speed=1.5)],seed=23)
    with pytest.raises(ValueError):evaluation_inventory(schedule[:-1])
    schedule[-1]=schedule[0]
    with pytest.raises(ValueError):evaluation_inventory(schedule)


def test_causal_support_uses_existing_trigger_only_while_constant_is_always_active():
    np.testing.assert_array_equal(np.asarray(evaluation_profile(np.array([0,1,2,0,1,2]),
        np.array([False,False,False,True,True,True]))),[0,0,7,0,7,7])


def test_paired_report_preserves_unknown_gains_and_losses():
    roles=['parent_plain','parent_causal','parent_support','free_GAE','finite_GAE','free_MC']
    rows=[]
    for event in ['e1','e2']:
        for role in roles:
            cat='safe_task_failed'
            if role=='parent_plain':cat='contact'
            if role=='finite_GAE':cat='safe_task' if event=='e1' else 'unknown'
            rows.append(dict(case_id=event+'/'+role,context='left',alpha=.25,event_id=event,
                             role=role,category=cat,scored_cost=10.))
    report=paired_report(rows)
    compare=report['comparisons']['finite_GAE_vs_parent_plain']
    assert compare['known']==1 and compare['unknown']==1
    assert compare['physical_gains']==1 and compare['task_gains']==1
    assert compare['details'][1]['physical_change'] is None
    with pytest.raises(ValueError):paired_report(rows[:-1])
    corrupted=copy.deepcopy(rows)
    corrupted[0].update(force_N=32.,pulse_onset_task_tick=400,pulse_duration_s=.1)
    with pytest.raises(ValueError):paired_report(corrupted)


def test_prepare_waits_for_complete_training_without_creating_an_evaluation(tmp_path):
    import json
    from sttw_control.finite_value_evaluation_run import prepare
    training=tmp_path/'training';training.mkdir()
    (training/'status.json').write_text(json.dumps(dict(phase='running',completed_updates=63)))
    output=tmp_path/'evaluation'
    with pytest.raises(ValueError,match='complete'):
        prepare(output,training=training)
    assert not output.exists()


@pytest.mark.parametrize('mode', [0,1,2])
def test_gate_audit_checks_real_prior_measurement_and_profile(mode):
    from sttw_control.finite_value_evaluation import verify_gate_log
    raw=dict(active_tick=np.ones(8,bool),physical_tick=np.arange(8),
             phi_dot=np.array([0.,0.,0.,1.,1.,1.,1.,1.]),
             relay_input_roll_rate=np.repeat([0.,1.],4),
             relay_remaining_s=np.repeat([0.,.6],4),
             relay_active=np.repeat([False,mode==1],4),
             relay_trigger=np.repeat([False,mode==1],4),
             relay_profile=np.repeat([7 if mode==2 else 0,7 if mode in [1,2] else 0],4))
    verify_gate_log(raw,dict(mode_index=mode))
    padded={k:np.concatenate([v,np.zeros(4,dtype=v.dtype)]) for k,v in raw.items()}
    verify_gate_log(padded,dict(mode_index=mode))
    raw['relay_profile'][4]=3
    with pytest.raises(AssertionError):verify_gate_log(raw,dict(mode_index=mode))


def test_analysis_waits_for_complete_evaluation_without_creating_output(tmp_path):
    import json
    from sttw_control.finite_value_evaluation_report import analyze
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='running',completed_cases=150)))
    with pytest.raises(ValueError,match='complete'):analyze(tmp_path)
    assert not (tmp_path/'analysis').exists()


def test_identical_actor_bytes_do_not_bypass_later_role_checkpoint_authentication(tmp_path):
    import pickle
    import torch
    import jax
    from sttw_control.finite_value_evaluation_run import authenticate_models
    from sttw_control.finite_value_schedule import fresh_learners
    from sttw_control.path_command_policy import make_torch_path_network,export_torch_actor
    from sttw_control.survival_online_learner import OnlineSurvivalLearner
    from sttw_control.preference_best import digest
    cfg={'network':{'architecture':'alpha_modulated_v1'}}
    torch.manual_seed(71)
    actor=make_torch_path_network(351,2,cfg);critic=make_torch_path_network(352,1,cfg)
    policy={'actor.'+k:v for k,v in actor.state_dict().items()}
    policy.update({'critic.'+k:v for k,v in critic.state_dict().items()})
    policy['log_std']=torch.tensor([-1.,-2.])
    parent=OnlineSurvivalLearner(cfg,policy,beta=0.,seed=72).checkpoint()
    learners=fresh_learners(cfg,parent,seed=73)
    actor_file=tmp_path/'common_actor.pkl'
    with actor_file.open('wb') as stream:pickle.dump(jax.device_get(export_torch_actor(actor,cfg)),stream)
    models={}
    for role,obj in learners.items():
        # Synthetic metadata fixtures; no training or result claim.
        obj.next_collection=64;obj.optimizer_steps=[248,256,256]
        obj.actor_steps=248;obj.critic_steps=256
        saved=obj.checkpoint()
        if role=='free_MC':saved['next_collection']=63
        file=tmp_path/(role+'.pt');torch.save(saved,file)
        models[role]=dict(actor=str(actor_file),actor_sha256=digest(actor_file),
                          checkpoint=str(file),checkpoint_sha256=digest(file),
                          fingerprint=obj.fingerprint(),value_contract=obj.value_contract())
    with pytest.raises(ValueError,match='endpoint'):
        authenticate_models(cfg,models)
