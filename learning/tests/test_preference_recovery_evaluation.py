import json,pickle
from pathlib import Path
import jax
import numpy as np
import pytest
import torch
from sttw_control.preference_best import atomic_json,digest,config_hash
from sttw_control.path_command_policy import SCHEMA,make_torch_path_network,export_torch_actor


def test_final_candidate_rejects_export_that_does_not_match_learner(tmp_path):
    from sttw_control.preference_recovery_evaluation import candidate_record
    cfg={'network':{'architecture':'alpha_modulated_v1'}}
    declaration=dict(seed=918301,parent_record=dict(checkpoint_sha256='parent',source_update=325),updates=27,critic_warmup_updates=2)
    identity=dict(schema=SCHEMA,alpha='conditional',lower_actor_sha256='lower',config_sha256=config_hash(cfg),seed=918301,parent_checkpoint_sha256='parent',parent_source_update=325,experiment='preference_recovery_v1',arm='guided',initialization='weights_only_fresh_optimizer')
    job=tmp_path/'guided';cp=job/'checkpoints';cp.mkdir(parents=True)
    actor=make_torch_path_network(351,2,cfg)
    learner=cp/'update_0027.pt';torch.save(dict(schema=SCHEMA,identity=identity,update=27,policy={'actor.'+k:v for k,v in actor.state_dict().items()}),learner)
    exported=cp/'actor_0027.pkl'
    with exported.open('wb') as f:pickle.dump(jax.device_get(export_torch_actor(actor,cfg)),f)
    atomic_json(job/'status.json',dict(phase='completed',completed_updates=27,policy_transitions_total=27*512*128))
    atomic_json(job/'manifest.json',dict(identity=identity,config=cfg,source={'candidate':{'actor_sha256':'lower'}}))
    atomic_json(job/'last_completed.json',dict(update=27,checkpoint=str(learner),checkpoint_sha256=digest(learner)))
    rows=[dict(update=i,training_stage='critic_warmup' if i<=2 else 'ppo',optimizer={'actor_bitwise_unchanged':True}) for i in range(1,28)]
    (job/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    record=candidate_record(tmp_path,'guided',declaration)
    assert record['actor_sha256']==digest(exported)
    with exported.open('rb') as f:data=pickle.load(f)
    data['params']['gain_2'][0]+=1
    with exported.open('wb') as f:pickle.dump(data,f)
    with pytest.raises(ValueError,match='export'):candidate_record(tmp_path,'guided',declaration)


def test_paired_task_results_distinguish_improvement_and_regression():
    from sttw_control.preference_recovery_evaluation import paired_results
    common=dict(alpha=0.,route='left',speed=2.6,onset_s=2.)
    parent=[dict(common,force_N=f,task_recovered=ok,performance_success=ok,physical_failure=False,domain_failure=not ok) for f,ok in [(0.,True),(8.,False),(-8.,True)]]
    candidate=[dict(r) for r in parent];candidate[1]['task_recovered']=True;candidate[2]['task_recovered']=False
    rows=paired_results(parent,{'guided':candidate},[0.])
    item=next(x for x in rows if x['comparison']=='u325_vs_guided' and x['alpha']==0 and x['metric']=='task_recovered')
    assert item['n']==3 and item['improved']==1 and item['regressed']==1 and item['difference']==0


def test_uniform_reuse_allows_only_sampler_scale_change_and_rejects_contract_drift():
    from copy import deepcopy
    from sttw_control.preference_recovery_evaluation import validate_uniform_reuse
    path=Path('runs/preference_recovery_train_v1/training_declaration.json')
    if not path.exists():pytest.skip('authentic training declaration unavailable')
    original=json.loads(path.read_text());new=deepcopy(original)
    new['state_sampler']={'target':2.5,'bandwidth':.5,'strength':.7,'shuffle_seed':918204}
    new['total_policy_transitions']=3538944
    validate_uniform_reuse(original,new,'uniform')
    for key,value in [('seed',42),('forces_N',[-8,8]),('updates',52),('parent_anchor_kl',.1),('config_sha256','changed'),('bank_receipt_sha256','changed')]:
        changed=deepcopy(new);changed[key]=value
        with pytest.raises(ValueError,match='reuse contract'):validate_uniform_reuse(original,changed,'uniform')
    with pytest.raises(ValueError,match='uniform only'):validate_uniform_reuse(original,new,'guided')


def test_actual_completed_uniform_reuse_authenticates_learner_and_source_declaration(tmp_path):
    from sttw_control.preference_recovery_evaluation import candidate_record
    source=Path('runs/preference_recovery_train_v1').resolve()
    if not (source/'uniform/last_completed.json').exists():pytest.skip('completed external control unavailable')
    original=source/'training_declaration.json';new=json.loads(original.read_text())
    new['state_sampler']={'target':2.5,'bandwidth':.5}
    new['reused_training']={'uniform':{'campaign':str(source),'declaration_sha256':digest(original)}}
    record=candidate_record(tmp_path,'uniform',new)
    assert record['training_reused_from']==str(source)
    assert record['actor_sha256']==digest(record['actor']) and record['source_update']==27
    new['reused_training']['uniform']['declaration_sha256']='changed'
    with pytest.raises(ValueError,match='declaration identity'):candidate_record(tmp_path,'uniform',new)
