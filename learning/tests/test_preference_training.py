import json,pickle
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from sttw_control import tolerance_training as training
from sttw_control.preference_best import digest


def test_actual_single_alpha_parent_restores_all_actor_critic_and_modulation():
    from sttw_control.path_command_training import make_algorithm,export_actor
    root=Path('runs/preference_recovery_train_v1/bank')
    if not (root/'manifest.json').exists():pytest.skip('external authentic conditional bank unavailable')
    manifest=json.loads((root/'manifest.json').read_text());record=dict(manifest['parent'])
    saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    record.update(identity=saved['identity'],source_update=saved['update'])
    cfg=training.short_training_config(manifest['config'])
    obs=TensorDict({'policy':torch.zeros(4,351),'critic':torch.zeros(4,352)},batch_size=[4])
    algo=make_algorithm(obs,128,cfg,'cpu')
    receipt=training.restore_preference_parent(algo,record,manifest['source'])
    assert receipt['new_update']==0 and not algo.optimizer.state
    for k,v in saved['policy'].items():assert torch.equal(algo.policy.state_dict()[k],v),k
    with Path(record['actor']).open('rb') as stream:reference=pickle.load(stream)
    for a,b in zip(jax.tree.leaves(export_actor(algo.policy)),jax.tree.leaves(reference)):
        np.testing.assert_array_equal(a,b)
    assert cfg['training_future_phase_C']['episode_s']==10
    assert cfg['path_reward']==manifest['config']['path_reward']


def test_declared_force_histogram_and_actual_alpha_exposure_are_not_old_force_grid():
    forces=j.array([-32.,-24.,-16.,-8.,8.,16.,24.,32.])
    acc=training.empty_rollout_totals({});acc['alpha_force_exposure']=j.zeros((5,9),j.int32)
    flags=dict(physical_failure=j.zeros(2,bool),tracking_domain_failure=j.zeros(2,bool))
    stats=dict(diagnostics={},active_ticks=j.array([4,2]),recovery=dict(family=j.array([0,1]),force_index=j.array([1,8]),alpha_index=j.array([1,4]),active_ticks=j.array([4,2])),pulse=dict(fired_now=j.ones(2,bool),force=j.array([-32.,32.]),fallback_now=j.zeros(2,bool),completed=j.zeros(2,bool),enabled=j.ones(2,bool),completed_impulse=j.zeros((2,3)),completed_expected_impulse=j.zeros((2,3))))
    result=training.accumulate_rollout_totals(acc,j.array([-.1,-.2]),flags,stats,force_levels=forces)
    np.testing.assert_array_equal(result['pulse']['force_histogram'],[1,0,0,0,0,0,0,1])
    assert result['alpha_force_exposure'][1,1]==4
    assert result['alpha_force_exposure'][4,8]==2
    assert int(result['alpha_force_exposure'].sum())==6


def test_conditional_schedule_is_declared_without_hardcoded_update_limit():
    parent={'path_reward':{},'training_future_phase_C':{'episode_s':10.,'actor_lr':.0001,'critic_lr':.0003}}
    declaration=dict(updates=52,actor_lr=.00002,critic_lr=.0002)
    cfg=training.preference_training_config(parent,declaration)
    assert cfg['training_future_phase_C']['updates_max']==52
    assert cfg['training_future_phase_C']['actor_lr']==.00002
    assert parent['training_future_phase_C']['actor_lr']==.0001
    assert cfg['training_future_phase_C']['episode_s']==10.


def test_sampler_scale_is_declaration_bound_and_cannot_change_snippet_duration(tmp_path):
    root=Path('runs/preference_recovery_train_v1')
    if not (root/'guided/plan.json').exists():pytest.skip('external matched declaration unavailable')
    plan=json.loads((root/'guided/plan.json').read_text())
    decl=json.loads(Path(plan['declaration']).read_text());cfg=json.loads(Path(plan['config']).read_text())
    parent=json.loads(Path(plan['parent_manifest']).read_text());bank=json.loads((Path(plan['bank_dir'])/'manifest.json').read_text())
    dest=tmp_path/'declaration.json';plan['declaration']=str(dest)
    decl['state_sampler']={'target':2.5,'bandwidth':.5}
    dest.write_text(json.dumps(decl));plan['declaration_sha256']=digest(dest)
    assert training.validate_preference_training_plan(plan,cfg,parent,bank)['state_sampler']==decl['state_sampler']
    decl['state_sampler']['duration_s']=8.
    dest.write_text(json.dumps(decl));plan['declaration_sha256']=digest(dest)
    with pytest.raises(ValueError,match='sampler parameter'):
        training.validate_preference_training_plan(plan,cfg,parent,bank)
