"""New coverage and retained route geometry are sampled without simulation."""
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from sttw_control.path_command_scenarios import sample_mixed_finetune_route,validate_mixed_finetune_distribution


def test_mixture_covers_both_directions_and_retained_tasks():
    cfg=json.loads((Path(__file__).parents[1]/'configs/h75_low_speed_mixed_finetune.json').read_text())
    validate_mixed_finetune_distribution(cfg)
    generate=jax.jit(jax.vmap(lambda eid:sample_mixed_finetune_route(919101,eid,j.int32(0),j.zeros(3),cfg)))
    r=jax.device_get(generate(j.arange(256)))
    low=r.sample_group==0
    assert .35<low.mean()<.65
    for mask in (low,~low):
        assert set(r.family[mask].tolist())=={0,1,2}
        assert np.any(r.angle[mask]<0) and np.any(r.angle[mask]>0)
    assert r.speed[low].min()>=1.6 and r.speed[low].max()<=2.2
    tight=low&(r.family!=0)
    assert r.radius[tight].min()>=.9 and r.radius[tight].max()<=1.5
    # Named original130-degree turns were outside the old90..100 sampler.
    assert np.any((~low)&np.isclose(abs(r.angle),np.deg2rad(130)))
    assert np.any((~low)&np.isclose(r.radius,2.7))
    assert np.all(np.diff(r.path.s,axis=1)>0)
    assert np.all(np.isfinite(r.path.xy))
    assert np.all(r.path.s[:,-1]>r.path.goal)
    # The adopted S tasks retain their +3m exit-goal extension.
    s=(~low)&(r.family==2)&np.isclose(r.radius,1.8)&np.isclose(r.speed,2.8)
    assert np.any(s)
    np.testing.assert_allclose(r.path.goal[s]-r.path.turn_end[s],7.,atol=2e-6)


def test_schedule_extension_preserves_distribution_and_authenticates_boundary(tmp_path):
    from copy import deepcopy
    from sttw_control.path_mixed_finetuning import validate_schedule_extension
    from sttw_control.preference_best import atomic_json,config_hash,digest
    import pytest
    old=dict(training_future_phase_C=dict(updates_initial=52,updates_max=52,best_on='final52',num_envs=512,rollout_steps=128),finetuning=dict(warmup_updates=2,budget=dict(ppo_updates=50,policy_transitions=52*65536,maximum_lower_ticks=52*65536*4),stop_rule='old',distribution=dict(low_fraction=.5)))
    atomic_json(tmp_path/'effective_config.json',old)
    identity=dict(config_sha256=config_hash(old));atomic_json(tmp_path/'manifest.json',dict(config=old,identity=identity))
    cp=tmp_path/'checkpoints/update_0052.pt';cp.parent.mkdir();cp.write_bytes(b'checkpoint')
    atomic_json(tmp_path/'status.json',dict(state='completed',completed_updates=52,final_checkpoint=str(cp),policy_transitions_total=52*65536))
    atomic_json(tmp_path/'last_completed.json',dict(update=52,checkpoint=str(cp),checkpoint_sha256=digest(cp)))
    new=deepcopy(old);new['training_future_phase_C'].update(updates_initial=202,updates_max=202,best_on='final202');f=new['finetuning'];f['budget'].update(ppo_updates=200,policy_transitions=202*65536,maximum_lower_ticks=202*65536*4);f['schedule_extension']=dict(source_run=str(tmp_path),source_config_sha256=digest(tmp_path/'effective_config.json'),source_final_update=52);f['stop_rule']='extended'
    assert validate_schedule_extension(new,tmp_path,completed=True)['target_update']==202
    bad=deepcopy(new);bad['finetuning']['distribution']['low_fraction']=.6
    with pytest.raises(ValueError,match='non-schedule'):
        validate_schedule_extension(bad,tmp_path,completed=True)
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        validate_schedule_extension(new,tmp_path,completed=True)
