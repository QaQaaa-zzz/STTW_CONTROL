import numpy as np
import jax.numpy as jp
from sttw_control.training import TrainingConfig


def test_exact_gaussian_kl_and_constraint_config():
    from sttw_control.training import gaussian_kl
    mu=jp.zeros((4,2));ls=jp.zeros(2)
    np.testing.assert_allclose(gaussian_kl(mu,ls,mu,ls),0,atol=1e-7)
    np.testing.assert_allclose(gaussian_kl(mu,ls,mu+1,ls),1,atol=1e-7)
    c=TrainingConfig(target_kl=.01,warmup_pool_size=4,warmup_steps=20)
    assert c.target_kl==.01


def test_screen_does_not_pass_a_frozen_or_failed_policy(tmp_path):
    import json
    from sttw_control.stability_screen import assess
    (tmp_path/'status.json').write_text('{"complete":true}')
    row=dict(control_transitions=100,steer_disturbed_transitions=10,force_disturbed_transitions=0,validation=dict(event_present=[True],failed=[False],post_event_hold_complete=[True]),optimizer_audit=dict(final_exact_kl=.001,accepted_minibatches=0))
    p=tmp_path/'metrics.jsonl';p.write_text(json.dumps(row)+'\n')
    criteria=dict(target_kl=.01,minimum_disturbed_fraction=.02,minimum_accepted_minibatches=8)
    assert not assess(tmp_path,criteria,True)['passed']
    row['optimizer_audit']['accepted_minibatches']=8;p.write_text(json.dumps(row)+'\n')
    assert assess(tmp_path,criteria,True)['passed']
    row['validation']['failed']=[True];p.write_text(json.dumps(row)+'\n')
    assert not assess(tmp_path,criteria,True)['passed']


def test_constraint_config_rejects_incomplete_warmup():
    import pytest
    for kwargs in [dict(target_kl=0),dict(target_kl=float('nan')),dict(warmup_pool_size=4),dict(warmup_steps=8)]:
        with pytest.raises(ValueError):TrainingConfig(**kwargs)
