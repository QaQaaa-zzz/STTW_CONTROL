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


def test_final_only_validation_schedule():
    from sttw_control.training import should_validate
    c=TrainingConfig(updates=8,checkpoint_interval=4,validation_final_only=True)
    assert [i for i in range(1,9) if should_validate(c,i)]==[8]
    c=TrainingConfig(updates=8,checkpoint_interval=4)
    assert [i for i in range(1,9) if should_validate(c,i)]==[1,4,8]


def test_wall_clock_warning_does_not_terminate():
    import subprocess
    from sttw_control.stability_screen import wait_stage
    class Child:
        def __init__(self):self.calls=[]
        def wait(self,timeout=None):
            self.calls.append(timeout)
            if len(self.calls)==1:raise subprocess.TimeoutExpired('test',timeout)
            return 0
        def terminate(self):raise AssertionError('warning must not terminate')
    child=Child();warnings=[]
    assert wait_stage(child,None,2700,warnings.append)==0
    assert child.calls==[2700,None] and len(warnings)==1


def test_explicit_legacy_timeout_still_stops_child():
    import subprocess
    import pytest
    from sttw_control.stability_screen import wait_stage
    class Child:
        stopped=False
        def wait(self,timeout=None):
            if not self.stopped:raise subprocess.TimeoutExpired('test',timeout)
            return -15
        def terminate(self):self.stopped=True
    child=Child()
    with pytest.raises(TimeoutError):wait_stage(child,1,2700,lambda _:None)
    assert child.stopped
def test_explicit_validation_updates_override_periodic_schedule():
    import pytest
    from sttw_control.training import TrainingConfig,should_validate
    c=TrainingConfig(updates=50,validation_updates=(25,50))
    assert [i for i in range(1,51) if should_validate(c,i)]==[25,50]
    with pytest.raises(ValueError):TrainingConfig(updates=50,validation_updates=(51,))
    with pytest.raises(ValueError):TrainingConfig(updates=50,validation_updates=(25,25))
