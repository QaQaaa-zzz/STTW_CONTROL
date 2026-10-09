import importlib.util
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as jp
from sttw_control import smooth_command_config as config
from sttw_control.direct_command_reward import interval_cost, failure_reward


def test_v52_reward_deltas_and_peak_contract():
    assert hasattr(config,'resolve_preference_v52')
    spec=config.resolve_preference_v52(1)
    # Use the same live controller adapter as existing reward tests.
    from sttw_control.controller import ControllerConfig
    cc=ControllerConfig()
    kwargs=dict(alpha=1.,chi=1.,g=0.,raw=jp.array([2.6,.25]),actual_speed=2.6,actual_steer=.25,
                heading_error=0.,roll=.2,roll_rate=0.,executed_offsets=jp.array([.10,0.]),
                final_command=jp.zeros(2),previous_final_command=jp.zeros(2),spec=spec,cc=cc)
    with pytest.raises(ValueError,match='peak'):interval_cost(**kwargs)
    got=interval_cost(**kwargs,peak_roll=.3154)
    assert float(got['raw_components']['reference_priority'])==0
    np.testing.assert_allclose(got['raw_components']['working_roll_excess'],71.148,rtol=1e-4)
    assert sum(spec['reward']['independent_component_caps'].values())==5080
    np.testing.assert_allclose(failure_reward(1,spec),-15.16,rtol=1e-4)
    kwargs['executed_offsets']=jp.array([-.10,0.]);assert interval_cost(**kwargs,peak_roll=.2)['raw_components']['reference_priority']>0


def test_v52_inherits_frozen_interfaces_and_separate_budget():
    assert hasattr(config,'resolve_preference_v52')
    for alpha in (0,1):
        old=config.resolve_preference_v51(alpha);new=config.resolve_preference_v52(alpha)
        for key in ('action','network','plant','limits','reference','lower_controller','lower_reference_centered','commands','actor_temporal_regularizer'):
            assert new[key]==old[key]
        assert new['reward']['primary_excess']==old['reward']['primary_excess']
        assert new['ppo']['validation_updates']==[100,150,200]
        assert new['ppo']['fresh_value_only_rollouts']==4
        assert new['reward']['normal']['speed_deadband_m_s']==.01


def test_best_safety_partial_and_replacement():
    import sttw_control as package
    assert (Path(package.__file__).parent/'preference_best.py').exists()
    from sttw_control.preference_best import is_better,qualified,convergence_decision
    base=dict(physical_failure_cases=0,working_limit_failure_cases=1,primary_failure_cases=0,joint_final_hold_failure_cases=1,physical_quality_score=2.,all_declared_traces_complete=True)
    safe={**base,'working_limit_failure_cases':0,'physical_quality_score':4.}
    assert is_better(safe,base) and not is_better(base,safe)
    assert not is_better({**base,'physical_quality_score':1.999},base)
    with pytest.raises(ValueError):is_better({**safe,'all_declared_traces_complete':False},base)
    assert not qualified(base)
    assert convergence_decision(400,400,[],[],False)['extend_by']==0


def test_saved_alignment_and_all_excursions():
    from sttw_control.lower_tracking_audit import intervals,held,stats
    time=np.arange(10)*.005;err=np.array([0,.2,.2,0,0,.3,.3,.3,0,0])
    out=stats(err,np.ones(10,bool),time,.005,.1)
    assert len(out['all_exceeding_intervals'])==2
    assert out['longest_exceeding_s']==.015
    assert not held(np.ones(10,bool),.005,.3).any()


def test_best_load_checks_saved_identity(tmp_path):
    import torch
    from sttw_control.preference_best import load_best,digest
    policy=torch.nn.Linear(2,1);path=tmp_path/'model.pt'
    torch.save(dict(policy=policy.state_dict(),update=100),path)
    import json
    (tmp_path/'best_model.json').write_text(json.dumps(dict(checkpoint=str(path),checkpoint_sha256=digest(path),source_update=100)))
    load_best(tmp_path,policy,'cpu')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):load_best(tmp_path,policy,'cpu')


def test_reconstruct_v51_baseline_under_v52_without_new_logged_component(tmp_path):
    import json
    from sttw_control.preference_command_reporting import reward_audit
    root=Path(__file__).resolve().parents[2]
    with np.load(root/'docs/evidence/r196_preference_v51_20261009/data/update100_timeseries.npz') as z:
        prefix='straight_hold__B0__';d={k[len(prefix):]:(z[k][:8] if z[k].ndim else z[k]) for k in z.files if k.startswith(prefix)}
    p=tmp_path/'alpha0';p.mkdir();(p/'frozen_config.json').write_text(json.dumps(config.resolve_preference_v52(0)))
    result=reward_audit(d,0,tmp_path,16,include_reconstruction=True)
    assert 'working_roll_excess' in result and 'passed' not in result
    assert all(np.isfinite(v).all() for v in result.values())


def test_peak_and_failure_against_independent_reference():
    from sttw_control.controller import ControllerConfig
    p=Path(__file__).resolve().parents[2]/'docs/preference_v52/attachment/reward_delta_reference.py'
    module=importlib.util.spec_from_file_location('v52_reference',p);ref=importlib.util.module_from_spec(module);module.loader.exec_module(ref)
    for a in (0,1):
        spec=config.resolve_preference_v52(a)
        for peak in (.2,.3154,.3676,2.):
            got=interval_cost(alpha=a,chi=1.,g=0.,raw=jp.array([2.6,0.]),actual_speed=2.6,actual_steer=0.,heading_error=0.,roll=.1,roll_rate=0.,executed_offsets=jp.array([.1,.08]),final_command=jp.zeros(2),previous_final_command=jp.zeros(2),spec=spec,cc=ControllerConfig(),peak_roll=peak)
            np.testing.assert_allclose(got['effective_components']['working_roll_excess'],ref.working_roll_cost(.1,peak)[1],rtol=1e-4,atol=1e-5)
            np.testing.assert_allclose(got['raw_components']['reference_priority'],ref.ref_priority(a,1,0,.1,.08),rtol=1e-5)
            np.testing.assert_allclose(got['effective_cost'],sum(got['effective_components'].values()))
        for n in (1,400,800):np.testing.assert_allclose(failure_reward(n,spec),ref.failure_reward(n),rtol=1e-5)
