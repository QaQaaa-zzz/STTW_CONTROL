"""Synthetic saved-trace audit checks; no simulator or training is launched."""
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from sttw_control.controller import ControllerConfig
from sttw_control import direct_command_audit as audit
from sttw_control.direct_command_audit import COMPONENTS, reconstruct_trace


SPEC = json.loads((Path(__file__).parents[1]/'configs'/'STTW_Direct_Command_V3.json').read_text())
CONTROLLER = asdict(ControllerConfig())


def _zero_trace(n=4, failed=False):
    dt=SPEC['plant']['control_dt_s']
    raw=np.tile([2.3,0.],(n,1))
    trace=dict(time=np.arange(n)*dt,limited_command=raw,raw_rates=np.zeros((n,2)),
        final_command=np.tile([0.,23.],(n,1)),offsets=np.zeros((n,2)),
        actual_forward_speed=np.full(n,2.3),actual_delta=np.zeros(n),
        e_psi_unwrapped=np.zeros(n),phi=np.zeros(n),phi_dot=np.zeros(n),
        alpha=np.zeros(n),chi=np.zeros(n),g=np.zeros(n),
        settle_clock=np.arange(n)*dt,raw_cost=np.zeros(n),effective_cost=np.zeros(n),
        cap_fraction=np.zeros(n),reference_yaw_unwrapped=np.zeros(n),
        yaw_unwrapped=np.zeros(n),physical_failure=np.zeros(n,dtype=bool),
        scored_tick_reward=np.zeros(n),failure_cost=np.zeros(n))
    for prefix in ('raw_cost','effective_cost','scored_cost'):
        for name in COMPONENTS:trace[f'{prefix}_{name}']=np.zeros(n)
    if failed:
        trace['physical_failure'][-1]=True
        N=round(SPEC['commands']['episode_seconds']/SPEC['plant']['policy_dt_s'])
        gamma=SPEC['ppo']['gamma'];r=SPEC['reward']
        tail=(-r['failure_extra_penalty']-
              r['scale']*SPEC['plant']['policy_dt_s']*r['cost_rate_cap']*
              (-math.expm1(N*math.log(gamma))/(1.-gamma)))
        trace['scored_tick_reward'][-1]=tail
        trace['failure_cost'][-1]=-tail
    return trace


def test_reconstructs_zero_cost_and_old_clock():
    result=reconstruct_trace(_zero_trace(),SPEC,[0.,23.],CONTROLLER)
    assert result['max_abs_error'] < 1e-8
    assert result['count']==4 and result['reward_sum']==0.


def test_detects_gate_timing_and_reward_tampering():
    trace=_zero_trace()
    trace['settle_clock'][1]=.8
    result=reconstruct_trace(trace,SPEC,[0.,23.],CONTROLLER)
    assert result['errors']['settle_clock']>.7
    trace=_zero_trace();trace['raw_cost_speed'][2]=.25
    result=reconstruct_trace(trace,SPEC,[0.,23.],CONTROLLER)
    assert result['errors']['raw_cost_speed']==.25


def test_failure_replaces_entire_policy_interval():
    trace=_zero_trace(failed=True)
    result=reconstruct_trace(trace,SPEC,[0.,23.],CONTROLLER)
    assert result['physical_failure'] and result['max_abs_error'] < 1e-8
    trace['scored_tick_reward'][0]=-.01
    result=reconstruct_trace(trace,SPEC,[0.,23.],CONTROLLER)
    assert result['errors']['scored_tick_reward']==.01


def test_rejects_missing_saved_cost_component():
    trace=_zero_trace();del trace['raw_cost_roll']
    with pytest.raises(ValueError,match='raw_cost_roll'):
        reconstruct_trace(trace,SPEC,[0.,23.],CONTROLLER)


def test_partial_main_and_missing_random_are_explicit(tmp_path,monkeypatch):
    (tmp_path/'pilot').mkdir();(tmp_path/'review'/'main').mkdir(parents=True)
    checkpoint=tmp_path/'pilot'/'checkpoints'/'update_0003.pt'
    checkpoint.parent.mkdir();checkpoint.write_bytes(b'checkpoint path fixture')
    (tmp_path/'pilot'/'last_completed.json').write_text(json.dumps(
        dict(update=3,checkpoint='pilot/checkpoints/update_0003.pt')))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(controller=CONTROLLER)))
    for method in ('B0','pi_alpha0','pi_alpha1'):
        trace=_zero_trace();trace['checkpoint_update']=np.asarray(3)
        trace['partial']=np.asarray(True)
        np.savez_compressed(tmp_path/'review'/'main'/(method+'.npz'),**trace)
    monkeypatch.setattr(audit,'_prepared_previous',lambda path:np.array([0.,23.]))
    monkeypatch.setattr(audit,'audit_sensitivity',lambda *args:dict(available=False,reason='no saved observations'))
    report=audit.audit_run(tmp_path,SPEC)
    assert report['coverage']==dict(expected=6,checked=3,complete=0,
        physical_failure=0,partial=3,missing=3,invalid=0)
    assert report['traces']['random']['B0']['state']=='missing'
    assert report['traces']['main']['pi_alpha1']['passed']
    assert not report['all_traces_passed']
    assert (tmp_path/'review'/'audit.json').is_file()
