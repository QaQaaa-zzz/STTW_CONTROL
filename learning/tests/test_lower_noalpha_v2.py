"""Only new v2 contracts: sampling, fine tracking, stream and exact resume."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jp
from sttw_control.lower_command import local_command_rows,sample_local_v2
from sttw_control.controller import ControllerConfig
from sttw_control.lower_command_validation import summarize_local
CFG=json.loads((Path(__file__).parents[1]/'configs/lower_local_tracking_no_alpha_v2.json').read_text())

def test_configured_slew_and_independent_small_sign_and_time():
    def sample(i):return sample_local_v2(i,jp.int32(0),jp.asarray(2.3),CFG,ControllerConfig())
    rows,slew,labels=map(np.asarray,jax.jit(jax.vmap(sample))(jp.arange(2048)))
    assert np.ptp(slew[:,0])>.9 and np.ptp(slew[:,1])>.6
    small=labels[:,0]==2
    assert small.sum()>350
    assert rows[small,5,0].max()<=7.60001
    signs=set(zip(np.sign(rows[small,2,2]),np.sign(rows[small,5,2])))
    assert signs=={(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)}

def test_live_controller_controls_target_and_screen():
    cc=ControllerConfig()
    from dataclasses import replace
    other=replace(cc,gravity=cc.gravity*.2)
    a=local_command_rows(jp.int32(4),jp.int32(0),jp.asarray(2.3),CFG,case='return_positive',cc=cc)[0]
    b=local_command_rows(jp.int32(4),jp.int32(0),jp.asarray(2.3),CFG,case='return_positive',cc=other)[0]
    assert float(a[2,2])!=float(b[2,2])

def test_small_zero_motion_is_not_pass_and_window_is_issued():
    n=2000;t=np.arange(n)*.005
    traces={}
    for name in CFG['validation']['case_ids']:
        cmd=np.tile([2.3,0.],(n,1))
        if name.startswith('post_return_small'):cmd[t>=7.5,1]=.04
        traces[name]=dict(time=t,limited_command=cmd,actual_forward_speed=np.full(n,2.3),actual_delta=cmd[:,1].copy(),phi=np.zeros(n),peak_roll=np.zeros(n),physical_failure=np.zeros(n,bool),raw_rates=np.zeros((n,2)),final_command_clipped=np.zeros(n,bool),actual_delta_rate=np.zeros(n))
    traces['post_return_small_positive']['actual_delta'][:]=0
    result=summarize_local(traces,CFG)
    assert result['cases']['post_return_small_positive']['primary_failure']
    assert result['cases']['post_return_small_positive']['small_direction_pass'] is False
    assert result['cases']['post_return_small_positive']['windows']['post_return_small']['ticks']==400

def test_stream_derivatives_and_exact_prepublication_replay():
    from sttw_control.lower_command import fixed_reference_stream
    from sttw_control.direct_command_scenarios import publish_issued_reference
    raw=jp.array([2.3,0.]);stream=fixed_reference_stream(raw,CFG,'post_return_small_positive',ControllerConfig())
    data=np.asarray(stream);assert data.shape==(2001,5)
    np.testing.assert_allclose(data[1:,3:5],np.diff(data[:,1:3],axis=0)/.005,atol=1e-6)
    for tick in (0,99,1499,1999):
        cmd,rates,_=publish_issued_reference(raw,stream,tick,jp.ones(2),.005,True)
        np.testing.assert_array_equal(cmd,data[tick,1:3]);np.testing.assert_array_equal(rates,data[tick,3:5])

def test_rollout_boundary_full_state_roundtrip(tmp_path):
    import torch
    from types import SimpleNamespace
    from sttw_control.lower_command_training import Training,restore_boundary
    t=Training.__new__(Training);t.cfg=CFG;t.out=tmp_path;t.resume_metadata={'hard_rejects':2,'train_best_reward':-.4}
    policy=torch.nn.Linear(2,2);opt=torch.optim.Adam(policy.parameters());algo=SimpleNamespace(policy=policy,optimizer=opt,accepted_policy_updates=7,hard_kl_stop=False,nonfinite_stop=False)
    states={'physics':jp.array([1.,2.]),'history':jp.arange(210),'mask':jp.ones(10),'tick':jp.int32(256)}
    path=t.checkpoint(algo,50,'training',states)
    record=torch.load(path,weights_only=False)
    assert record['environment_continuation']=='full_rollout_boundary'
    restored=restore_boundary(path,algo,CFG)
    assert restored[1]==50 and restored[2]['hard_rejects']==2
    np.testing.assert_array_equal(restored[0]['history'],states['history'])

def test_accepted_count_without_temporal_regularizer():
    from types import SimpleNamespace
    from sttw_control.lower_command_training import account_accepted_update
    algo=SimpleNamespace(accepted_policy_updates=0,temporal_spec=None)
    account_accepted_update(algo,{'accepted_epochs':4});assert algo.accepted_policy_updates==1
    account_accepted_update(algo,{'accepted_epochs':0});assert algo.accepted_policy_updates==1
    algo.temporal_spec={'already_counted_by_ppo':True}
    account_accepted_update(algo,{'accepted_epochs':4});assert algo.accepted_policy_updates==1

def test_resume_preflight_rejects_stale_boundary_and_budget(tmp_path):
    import torch,pytest
    from sttw_control.lower_command_training import resume_preflight
    root=tmp_path/'training';(root/'checkpoints').mkdir(parents=True)
    path=root/'checkpoints/resume_boundary.pt'
    torch.save(dict(config=CFG,update=50,environment_continuation='full_rollout_boundary'),path)
    (root/'metrics.jsonl').write_text(json.dumps({'update':51})+'\n')
    with pytest.raises(ValueError,match='latest'):resume_preflight(path,tmp_path,CFG)
    (root/'metrics.jsonl').write_text(json.dumps({'update':50})+'\n')
    with pytest.raises(ValueError,match='target'):resume_preflight(path,tmp_path,{**CFG,'updates':50})
