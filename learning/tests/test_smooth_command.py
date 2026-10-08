import importlib.util
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from sttw_control.smooth_command_config import resolve
from sttw_control.direct_command_reward import interval_cost,upper_motion_cost,failure_reward
root=Path(__file__).resolve().parents[2]
def reference():
    s=importlib.util.spec_from_file_location('ref',root/'docs/smooth_v4/attachment/reward_reference.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_reference_random_states():
    ref=reference();rng=np.random.default_rng(4)
    for alpha in (0,1):
        spec=resolve(alpha)
        for _ in range(30):
            v,st,hd,yaw,phi,phid=rng.normal(size=6);raw=np.array([2.5,.2]);off=rng.normal(size=2)*.1;chi,g=rng.random(2)
            r=ref.state_cost(alpha=alpha,chi=chi,recovery_weight=g,speed=v,steer=st,raw_speed=raw[0],raw_steer=raw[1],heading_debt=hd,yaw_rate=yaw,roll=phi,roll_rate=phid,offsets=off,wheelbase=spec['reference']['wheelbase_m_expected'],caster_rad=np.deg2rad(spec['reference']['caster_deg_expected']))
            x=interval_cost(alpha=alpha,chi=chi,g=g,raw=jnp.array(raw),actual_speed=v,actual_steer=st,heading_error=hd,yaw_rate=yaw,roll=phi,roll_rate=phid,executed_offsets=jnp.array(off),final_command=jnp.zeros(2),previous_final_command=jnp.zeros(2),spec=spec)
            for k in r['raw']:np.testing.assert_allclose(x['raw_components'][k],r['raw'][k],rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(x['effective_cost'],r['cost'],rtol=1e-5)
        for n in (1,10,800):np.testing.assert_allclose(failure_reward(n,spec),ref.failure_reward(n),rtol=2e-5)

def test_offset_endpoint_reset():
    s=resolve(0);ref=reference()
    for valid in [False,True]:
        now=jnp.array([-.01,.005]);previous=jnp.zeros(2);prior=jnp.array([.4,-.2])
        rate,acc,raw,eff=upper_motion_cost(now,previous,prior,valid,s)
        x=ref.upper_motion_cost(now,previous,prior,acceleration_valid=valid)
        for k in raw:np.testing.assert_allclose(raw[k],x['raw'][k],rtol=1e-6)
    _,_,raw,_=upper_motion_cost(jnp.zeros(2),jnp.zeros(2),jnp.ones(2),False,s)
    assert all(float(v)==0 for v in raw.values())

def test_temporal_pairs_and_value_only():
    import torch
    from tensordict import TensorDict
    from sttw_control.direct_command_ppo import make_algorithm
    s=resolve(0);s['ppo']['minibatches']=1
    obs=TensorDict({'policy':torch.zeros(2,345),'critic':torch.zeros(2,346)},batch_size=[2])
    algo=make_algorithm(obs,3,s,'cpu');torch.testing.assert_close(algo.policy.log_std.exp(),torch.tensor([.1,.1]));algo.temporal_spec=s;algo.accepted_policy_updates=10
    algo.storage.observations['policy'].zero_();algo.storage.dones.zero_();algo.storage.dones[0,0]=1
    algo.storage.observations['policy'][1,1,310]=1 # exclude two pairs by raw command rate
    algo.prepare_temporal_pairs()
    assert algo.temporal_denominator==3
    assert int(algo.temporal_mask.sum())==1
    # Different current Actor outputs at true paired observations, both branches differentiable.
    with torch.no_grad():algo.policy.actor[-1].weight.normal_(0,.01)
    algo.storage.observations['policy'][2,0,308]=.1
    algo.prepare_temporal_pairs();loss=algo.temporal_loss(torch.arange(4));loss.backward()
    assert torch.isfinite(loss) and float(loss.detach())>0
    for _ in range(3):
        with torch.no_grad():algo.act(obs);algo.process_env_step(obs,torch.ones(2),torch.zeros(2,dtype=torch.bool),{})
    with torch.no_grad():algo.compute_returns(obs)
    result=algo.value_only_update(2);assert result['actor_bitwise_unchanged']

def test_conflict_plateau_and_reversal():
    import jax
    from sttw_control.direct_command_scenarios import _smooth_conflict
    s=resolve(0);s['smooth_v4']['training_commands']['conflict_single_fraction']=0
    slew=jnp.array([.3,.15]);rows=np.array(_smooth_conflict(jax.random.PRNGKey(2),2.3,s,slew))
    plateau=rows[3,0]-rows[2,0]-abs(rows[2,2]-rows[1,2])/.15
    assert .7999<=plateau<=1.5001


def test_fresh_150_amendment():
    s=resolve(1,fresh=True,train_wall=4500,total_wall=10800)
    assert s['initialization_mode']=='scratch'
    assert not s['smooth_v4']['upper']['warm_start_actor_only']
    assert s['ppo']['default_updates']==150
    assert s['ppo']['validation_updates']==[60,150]
    assert s['budget']['training_wall_seconds']==4500
    assert s['lower_controller']['alias']=='STTW_R196_ALPHA1'

def test_best_prefers_physical_hold_not_last(tmp_path):
    import json
    from sttw_control.smooth_command_reporting import select_physical_checkpoint
    for stage in [60,150]:
        root=tmp_path/f'evaluation{stage}';(root/'straight_hold').mkdir(parents=True)
        row=dict(physical_failure=False,working_roll_violation_s=0.,joint_final_hold=stage==60,command={'speed_rmse':.1 if stage==60 else .01,'steer_rmse':.02},smoothness={'offset_steer':{'acceleration_rms':1.}})
        data={str(c):{f'new{a}':row for a in [0,1]} for c in range(6)}
        (root/'metrics.json').write_text(json.dumps(data))
        for a in [0,1]:
            np.savez(root/f'straight_hold/alpha{a}.npz',checkpoint_update=stage)
            folder=tmp_path/f'alpha{a}/checkpoints';folder.mkdir(parents=True,exist_ok=True);(folder/f'update_{stage:04d}.pt').write_bytes(b'test checkpoint')
    select_physical_checkpoint(tmp_path)
    for a in [0,1]:
        result=json.loads((tmp_path/f'alpha{a}/best_model.json').read_text())
        assert result['update']==60 and result['qualified']
        assert (tmp_path/f'alpha{a}/bestmodel.pt').resolve().name=='update_0060.pt'
