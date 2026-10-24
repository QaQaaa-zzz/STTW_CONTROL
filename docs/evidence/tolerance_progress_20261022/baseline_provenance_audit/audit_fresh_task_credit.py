"""Reconstruct original task Critic views/GAE from authenticated fresh decisions.

No integration, resampling, optimizer or original-artifact edits.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as j
from sttw_control.preference_best import digest,atomic_json
from sttw_control.survival_task_credit import reconstruct_task_critic,complete_task_gae
from sttw_control.path_command_policy import make_torch_path_network
from sttw_control.contact_only_environment import load_environment_contact_only
from sttw_control.geometric_path import build_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('preserve previous audit outputs')
    plan=json.loads((root/'plan.json').read_text());results=json.loads((root/'results.json').read_text())
    prelaunch=json.loads((root/'prelaunch_receipt.json').read_text())
    assert digest(root/'plan.json')==prelaunch['plan_sha256']
    assert digest(plan['source_declaration'])==plan['source_declaration_sha256']
    decl=json.loads(Path(plan['source_declaration']).read_text());cfg=decl['config']
    for kind in ['actor','checkpoint']:
        assert digest(plan['model'][kind])==plan['model'][kind+'_sha256']
    env_source=Path('learning/src/sttw_control/path_command_env.py').resolve()
    assert digest(env_source)==plan['source_sha256'][str(env_source)]
    checkpoint=torch.load(plan['model']['checkpoint'],map_location='cpu',weights_only=False)
    critic=make_torch_path_network(352,1,cfg)
    critic.load_state_dict({k[7:]:v for k,v in checkpoint['policy'].items() if k.startswith('critic.')},strict=True)
    critic.eval()
    before=[p.detach().clone() for p in critic.parameters()]
    env,initial=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2')
    with (root/'initial_carry.pkl').open('rb') as stream:carry=pickle.load(stream)
    pose=np.asarray(env.physics.helpers.pose(initial.data))
    routes={r['name']:build_path(r['builder'],pose,cfg) for r in decl['routes']}
    horizon=plan['horizon_s'];dt=env.cc.dt
    train=cfg['training_future_phase_C'];rows=[];inputs={};gradient_data=[]
    out.mkdir(parents=True)
    for row,case in zip(results['cases'],plan['cases']):
        index=row['case'];assert index==len(rows)
        raw=root/f'case_{index:03d}.npz'
        assert digest(raw)==row['raw_sha256'];inputs[str(raw)]=digest(raw)
        with np.load(raw,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        valid=data['decision_valid'];ticks=data['decision_task_tick']
        view=reconstruct_task_critic(data['decision_observation'],ticks,valid,dt_s=dt,horizon_s=horizon)
        state=jax.tree.map(lambda x:j.asarray(x[index]),carry.state)
        actual_obs,actual_critic,fault=env.observation(state,routes[case['route']],j.float32(case['speed']),horizon)
        assert not bool(fault)
        initial_error=float(np.max(np.abs(view[0]-np.asarray(actual_critic))))
        np.testing.assert_allclose(view[0],np.asarray(actual_critic),atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(data['decision_observation'][0],np.asarray(actual_obs),atol=1e-6,rtol=1e-6)
        # Independent direct JAX implementation of the environment's clock expression.
        remaining=np.asarray((horizon-j.asarray(ticks[valid])*dt)/horizon)
        np.testing.assert_array_equal(view[valid,-1],remaining)
        values=np.full(valid.shape,np.nan,dtype=np.float32)
        with torch.no_grad():values[valid]=critic(torch.tensor(view[valid])).squeeze(-1).numpy()
        active=data['task_reward_valid']&data['active_tick']
        rewards=np.where(active,data['scored_tick_reward'],0.).reshape(-1,4).sum(axis=1,dtype=np.float64)
        assert np.all(~active.reshape(-1,4).any(axis=1)|valid)
        rewards[~valid]=np.nan
        terminal=np.zeros_like(valid);last=np.flatnonzero(valid)[-1];terminal[last]=True
        event=data['domain_exit']|data['contact_failure']|data['horizon_reached']
        assert np.any(event[last*4:last*4+4]&active[last*4:last*4+4])
        targets=complete_task_gae(rewards,values,valid,terminal,gamma=train['gamma'],gae_lambda=train['gae_lambda'])
        assert not np.any(data['nonfinite']|data['lower_fault']|data['policy_fault'])
        survival_target=0. if np.any(data['contact_failure']) else 1.
        assert survival_target==0. or np.any(data['horizon_reached'])
        gradient_data.append((data['decision_observation'][valid],data['decision_raw_action'][valid],
            data['decision_log_prob'][valid],targets['advantages'][valid],
            np.full(int(valid.sum()),survival_target-.5,dtype=np.float32)))
        assert np.isclose(np.nansum(rewards),row['task_scored_return'],atol=1e-4,rtol=1e-7)
        output=out/f'case_{index:03d}_task.npz'
        np.savez_compressed(output,critic_observation=view,old_task_value=values,decision_reward=rewards,
                            task_valid=valid,task_terminal=terminal,**targets)
        rows.append(dict(case=index,legal_actions=int(valid.sum()),initial_env_view_max_error=initial_error,initial_env_view_within_tolerance=True,
            all_remaining_clock_exact=True,old_value_min=float(values[valid].min()),old_value_max=float(values[valid].max()),
            task_return=float(np.nansum(rewards)),advantage_mean=float(targets['advantages'][valid].mean()),
            terminal_task_tick=int(ticks[last]),output_sha256=digest(output)))
    assert all(torch.equal(p.detach(),old) for p,old in zip(critic.parameters(),before))
    # Describe local gradient geometry only; do not select beta or step weights.
    actor=make_torch_path_network(351,2,cfg)
    actor.load_state_dict({k[6:]:v for k,v in checkpoint['policy'].items() if k.startswith('actor.')},strict=True)
    log_std=torch.nn.Parameter(checkpoint['policy']['log_std'].detach().clone())
    parameters=list(actor.parameters())+[log_std]
    x,a,old,task_adv,physical_adv=[torch.tensor(np.concatenate(parts),dtype=torch.float32)
        for parts in zip(*gradient_data)]
    new=torch.distributions.Normal(actor(x),log_std.exp()).log_prob(a).sum(-1)
    assert float((new.detach()-old).abs().max())<1e-3
    task_adv=(task_adv-task_adv.mean())/task_adv.std(unbiased=False).clamp_min(1e-8)
    ratio=(new-old).exp();clipped=ratio.clamp(.8,1.2)
    denominator=len(plan['cases'])*round(horizon/(4*dt))
    task_loss=-torch.minimum(ratio*task_adv,clipped*task_adv).sum()/denominator
    physical_loss=-torch.minimum(ratio*physical_adv,clipped*physical_adv).sum()/denominator
    gt=torch.autograd.grad(task_loss,parameters,retain_graph=True)
    gs=torch.autograd.grad(physical_loss,parameters)
    gt=torch.cat([g.reshape(-1) for g in gt]);gs=torch.cat([g.reshape(-1) for g in gs])
    assert torch.isfinite(gt).all() and torch.isfinite(gs).all()
    geometry=dict(task_gradient_norm=float(gt.norm()),physical_gradient_norm=float(gs.norm()),
        cosine=float(torch.dot(gt,gs)/(gt.norm()*gs.norm())),actor_denominator=denominator,
        task_advantage_normalization='batch mean/std, frozen for this diagnostic',
        physical_baseline=.5,selected_beta=None,optimizer_steps=0,
        scope='One development batch local gradient geometry, not efficacy or a new gradient method.')
    atomic_json(out/'gradient_geometry.json',geometry)
    atomic_json(out/'receipt.json',dict(cases=rows,input_sha256=inputs,
        plan_sha256=digest(root/'plan.json'),checkpoint_sha256=digest(plan['model']['checkpoint']),
        initial_carry_sha256=digest(root/'initial_carry.pkl'),
        source_sha256={str(p.resolve()):digest(p) for p in [Path(__file__),env_source,
            Path('learning/src/sttw_control/survival_task_credit.py')]},
        gamma=train['gamma'],gae_lambda=train['gae_lambda'],new_physics_ticks=0,optimizer_steps=0,
        critic_unchanged=True,scope='Original task inputs/old values/complete-task GAE reconstructed; no performance claim.'))
    print(json.dumps(dict(cases=rows,gradient_geometry=geometry)))


if __name__=='__main__':main()
