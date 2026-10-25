"""Fresh live-task batches: only legal actions, all actual task rewards."""
import numpy as np
from .live_task_credit import live_task_gae
from .survival_credit import resolve_survival_credit
from .survival_task_credit import reconstruct_task_critic


def assemble_live_arm(decisions,physical,cases,indices,*,collection_id,dt_s,horizon_s,gamma,gae_lambda):
    """The caller authenticates frozen-policy identity before collecting.

    Physical targets remain legacy learner telemetry; task-only callers use
    beta=0. No new physical critic or analytic tolerance reward is introduced.
    """
    import torch
    if not indices:raise ValueError('at least one complete episode required')
    names=['obs','critic','action','old_mean','old_logp','old_task','old_physical',
        'task_advantage','task_returns','physical_targets']
    vectors={k:[] for k in names};rows=[];ranges=[];offset=0
    for i in indices:
        d={k:np.asarray(v[:,i]) for k,v in decisions.items()}
        p={k:np.asarray(v[:,i]) for k,v in physical.items()};c=cases[i]
        valid=d['valid'];active=p['active_tick']
        if active.shape!=(4*len(valid),) or valid.dtype!=np.bool_:
            raise ValueError('four actual lower slots per Boolean upper decision required')
        action_mask=np.zeros_like(active);action_mask[np.flatnonzero(valid)*4]=True
        credit=resolve_survival_credit(active_tick=active,actor_action_valid=action_mask,
            contact_failure=p['contact_failure'],engineering_fault=p['nonfinite']|p['lower_fault']|p['policy_fault'],
            horizon_reached=p['horizon_reached'],physical_tick=p['physical_tick'],window_start_tick=0,
            window_ticks=round(horizon_s/dt_s),collection_policy_id=collection_id,learner_policy_id=collection_id)
        if credit['outcome'] not in ['survived','contact']:
            raise ValueError('complete known physical outcome required; unknown is not a failure target')
        if not np.array_equal(credit['actor_mask'][::4],valid):
            raise ValueError('physical and policy action masks disagree')
        if not np.array_equal(p['task_reward_valid'],active):
            raise ValueError('live task rewards must cover every actual lower tick, including fallback')
        if not np.isfinite(p['scored_tick_reward'][active]).all():
            raise ValueError('nonfinite live task reward')
        if not np.array_equal(d['task_tick'][valid],np.flatnonzero(valid)*4):
            raise ValueError('legal decisions must retain the actual non-reset task clock')
        reconstructed=reconstruct_task_critic(d['observation'],d['task_tick'],valid,dt_s=dt_s,horizon_s=horizon_s)
        np.testing.assert_allclose(d['critic_observation'][valid],reconstructed[valid],rtol=1e-6,atol=1e-6)
        rewards=np.where(active,p['scored_tick_reward'],0.).reshape(-1,4).sum(axis=1,dtype=np.float64)
        upper_active=active.reshape(-1,4).any(axis=1)
        terminal=(p['contact_failure']|p['horizon_reached']).reshape(-1,4).any(axis=1)
        targets=live_task_gae(rewards,d['old_task_value'],valid,upper_active,terminal,gamma=gamma,gae_lambda=gae_lambda)
        for field,source in dict(obs='observation',critic='critic_observation',action='raw_action',old_mean='mean',
                old_logp='log_prob',old_task='old_task_value',old_physical='physical_value').items():
            vectors[field].append(d[source][valid])
        vectors['task_advantage'].append(targets['advantages'][valid]);vectors['task_returns'].append(targets['returns'][valid])
        vectors['physical_targets'].append(credit['targets'][::4][valid])
        count=int(valid.sum());ranges.append(dict(case_index=int(i),start=offset,end=offset+count))
        force=c.get('force_N',0.);impulse=p['applied_force_impulse'][active].sum(axis=0,dtype=np.float64)
        expected=force*c.get('pulse_duration_s',.1)*np.asarray(c.get('world_direction',[0.,0.,0.]))
        reentry=p.get('reentry_actor_active',np.zeros_like(active)).reshape(-1,4).any(axis=1)&valid
        rows.append(dict(case_index=int(i),alpha=c['alpha'],force_N=force,outcome=credit['outcome'],
            actor_decisions=count,physical_ticks=credit['physical_ticks'],fallback_ticks=int(p['fallback_active'][active].sum()),
            reentry_decisions=int(reentry.sum()),task_scored_return=float(rewards[upper_active].sum()),
            unassigned_prefix_steps=targets['unassigned_prefix_steps'],
            longest_policy_transition_upper_steps=int(targets['duration_steps'].max()),
            mean_true_speed=float(p['actual_forward_speed'][active].mean()),
            max_task_path_error=float(np.max(np.abs(p['path_cross_track'][active]))),
            full_pulse_exposure=bool(np.allclose(impulse,expected,atol=.025,rtol=.05)),
            impulse_Ns=impulse.tolist(),expected_impulse_Ns=expected.tolist(),vector_start=offset,vector_end=offset+count))
        offset+=count
    batch={k:torch.tensor(np.concatenate(v),dtype=torch.float32) for k,v in vectors.items()}
    if not all(torch.isfinite(v).all() for v in batch.values()):raise ValueError('nonfinite selected live decision data')
    batch.update(collection_id=collection_id,case_ranges=ranges)
    return batch,rows
