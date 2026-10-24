"""Blocked online episode assignments and completed-tail dataset construction."""
import numpy as np
from .survival_credit import resolve_survival_credit
from .survival_task_credit import complete_task_gae,reconstruct_task_critic


def make_online_cases(reference,*,alphas=(0.,.5,1.),replicates=4):
    if len(reference)!=4 or replicates<1 or any(not 0<=a<=1 for a in alphas):
        raise ValueError('four declared reference force cases, legal alphas and positive replication required')
    cases=[]
    for arm_index,arm in enumerate(['task_only','physical_credit']):
        for source_case,c in enumerate(reference):
            for ai,alpha in enumerate(alphas):
                for rep in range(replicates):
                    cases.append(dict(c,alpha=float(alpha),arm_index=arm_index,arm=arm,source_case=source_case,
                        replicate=rep,rng_fold_in=((source_case//2)*len(alphas)+ai)*replicates+rep))
    return cases


def assemble_online_arm(decisions,physical,cases,indices,*,collection_id,dt_s,horizon_s,gamma,gae_lambda):
    import torch
    vectors={k:[] for k in ['obs','critic','action','old_mean','old_logp','old_task','old_physical',
                           'task_advantage','task_returns','physical_targets']}
    rows=[];offset=0
    for i in indices:
        d={k:v[:,i] for k,v in decisions.items()};p={k:v[:,i] for k,v in physical.items()};case=cases[i]
        valid=d['valid'];active=p['active_tick'];action_mask=np.zeros_like(active)
        action_mask[np.flatnonzero(valid)*4]=True
        credit=resolve_survival_credit(active_tick=active,actor_action_valid=action_mask,
            contact_failure=p['contact_failure'],engineering_fault=p['nonfinite']|p['lower_fault']|p['policy_fault'],
            horizon_reached=p['horizon_reached'],physical_tick=p['physical_tick'],window_start_tick=0,
            window_ticks=round(horizon_s/dt_s),collection_policy_id=collection_id,learner_policy_id=collection_id)
        if credit['outcome'] not in ['survived','contact']:raise ValueError('complete known physical tail required for update')
        if not np.array_equal(credit['actor_mask'][::4],valid):raise ValueError('physical and Actor decision masks disagree')
        reconstructed=reconstruct_task_critic(d['observation'],d['task_tick'],valid,dt_s=dt_s,horizon_s=horizon_s)
        np.testing.assert_allclose(d['critic_observation'][valid],reconstructed[valid],rtol=1e-6,atol=1e-6)
        task=p['task_reward_valid']&active
        if np.any(task.reshape(-1,4).any(axis=1)&~valid):raise ValueError('task rewards without a legal Actor decision')
        rewards=np.where(task,p['scored_tick_reward'],0.).reshape(-1,4).sum(axis=1,dtype=np.float64)
        rewards[~valid]=np.nan
        last=np.flatnonzero(valid)[-1];terminal=np.zeros_like(valid);terminal[last]=True
        event=p['domain_exit']|p['contact_failure']|p['horizon_reached']
        if not np.any(event[last*4:last*4+4]&task[last*4:last*4+4]):raise ValueError('task prefix lacks a true terminal')
        targets=complete_task_gae(rewards,d['old_task_value'],valid,terminal,gamma=gamma,gae_lambda=gae_lambda)
        for name,source in dict(obs='observation',critic='critic_observation',action='raw_action',old_mean='mean',
                old_logp='log_prob',old_task='old_task_value',old_physical='physical_value').items():vectors[name].append(d[source][valid])
        vectors['task_advantage'].append(targets['advantages'][valid]);vectors['task_returns'].append(targets['returns'][valid])
        vectors['physical_targets'].append(credit['targets'][::4][valid])
        count=int(valid.sum());force=case.get('force_N',0.)
        impulse=p['applied_force_impulse'][active].sum(axis=0,dtype=np.float64)
        expected=force*case.get('pulse_duration_s',.1)*np.asarray(case.get('world_direction',[0.,0.,0.]))
        rows.append(dict(case_index=int(i),alpha=case['alpha'],force_N=force,outcome=credit['outcome'],
            actor_decisions=count,physical_ticks=credit['physical_ticks'],hold_ticks=int(p['fallback_active'][active].sum()),
            task_scored_return=float(np.nansum(rewards)),mean_true_speed=float(p['actual_forward_speed'][active].mean()),
            max_task_path_error=float(np.max(np.abs(p['path_cross_track'][task]))),
            full_pulse_exposure=bool(np.allclose(impulse,expected,atol=.025,rtol=.05)),
            impulse_Ns=impulse.tolist(),expected_impulse_Ns=expected.tolist(),
            vector_start=offset,vector_end=offset+count))
        offset+=count
    batch={k:torch.tensor(np.concatenate(v),dtype=torch.float32) for k,v in vectors.items()}
    batch['collection_id']=collection_id
    return batch,rows
