"""Fixed five-collection audit of historical stochastic training; no new rollout."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.live_training_signal_audit import discounted_returns,value_calibration
from sttw_control.live_task_credit import live_task_gae
from sttw_control.survival_endpoint_task_metrics import task_metrics
from sttw_control.rejoin_comparison import geometric_task_trace
from sttw_control.anticipatory_analysis import save_csv

PROBES=(3,16,32,48,64)


def main():
    from prepare_short_recovery_campaign import register
    source=Path('runs/matched_live_ppo_v1').resolve();out=Path('runs/live_training_signal_audit_v1').resolve();out.mkdir(exist_ok=False)
    state=dict(phase='running',pid=os.getpid(),completed_probes=0,declared_probes=5,physics_steps=0,optimizer_updates=0)
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(out/'status.json',state)
    update();register(out/'ACTIVE_RUN.json','STTW_CONTROL 持续跟踪训练信号只读诊断',out/'status.json')
    try:
        plan=json.loads((source/'plan.json').read_text());assert digest(source/'plan.json')==json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']
        assert json.loads((source/'status.json').read_text())['phase']=='completed'
        for f,sha in plan['files_sha256'].items():assert digest(source/f)==sha
        cfg=json.loads((source/'effective_config.json').read_text());schedule=json.loads((source/'training_cases.json').read_text())
        atomic_json(out/'declaration.json',dict(source=str(source),source_plan_sha256=digest(source/'plan.json'),probes=list(PROBES),
            interpretation='Historical changing stochastic pre-update policies, not fixed64 endpoint comparisons or independent replicates.',
            physics_steps=0,optimizer_updates=0,analysis_code_sha256=digest(__file__)))
        journals=[];paths={};origins={};all_rows=[];signals=[];receipts=[]
        for context,c in plan['contexts'].items():
            assert digest(c['path'])==c['path_sha256']
            with np.load(c['path']) as z:paths[context]={k:z[k] for k in z.files}
            route=Path('runs/matched_live_ppo_v1_fixed64_evaluation')/context
            rp=json.loads((route/'plan.json').read_text());assert rp['source_initial_state_sha256']==c['initial_sha256']
            origins[context]=json.loads((route/'reentry/case_index.json').read_text())[0]['initial_xy']
        for number in range(1,65):
            d=source/'execution'/f'update_{number:04d}';complete=json.loads((d/'collection_complete.json').read_text());identity=json.loads((d/'sampling_identity.json').read_text())
            assert identity['cases']==schedule[number-1]
            for arm in ['plain','support']:
                result=json.loads((d/(arm+'_learner_result.json')).read_text());assert result['collection_index']==number-1
                rows=complete['case_summaries'][arm];assert len(rows)==60 and result['samples']==sum(x['actor_decisions'] for x in rows)
                journals.append(dict(collection=number,arm=arm,critic_only=result['warmup'],
                    contacts=sum(x['outcome']=='contact' for x in rows),survived=sum(x['outcome']=='survived' for x in rows),
                    mean_scored_return=float(np.mean([x['task_scored_return'] for x in rows])),
                    nominal_mean_return=float(np.mean([x['task_scored_return'] for x in rows if not x['force_N']])),
                    disturbed_mean_return=float(np.mean([x['task_scored_return'] for x in rows if x['force_N']])),
                    actor_samples=result['samples'],advantage_scale=result['task_advantage_scale'],
                    task_value_loss_first=result['history'][0]['task_value_loss'],task_value_loss_last=result['history'][-1]['task_value_loss'],
                    exact_gaussian_KL_mean=result['exact_gaussian_KL_mean'],likelihood_ratio_clip_fraction=result['likelihood_ratio_clip_fraction'],
                    sampled_latent_std=np.exp(identity['models'][arm]['log_std']).tolist()))
        save_csv(out/'all64_journals.csv',journals)
        for number in PROBES:
            update(stage='reading_probe',current_probe=number)
            d=source/'execution'/f'update_{number:04d}';complete=json.loads((d/'collection_complete.json').read_text());identity=json.loads((d/'sampling_identity.json').read_text());cases=identity['cases']
            assert digest(d/'physical.npz')==complete['raw_physical_sha256'] and digest(d/'decisions.npz')==complete['raw_decisions_sha256']
            keys=['active_tick','actor_action_valid','actual_xy','actual_yaw','actual_forward_speed','path_cross_track','path_heading_error','path_progress','goal_section_signed_distance','peak_roll','contact_failure','nonfinite','lower_fault','policy_fault','horizon_reached','physical_tick','scored_tick_reward','effective_costs','raw_costs','chi']
            with np.load(d/'physical.npz') as z:physical={k:z[k] for k in keys}
            with np.load(d/'decisions.npz') as z:decision={k:z[k] for k in ['valid','old_task_value']}
            active=physical['active_tick'].reshape(500,4,120).any(axis=1)
            rewards=np.where(physical['active_tick'],physical['scored_tick_reward'],0.).reshape(500,4,120).sum(axis=1,dtype=np.float64)
            mc=discounted_returns(rewards,active,gamma=plan['gamma']);adv=np.full_like(mc,np.nan);returns=np.full_like(mc,np.nan)
            terminal=(physical['contact_failure']|physical['horizon_reached']).reshape(500,4,120).any(axis=1)
            for lane,c in enumerate(cases):
                g=live_task_gae(rewards[:,lane],decision['old_task_value'][:,lane],decision['valid'][:,lane],active[:,lane],terminal[:,lane],gamma=plan['gamma'],gae_lambda=plan['gae_lambda'])
                adv[:,lane]=g['advantages'];returns[:,lane]=g['returns']
                raw={k:v[:,lane] for k,v in physical.items()};a=raw['active_tick']
                geometry=geometric_task_trace(raw,paths[c['context']],cfg,initial_xy=origins[c['context']])
                metrics=task_metrics(dict(raw,**geometry,domain_exit=np.zeros_like(a),task_reward_valid=a),speed=c['speed'],goal_s=float(paths[c['context']]['goal']),config=cfg)
                outcome='contact' if np.any(a&raw['contact_failure']) else 'survived'
                all_rows.append(dict(collection=number,arm=c['arm'],case_index=lane,context=c['context'],alpha=c['alpha'],force_N=c['force_N'],
                    sampled_policy_collection_index=identity['models'][c['arm']]['collection_index'],outcome=outcome,**metrics))
            upper_t=np.arange(500)[:,None]*.02;lower_t=np.arange(2000)[:,None]*.005
            onset=np.asarray([c['pulse_onset_task_tick']*.005 for c in cases])[None,:];force=np.asarray([c['force_N']!=0 for c in cases])[None,:]
            phase_masks=dict(nominal_all=np.broadcast_to(~force,active.shape),nominal_tail=np.broadcast_to(~force,active.shape)&(upper_t>=9),
                disturbed_pre=force&(upper_t<onset),disturbed_first_second=force&(upper_t>=onset)&(upper_t<onset+1),
                disturbed_later=force&(upper_t>=onset+1),disturbed_tail=force&(upper_t>=9))
            for arm in ['plain','support']:
                columns=np.asarray([c['arm']==arm for c in cases])[None,:];legal=decision['valid']&columns
                result=json.loads((d/(arm+'_learner_result.json')).read_text());values=adv[legal]
                np.testing.assert_allclose(np.mean(values),result['task_advantage_center'],rtol=1e-5,atol=2e-5)
                np.testing.assert_allclose(np.std(values),result['task_advantage_scale'],rtol=1e-5,atol=2e-5)
                normalized=(adv-result['task_advantage_center'])/max(result['task_advantage_scale'],1e-8)
                total_abs=float(np.sum(abs(normalized[legal])))
                for phase,mask in phase_masks.items():
                    m=legal&mask;lower_m=physical['active_tick']&np.repeat((columns&mask),4,axis=0)
                    factor=np.array([1-(1-cfg['path_reward']['path_secondary_ratio'])*c['alpha'] for c in cases])[None,:]
                    path_cap=factor*cfg['path_reward']['component_caps']['path']
                    signals.append(dict(collection=number,arm=arm,phase=phase,
                        sample_fraction=float(m.sum()/legal.sum()),absolute_normalized_advantage_mass_fraction=float(np.sum(abs(normalized[m]))/total_abs),
                        mean_normalized_advantage=float(np.mean(normalized[m])) if m.any() else None,
                        mc_calibration=value_calibration(decision['old_task_value'],mc,m),gae_calibration=value_calibration(decision['old_task_value'],returns,m),
                        base_path_cost_capped_fraction=float(np.mean((physical['raw_costs'][:,:,1]>path_cap)[lower_m])) if lower_m.any() else None,
                        mean_turn_gate_chi=float(np.mean(physical['chi'][lower_m])) if lower_m.any() else None))
            _plot_xy(out,number,cases,physical,paths)
            receipts.append(dict(collection=number,physical_sha256=complete['raw_physical_sha256'],decisions_sha256=complete['raw_decisions_sha256']))
            atomic_json(out/'probe_task_metrics.json',all_rows);atomic_json(out/'phase_signal_metrics.json',signals);atomic_json(out/'raw_receipts.json',receipts)
            update(stage='probe_complete',completed_probes=len(receipts))
        save_csv(out/'probe_task_metrics.csv',all_rows);save_csv(out/'phase_signal_metrics.csv',signals)
        atomic_json(out/'receipt.json',dict(probes=list(PROBES),episodes=600,arms=2,all64_journal_rows=128,physics_steps=0,optimizer_updates=0,visual_review_complete=False,
            note='MC calibration mixes irreducible stochastic uncertainty and value error; normalized-advantage magnitude is not an actual policy-gradient norm.'))
        update(phase='completed',stage='numeric_complete_review_pending',completed_probes=5)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


def _plot_xy(out,number,cases,physical,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained');colors=plt.get_cmap('viridis')
    for ri,context in enumerate(['left','right']):
        for ci,arm in enumerate(['plain','support']):
            ax=axes[ri,ci];path=paths[context];ref=path['xy'][path['s']<=float(path['goal'])];ax.plot(ref[:,0],ref[:,1],'k:',lw=2)
            for lane,c in enumerate(cases):
                if c['context']!=context or c['arm']!=arm:continue
                a=physical['active_tick'][:,lane];xy=physical['actual_xy'][a,lane];fall=physical['contact_failure'][a,lane].any()
                ax.plot(xy[:,0],xy[:,1],color=colors(c['alpha']),ls='-' if c['force_N']==0 else '--',alpha=.7 if c['force_N']==0 else .3,lw=1)
                ax.scatter(*xy[-1],marker='x' if fall else '.',color=colors(c['alpha']),s=15)
            ax.set(title=f'{context} / {arm} /30 stochastic episodes',xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    fig.suptitle(f'Pre-update collection{number} policy / solid zero-force, dashed disturbed / alpha0 purple to1 yellow')
    fig.savefig(out/f'all_xy_collection_{number:04d}.png',dpi=110);plt.close(fig)


if __name__=='__main__':main()
