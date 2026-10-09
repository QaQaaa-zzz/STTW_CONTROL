"""V5.2 orchestration around the existing optimized SmoothCampaign/PPO."""
from pathlib import Path
import json,os,pickle,shutil,time
import numpy as np
from .preference_best import (digest,config_hash,atomic_json,summarize,is_better,persist_best,load_best,convergence_decision,CASES)
from .direct_command_training import write,notify


def warm_start_endpoint(c,alpha):
    import torch
    from .smooth_command_training import capture_rng
    ep=c.endpoints[alpha];algo=ep['algo'];root=c.repair_parent
    pointer=json.loads((root/f'alpha{alpha}/last_completed.json').read_text());src=Path(pointer['checkpoint'])
    saved=torch.load(src,map_location='cuda',weights_only=False)
    cfg=saved['config'];spec=c.resolve(alpha)
    if saved['update']!=100 or pointer['update']!=100 or cfg['upper_alpha']!=alpha or not cfg.get('priority_recovery_v51'):
        raise ValueError('requires corresponding V5.1 alpha endpoint at100')
    for key in ('action','network','plant','limits','reference','lower_controller','lower_reference_centered','commands','actor_temporal_regularizer'):
        if spec[key]!=cfg[key]:raise ValueError('parent interface mismatch '+key)
    critic={k:v.clone() for k,v in algo.policy.critic.state_dict().items()}
    algo.policy.actor.load_state_dict({k[6:]:v for k,v in saved['policy'].items() if k.startswith('actor.')})
    with torch.no_grad():algo.policy.log_std.copy_(saved['policy']['log_std'])
    assert all(torch.equal(v,algo.policy.critic.state_dict()[k]) for k,v in critic.items()) and not algo.optimizer.state
    ep['rng']=capture_rng();ep['parent']=dict(checkpoint=str(src),sha256=digest(src),actor=str(src.with_name('actor_0100.pkl')),update=100,alpha=alpha)
    write(c.out/f'alpha{alpha}/initialization.json',dict(initialization='parent_actor_and_log_std_only',parent=ep['parent'],
        actor_exact=True,log_std_exact=True,critic_fresh=True,optimizer_empty=True,physical_state_restored=False,
        physical_eso_history='new complete episodes from original prepared bank',value_only_rollouts=4,parent_updates=100,additional_updates=0,lineage_updates=100,reward_version='V5.2'))


def accumulate_episodes(c,alpha,reward,done,failed,diag,warmup,index):
    """Record complete episodes across batches and pre-reset family/identity."""
    import jax
    ep=c.endpoints[alpha];d=jax.device_get(diag['episode_ticks']);r,done,failed=jax.device_get((reward,done,failed))
    n=c.n
    if 'v52_episodes' not in ep:
        ep['v52_episodes']=dict(count=np.zeros(n,int),reward=np.zeros(n),sum=np.zeros((n,8)),start=np.full(n,index),start_phase=np.full(n,'value_only' if warmup else 'policy',object),
            quiet=np.zeros(n,int),last_gov=np.asarray(d['governed'][:,0]).copy(),lower_bad=np.zeros(n,int),lower_longest=np.zeros(n,int),
            recovery_ticks=np.zeros(n,int),hold=np.zeros(n,int),recovery_time=np.full(n,np.nan),had_recovery=np.zeros(n,bool))
    x=ep['v52_episodes'];fresh=x['count']==0;x['start'][fresh]=index;x['start_phase'][fresh]='value_only' if warmup else 'policy';x['reward']+=r
    for t in range(4):
        active=d['active_tick'][:,t];raw=d['limited_command'][:,t];gov=d['governed'][:,t]
        ev=d['actual_forward_speed'][:,t]-raw[:,0];ed=d['actual_delta'][:,t]-raw[:,1];head=d['e_psi_unwrapped'][:,t];peak=d['peak_roll'][:,t]
        lv=d['actual_forward_speed'][:,t]-gov[:,0];ld=d['actual_delta'][:,t]-gov[:,1]
        rate=np.nan_to_num((gov-x['last_gov'])/.005,nan=0.);quiet=(abs(rate[:,0])<=.1)&(abs(rate[:,1])<=.02)&active
        x['quiet']=np.where(quiet,x['quiet']+1,0);bad=(x['quiet']>=100)&((abs(lv)>.15)|(abs(ld)>.04))&active
        x['lower_bad']=np.where(bad,x['lower_bad']+1,0);x['lower_longest']=np.maximum(x['lower_longest'],x['lower_bad'])
        recovery=d['recovery_phase'][:,t]&active;x['had_recovery']|=recovery;x['recovery_ticks']+=recovery
        hold=recovery&(abs(ev)<=.10)&(abs(ed)<=.04)&(abs(head)<=.05)&(peak<=.302)
        x['hold']=np.where(hold,x['hold']+1,0);hit=(x['hold']>=100)&np.isnan(x['recovery_time']);x['recovery_time'][hit]=x['recovery_ticks'][hit]*.005
        # Same physical-cost definition per family across V5.2 windows, independent of reward version.
        conflict=d['chi'][:,t]>=.8
        primary=(ed/.05)**2 if alpha==0 else (ev/.08)**2
        secondary=(ev/.60)**2 if alpha==0 else (ed/.15)**2
        task=np.where(conflict,primary+.2*secondary,(ev/.10)**2+(ed/.04)**2)+(head/.05)**2+2*(peak>.302)/.01
        terms=np.stack([ev**2,ed**2,head**2,peak>.302,lv**2,ld**2,task,d['final_command_clipped'][:,t]],-1)
        x['sum']+=terms*active[:,None];x['count']+=active;x['last_gov']=gov.copy()
    families=np.asarray(ep['state'].family);ids=np.asarray(ep['state'].episode_index)
    rows=[]
    for i in np.flatnonzero(done):
        count=int(x['count'][i]);v=x['sum'][i]/max(count,1)
        row=dict(family=int(families[i]),episode_index=int(ids[i]),env_id=int(i),start_update=int(x['start'][i]),end_update=index,
            start_phase=str(x['start_phase'][i]),end_phase='value_only' if warmup else 'policy',mixed_policy_versions=bool(x['start'][i]!=index or x['start_phase'][i]!=('value_only' if warmup else 'policy')),
            actual_ticks=count,return_sum=float(x['reward'][i]),physical_failure=bool(failed[i]),speed_rmse=float(np.sqrt(v[0])),steer_rmse=float(np.sqrt(v[1])),heading_rmse=float(np.sqrt(v[2])),
            working_violation_s=float(x['sum'][i,3]*.005),lower_speed_rmse=float(np.sqrt(v[4])),lower_steer_rmse=float(np.sqrt(v[5])),
            physical_task_cost=float(v[6]),final_clip_fraction=float(v[7]),lower_persistent_longest_s=float(x['lower_longest'][i]*.005),
            recovery_applicable=bool(x['had_recovery'][i]),recovery_success=bool(np.isfinite(x['recovery_time'][i])),recovery_time_s=float(x['recovery_time'][i]) if np.isfinite(x['recovery_time'][i]) else None)
        rows.append(row)
        for k in ('count','reward','sum','quiet','lower_bad','lower_longest','recovery_ticks','hold','had_recovery'):x[k][i]=0
        x['recovery_time'][i]=np.nan;x['start'][i]=index;x['start_phase'][i]='value_only' if warmup else 'policy'
        # First reset tick compares to its own actual initial governed reference.
        x['last_gov'][i]=np.nan
    # A reset starts a new derivative sequence, never measures a jump across episodes.
    if rows:
        with (c.out/f'alpha{alpha}/complete_episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    c.write_reset_events(jax.tree.map(lambda x:np.asarray(x)[None],jax.device_get(diag['reset_event'])),f'alpha{alpha}')


def episode_window(c,alpha,update):
    path=c.out/f'alpha{alpha}/complete_episodes.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    rows=[r for r in rows if r['end_phase']=='policy' and update-25<r['end_update']<=update and r['start_phase']=='policy']
    out=dict(end_update=update,window_updates=25,complete_episodes=len(rows),families={},family_task_cost={})
    for family in range(4):
        group=[r for r in rows if r['family']==family]
        if not group:out['families'][str(family)]=dict(status='no_samples');continue
        keys=('return_sum','physical_task_cost','speed_rmse','steer_rmse','heading_rmse','working_violation_s','lower_speed_rmse','lower_steer_rmse','lower_persistent_longest_s')
        val={k:float(np.mean([r[k] for r in group])) for k in keys};val['count']=len(group);val['physical_failure_count']=sum(r['physical_failure'] for r in group);val['physical_failure_fraction']=val['physical_failure_count']/len(group)
        recover=[r for r in group if r['recovery_applicable']];times=[r['recovery_time_s'] for r in recover if r['recovery_time_s'] is not None]
        val.update(recovery_samples=len(recover),recovery_success_fraction=float(np.mean([r['recovery_success'] for r in recover])) if recover else None,recovery_time_s=float(np.mean(times)) if times else None)
        out['families'][str(family)]=val;out['family_task_cost'][str(family)]=val['physical_task_cost']
    ep=c.endpoints[alpha];ep.setdefault('episode_windows',[]).append(out)
    with (c.out/f'alpha{alpha}/episode_windows.jsonl').open('a') as f:f.write(json.dumps(out,allow_nan=False)+'\n')
    return out


def candidate_identity(c,alpha,update,source_update=None):
    return dict(source_update=update if source_update is None else source_update,parent_updates=100,additional_updates=update,lineage_updates=100+update,
        reward_version='V5.2',parent=c.endpoints[alpha]['parent'],config=c.resolve(alpha),config_sha256=config_hash(c.resolve(alpha)),
        protocol_id='fixed_command_six_v1',seed=77001,bank_index=3,prepared_bank_sha256=json.loads((c.out/'manifest.json').read_text())['prepared_bank_sha256'],
        scope='best among validated candidates only',physical_state_saved=False)


def consider(c,alpha,update,traces,checkpoint,actor,source_update=None):
    summary=summarize(traces,alpha);ep=c.endpoints[alpha]
    identity=candidate_identity(c,alpha,update,source_update);identity.update(checkpoint_sha256=digest(checkpoint),metrics=summary,source_reward_version='V5.1' if update==0 else 'V5.2',validation_metrics_version='V5.2 physical protocol')
    ep.setdefault('candidates',[]).append(dict(additional_updates=update,checkpoint_sha256=identity['checkpoint_sha256'],score_tuple=summary['score_tuple'],qualified=summary['qualified']))
    ep.setdefault('validations',[]).append(summary)
    out=c.out/f'alpha{alpha}'
    atomic_json(out/f'validation_{update:04d}.json',identity)
    incumbent=ep.get('best')
    if is_better(summary,None if incumbent is None else incumbent['metrics']):
        ep['best']=persist_best(out,checkpoint,actor,summary,identity,ep['candidates'])
    ep['best']['candidate_set']=list(ep['candidates'])
    atomic_json(out/'best_model.json',ep['best'])
    atomic_json(out/'validated_candidates.json',ep['candidates'])
    return summary


def validate(c,update,alphas=(0,1)):
    from .preference_command_reporting import _load,reward_audit
    # Only Actor methods; B0 physics is reused from the already frozen parent protocol.
    c.evaluate_preference(update,batched=True,methods_filter=[f'alpha{a}' for a in alphas],render=False)
    for a in alphas:
        traces={case:_load(c.out/f'evaluation{update}/{case}/alpha{a}.npz') for case in CASES}
        audits={case:reward_audit(d,a,c.out,16) for case,d in traces.items()}
        atomic_json(c.out/f'evaluation{update}/alpha{a}_reward_audits.json',audits)
        if not all(x['passed'] for x in audits.values()):raise RuntimeError('V5.2 reward reconstruction failed')
        ckpt=c.out/f'alpha{a}/checkpoints/update_{update:04d}.pt'
        consider(c,a,update,traces,ckpt,ckpt.with_name(f'actor_{update:04d}.pkl'))


def parent_candidates(c):
    from .preference_command_reporting import _load
    old=json.loads((c.repair_parent/'manifest.json').read_text());new=json.loads((c.out/'manifest.json').read_text())
    for key in ('prepared_bank_sha256','lower'):
        if old[key]!=new[key]:raise ValueError('parent evaluation identity mismatch '+key)
    # Original traces were produced by the same fixed protocol/sample/seed implementation.
    for a in (0,1):
        ep=c.endpoints[a];traces={case:_load(c.repair_parent/f'evaluation100/{case}/alpha{a}.npz') for case in CASES}
        consider(c,a,0,traces,ep['parent']['checkpoint'],ep['parent']['actor'],source_update=100)
    atomic_json(c.out/'parent_evaluation_reuse.json',dict(source=str(c.repair_parent/'evaluation100'),protocol_id='fixed_command_six_v1',seed=77001,bank_index=3,
        prepared_bank_sha256=new['prepared_bank_sha256'],lower=new['lower'],selection='physical metrics only; old reward not compared to V5.2',new_physics_ticks=0))


def _final_best_impl(c):
    from .preference_command_reporting import _load,_plots,reward_audit
    records={a:load_best(c.out/f'alpha{a}',c.endpoints[a]['algo'].policy) for a in (0,1)}
    atomic_json(c.out/'final_best_loaded.json',{str(a):r for a,r in records.items()})
    print('FINAL BEST SHA '+json.dumps({str(a):dict(sha=r['checkpoint_sha256'],source_update=r['source_update']) for a,r in records.items()}),flush=True)
    c.evaluate_preference('best',batched=True,methods_filter=['alpha0','alpha1'],render=False)
    root=c.out/'evaluationbest';results={}
    for case in CASES:
        parent=c.repair_parent/f'evaluation100/{case}/B0.npz';shutil.copyfile(parent,root/case/'B0.npz')
        traces={m:_load(root/case/f'{m}.npz') for m in ('alpha0','alpha1','B0')}
        # Re-score B0 on V5.2 for plotting; old raw reward fields are not used as new evidence.
        for m in ('alpha0','alpha1'):
            audit=reward_audit(traces[m],int(m[-1]),c.out,16)
            if not audit['passed']:raise RuntimeError('final best reward audit failed')
        _plots(traces,root,case,'best (see SHA manifest)',10,(c.out,16))
        if case=='fast_turn':_plots(traces,root,case,'best (see SHA manifest)',16,(c.out,16))
    for a in (0,1):results[str(a)]=summarize({case:_load(root/case/f'alpha{a}.npz') for case in CASES},a)
    atomic_json(root/'validation_metrics.json',results)
    lines=['# V5.2 reloaded best — physical results','', 'Best identity: ../final_best_loaded.json. Last remains the final learner; best is selected only among fixed-protocol validated candidates.','']
    for case in CASES:lines.append(f'![{case}]({case}_10s_xy.png)')
    lines+=['','10s conclusions are primary; fast_turn16s is separate. See validation_metrics.json and per-endpoint validated_candidates.json.']
    (root/'INDEX.md').write_text('\n'.join(lines)+'\n')
    return results


def final_best(c):
    # Reporting must never contaminate last learner/Adam if final evaluation fails.
    saved={a:{k:v.detach().clone() for k,v in ep['algo'].policy.state_dict().items()} for a,ep in c.endpoints.items()}
    try:return _final_best_impl(c)
    finally:
        for a,weights in saved.items():c.endpoints[a]['algo'].policy.load_state_dict(weights)


def run_preference_v52(output,repair_parent,additional_updates=200,max_additional_updates=400,smoke=False):
    from .smooth_command_training import SmoothCampaign
    c=SmoothCampaign(output,fresh=True,preference_v52=True,repair_parent=repair_parent,additional_updates=additional_updates,max_additional_updates=max_additional_updates,smoke=smoke,unlimited_wall=True)
    try:
        c.initialize()
        manifest=json.loads((c.out/'manifest.json').read_text());manifest.update(implementation_parent='26c3fedac76b099ae98ff153f3f50f09f2730da9',performance_parent='7ab2cd8',reward_version='V5.2',
            initialization='corresponding parent100 Actor and log_std; new Critic/Adam; physical reset',parent_run=str(c.repair_parent),policy_budget_semantics='additional PPO batches; 4 value-only excluded; initial200 conditional100 extensions capped400')
        write(c.out/'manifest.json',manifest)
        for a in (0,1):c.create_endpoint(a)
        c._status(parent_updates=100,additional_updates={'0':0,'1':0},lineage_updates={'0':100,'1':100},reward_version='V5.2',max_additional_updates=max_additional_updates)
        if smoke:
            # One engineering check, two total PPO batches: one per independent endpoint.
            for a in (0,1):
                if not c.batch(a):raise RuntimeError('engineering PPO stopped')
                c.save(a,c.endpoints[a]['update'])
            write(c.out/'smoke_result.json',dict(passed=True,num_envs=8,policy_steps=16,total_updates=2,updates_per_endpoint=1,new_policy_transitions=256,scope='engineering only; no scene evaluation'))
            c._status(state='complete',stage='engineering_complete',declared_policy_batches_per_endpoint=1);return
        parent_candidates(c)
        for a in (0,1):
            for _ in range(4):
                if not c.batch(a,warmup=True):raise RuntimeError('value-only warmup stopped')
        targets={a:additional_updates for a in (0,1)};discordant={a:False for a in (0,1)}
        for update in range(1,max_additional_updates+1):
            active=[a for a in (0,1) if update<=targets[a]]
            if not active:break
            for a in active:
                if not c.batch(a):raise RuntimeError(f'alpha{a}: '+str(c.endpoints[a]['stopped']))
            c._status(additional_updates={str(a):ep['update'] for a,ep in c.endpoints.items()},lineage_updates={str(a):100+ep['update'] for a,ep in c.endpoints.items()},targets=targets)
            if update>=100 and update%50==0:
                for a in active:c.save(a,update)
                validate(c,update,active)
            for a in active:
                if update==targets[a]:
                    ep=c.endpoints[a];decision=convergence_decision(update,max_additional_updates,ep['validations'][1:],ep.get('episode_windows',[]),discordant[a])
                    discordant[a]=decision['discordant_extension_used'];targets[a]+=decision['extend_by'];ep['decision']=decision
                    write(c.out/f'alpha{a}/decision_{update:04d}.json',decision)
                    c.notify_once(f'training_{a}_{update}','STTW V5.2训练阶段结束',f'alpha{a} 新增{update}；{decision["reason"]}；目标{targets[a]}')
        for a,ep in c.endpoints.items():c.save(a,ep['update'])
        result=final_best(c)
        c._status(state='complete',stage='best_evaluated',qualified={a:s['qualified'] for a,s in result.items()},decisions={str(a):ep.get('decision') for a,ep in c.endpoints.items()})
        c.notify_once('pipeline_complete','STTW V5.2完成','best已按SHA重新加载评价，完整图保存；是否合格见指标')
    except BaseException as exc:
        for a,ep in c.endpoints.items():
            try:c.save(a,ep['update'])
            except Exception:pass
        c._status(state='error',reason=repr(exc));notify('STTW V5.2错误停止',repr(exc));raise
    finally:
        for ep in c.endpoints.values():ep['writer'].close()
