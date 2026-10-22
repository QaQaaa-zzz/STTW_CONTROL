"""Performance-gated course for the distilled persistent-preference Actor.

One finite experiment, four reset-time distributions, no old PPO sample replay.
Existing batched physics/evaluation, sparse reset and DirectCommandPPO are reused.
"""
from pathlib import Path
import copy,json,pickle,time,os,subprocess,gc
import numpy as np
import jax
import jax.numpy as j
from .preference_best import atomic_json
from .path_command_training import PathBatch,Episode,load_environment,make_algorithm,save_checkpoint,export_actor,_start_tensorboard,_verify_reward_http
from .path_command_distillation import batched_advance
from .path_command_policy import make_path_actor
from .path_command_statistics import merge_statistics,reduce_statistics,summarize_statistics


def route_for_stage(seed,eid,episode,pose,cfg,stage):
    from .path_command_scenarios import episode_key,_integrate_route
    key=episode_key(seed,eid,episode);u=jax.random.uniform(key,(12,))
    level=j.maximum(0,stage-j.asarray((stage>0)&(u[10]<cfg['curriculum']['previous_fraction']),j.int32))
    levels=cfg['curriculum']['stages'];speeds=j.asarray([x['speed_m_s'] for x in levels]);radii=j.asarray([x['radius_m'] for x in levels])
    speed=speeds[level,0]+u[5]*(speeds[level,1]-speeds[level,0]);radius=radii[level,0]+u[3]*(radii[level,1]-radii[level,0])
    group=j.where(eid<102,0,j.where(eid<306,1,2)).astype(j.int32)
    family=group
    angle=j.deg2rad(j.where(group==0,j.where(u[0]<.5,0.,30.),90.+10*u[4]));radius=j.where(group==0,j.maximum(radius,3.),radius)
    sign=j.where(eid%2==0,1.,-1.)
    lat=j.where(u[6]<.2,-.1+.2*u[7],0.);heading=j.where(u[6]<.2,-.05+.1*u[8],0.)
    return _integrate_route(pose,cfg['path'],family,radius,angle,speed,sign,lat,heading,sample_group=group)


class CurriculumBatch(PathBatch):
    def reset(self,eid,episode,absolute_update=0):
        from .path_command_conditioning import sample_episode_alpha
        pose=self.env.physics.helpers.pose(self.snapshot.data)
        route=route_for_stage(self.config['curriculum']['route_seed'],eid,episode,pose,self.config,absolute_update)
        alpha=sample_episode_alpha(self.config['alpha_conditioning']['seed'],eid,episode,self.config,1)
        state=self.env.reset(self.snapshot,route.path,route.speed,alpha)
        return Episode(state,route,eid,episode,j.zeros(6),j.zeros(10))


def panel_summary(panels,cfg):
    """Common completion precedes descriptive preference; no corridor added."""
    from .path_command_reporting import _interval_context
    b=cfg['stage_B_baseline'];rows=[];dt=cfg['timing']['lower_dt_s'];h=cfg['training_future_phase_C']['episode_s']
    for key,traces in panels.items():
        alpha=float(key.removeprefix('alpha_').replace('p','.'))
        for case,d in traces.items():
            ev=np.asarray(d['actual_forward_speed'])-d['v_user'];ey=d['path_cross_track'];eh=d['path_heading_error'];p=d['path_progress'];roll=d['peak_roll'];n=len(ev)
            tail=np.asarray(d['time'])+dt>h-b['hold_s']+1e-5
            fail=bool(np.any(d['physical_failure']));domain=bool(np.any(d.get('tracking_domain_failure',d['domain_exit'])))
            goal=bool('straight' in case or np.any((d['goal_section_signed_distance']>=0)&(p>=d['goal_progress'])))
            hold=bool(n==round(h/dt) and tail.sum()==round(b['hold_s']/dt) and np.all(abs(ev[tail])<=.1) and np.all(abs(ey[tail])<=.1) and np.all(abs(eh[tail])<=.05))
            turn=abs(d['path_curvature'])>1e-4
            mask=turn if turn.any() else np.ones(n,bool)
            rmse=lambda x:float(np.sqrt(np.mean(x[mask]**2)))
            rows.append(dict(alpha=alpha,case=case,physical=fail,domain=domain,goal=goal,hold=hold,working=bool(np.any(roll>.302)),peak_roll=float(max(roll)),roll_exposure_s=float(np.sum(roll>.302)*dt),path_rmse=rmse(ey),speed_rmse=rmse(ev),samples=n))
    common=not any(x['physical'] or x['domain'] or not x['goal'] or not x['hold'] or x['working'] for x in rows)
    primary=all(x['path_rmse']<=.1 if x['alpha']==0 else x['speed_rmse']<=.05 if x['alpha']==1 else True for x in rows if 'straight' not in x['case'])
    score=[sum(x['physical'] for x in rows),sum(x['domain'] for x in rows),sum(not(x['goal'] and x['hold']) for x in rows),sum(x['working'] for x in rows),max(((1-x['alpha'])*x['path_rmse']/.1+x['alpha']*x['speed_rmse']/.05) for x in rows),sum(x['roll_exposure_s'] for x in rows)]
    return dict(common_qualified=common,endpoint_primary_qualified=primary,score=score,cases=rows,corridor='diagnostic only; no .4m gate')


def promotion(stage,stage_updates,streak,summary,course):
    passed=summary['common_qualified'] and (stage==0 or summary['endpoint_primary_qualified'])
    streak=streak+1 if passed else 0
    upgrade=stage<len(course['stages'])-1 and stage_updates>=course['minimum_stage_updates'] and streak>=course['consecutive_passes']
    return (stage+1,0) if upgrade else (stage,streak)


def stage_config(cfg,stage):
    c=copy.deepcopy(cfg);level=c['curriculum']['stages'][stage]
    c['path']['turn_radius_m']=min(level['radius_m'])
    c['stage_C_evaluation']['low_speed_m_s']=min(level['speed_m_s']);c['stage_C_evaluation']['conflict_speed_m_s']=max(level['speed_m_s'])
    return c


def draw_panel(folder,panels,cfg,baseline=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path_command_reporting import reward_series
    folder=Path(folder);names=list(next(iter(panels.values())))
    for kind in ('XY','XY_TURN_EXIT','STEP_REWARDS','CUMULATIVE_REWARDS'):
        fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
        for ax,name in zip(axes.flat,names):
            path=np.load(folder/'alpha_0'/(name+'_path.npz'))
            if kind.startswith('XY'):ax.plot(path['xy'][:,0],path['xy'][:,1],'k--',label='fixed reference')
            for key,traces in panels.items():
                a=float(key.removeprefix('alpha_').replace('p','.'));d=traces[name]
                if kind.startswith('XY'):ax.plot(d['actual_xy'][:,0],d['actual_xy'][:,1],label=key)
                else:
                    r,_=reward_series(d,a,cfg);ax.plot(np.arange(len(r)),np.cumsum(r) if kind.startswith('CUM') else r,label=key)
                if kind.startswith('XY') and np.any(d['physical_failure']|d['domain_exit']):ax.plot(*d['actual_xy'][-1],'x',c='red')
                if baseline is not None and not kind.startswith('XY'):
                    r,_=reward_series(baseline[name],a,cfg);ax.plot(np.arange(len(r)),np.cumsum(r) if kind.startswith('CUM') else r,ls='--',lw=.7,alpha=.5)
            if baseline is not None and kind.startswith('XY'):ax.plot(baseline[name]['actual_xy'][:,0],baseline[name]['actual_xy'][:,1],label='local300 B0',c='gray')
            if kind=='XY_TURN_EXIT' and 'straight' not in name:
                curved=np.flatnonzero(abs(path['curvature'])>1e-8);start=float(path['s'][curved[0]]);end=float(path['turn_end']);mask=(path['s']>=start-2)&(path['s']<=end+7);xy=path['xy'][mask]
                lo=xy.min(axis=0)-1.5;hi=xy.max(axis=0)+1.5
                for dd in list(traces[name] for traces in panels.values())+([] if baseline is None else [baseline[name]]):
                    if np.any(dd['physical_failure']|dd['domain_exit']):lo=np.minimum(lo,dd['actual_xy'][-1]-1);hi=np.maximum(hi,dd['actual_xy'][-1]+1)
                ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
            ax.set_title(name);ax.grid(alpha=.2);ax.legend(fontsize=6)
            if kind.startswith('XY'):ax.set_aspect('equal',adjustable='box');ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)')
            else:ax.set_xlabel('actual lower tick (5 ms)');ax.set_ylabel('reward');ax.set_yscale('symlog',linthresh=.01)
        fig.suptitle(f'{folder.name} | {kind} | frozen local300 | real endpoints')
        fig.savefig(folder/(kind+'.png'),dpi=120);plt.close(fig)
    (folder/'INDEX.md').write_text(f'# {folder.name}\n\n![XY turn/exit](XY_TURN_EXIT.png)\n\n[Full XY](XY.png)\n\n[Per step](STEP_REWARDS.png) · [Cumulative](CUMULATIVE_REWARDS.png) · [Metrics](metrics.json)\n')


def train_course(cfg,source,actor,root,status,account,notice,tb,smoke=False):
    import torch
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter
    from .path_command_evaluation import BatchedPanelEvaluator,evaluate_conditional
    from .direct_command_training import clean_numbers
    from .lower_command_training import handle_candidate_ppo_result
    torch.manual_seed(cfg['run_seed']+(1 if smoke else 0));np.random.seed(cfg['run_seed']+(1 if smoke else 0))
    root=Path(root);root.mkdir();atomic_json(root/'manifest.json',dict(config=cfg,source=source,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),terminal_contract='terminal_contract_v2',data_role='TRAIN random course; reused DEV, no holdout'))
    env,snapshot=load_environment(cfg,source,'terminal_contract_v2');batch=CurriculumBatch(env,snapshot,cfg,cfg['run_seed'],None)
    n,steps=cfg['engineering_dimensions'] if smoke else (cfg['training_future_phase_C']['num_envs'],cfg['training_future_phase_C']['rollout_steps']);total=1 if smoke else cfg['curriculum']['updates'];stage=0;stage_start=0;streak=0
    reset=jax.jit(jax.vmap(batch.reset,in_axes=(0,0,None)));eps=reset(j.arange(n,dtype=j.int32),j.zeros(n,j.int32),j.int32(stage))
    env.set_log_template(jax.tree.map(lambda x:x[0],eps.state),jax.tree.map(lambda x:x[0],eps.route.path),eps.route.speed[0]);observe=jax.jit(jax.vmap(batch.observation));obs=observe(eps)
    td=lambda o:TensorDict({'policy':torch.utils.dlpack.from_dlpack(o[0]),'critic':torch.utils.dlpack.from_dlpack(o[1])},batch_size=[n])
    algo=make_algorithm(td(obs),steps,cfg,'cuda');algo.policy.actor.load_state_dict(actor.state_dict())
    status(stage='compiling_ppo',completed_updates=0,declared_updates=total);advance=jax.jit(batched_advance(batch)).lower(eps,j.zeros((n,2)),j.int32(stage)).compile()
    writer=SummaryWriter(str(root.parent/'tensorboard'/('engineering' if smoke else 'ppo')));current=td(obs);best={};evaluators={};baselines={};rejections=0;accepted_updates=0
    identity=dict(mode='distilled_conditional_curriculum_v1',seed=cfg['run_seed'],reward=cfg['path_reward'],lower_actor_sha256=source['candidate']['actor_sha256'])
    def evaluate(update):
        scfg=stage_config(cfg,stage)
        if stage not in evaluators:
            evaluators[stage]=BatchedPanelEvaluator(env,snapshot,scfg,status=lambda **kw:status(stage='validation',**{k:v for k,v in kw.items() if k not in ('state','stage')}),account=lambda count:account('dev',count))
            baselines[stage]=evaluators[stage].evaluate(make_path_actor(scfg).init(jax.random.PRNGKey(0),j.zeros(351)),0.,root/f'stage_{stage+1}_baseline')
        dest=root/'validation'/f'update_{update:04d}_stage{stage+1}'
        panels=evaluate_conditional(evaluators[stage],export_actor(algo.policy),[0.,.25,.5,.75,1.],dest,diagnostic_only=True)
        summary=panel_summary(panels,scfg);atomic_json(dest/'metrics.json',summary);draw_panel(dest,panels,scfg,baselines[stage])
        if stage not in best or tuple(summary['score'])<tuple(best[stage]['score']):
            best[stage]=dict(update=update,stage=stage,score=summary['score'],validation=str(dest.resolve()),actor=str((root/'checkpoints'/f'actor_{update:04d}.pkl').resolve()),qualified=summary['common_qualified'] and summary['endpoint_primary_qualified'])
            atomic_json(root/f'best_stage_{stage+1}.json',best[stage])
        writer.add_scalar('dev/common_qualified',summary['common_qualified'],update);writer.add_scalar('dev/goal_hold_failures',summary['score'][2],update);writer.add_scalar('dev/working_failures',summary['score'][3],update)
        writer.add_scalar('dev/physical_failures',summary['score'][0],update);writer.add_scalar('dev/domain_failures',summary['score'][1],update)
        return summary
    warmup=0 if smoke else cfg['critic_warmup_batches']
    for update in range(-warmup+1,total+1):
        # update <=0 is value-only; it never consumes one of the 400 PPO slots.
        sampling=time.monotonic();payloads=[]
        status(stage='critic_warmup' if update<=0 else 'sampling',completed_updates=max(0,update-1),declared_updates=total,curriculum_stage=stage+1)
        with torch.no_grad():
            for _ in range(steps):
                z=algo.act(current);eps,nxt,reward,done,payload=advance(eps,jax.dlpack.from_dlpack(z.detach().contiguous()),j.int32(stage));current=td(nxt)
                algo.process_env_step(current,torch.utils.dlpack.from_dlpack(reward),torch.utils.dlpack.from_dlpack(done),{});payloads.append(payload)
            host=jax.device_get(payloads)
            if any(int(x['engineering_fault_count']) for x in host):raise RuntimeError('PPO rollout engineering fault; no update')
            algo.compute_returns(current)
        ticks=sum(int(x['active_ticks']) for x in host);account('engineering' if smoke else 'warmup' if update<=0 else 'training',ticks)
        sample_seconds=time.monotonic()-sampling;started=time.monotonic()
        result=algo.value_only_update(4) if update<=0 else algo.update()
        if update>0:
            rejections,stop=handle_candidate_ppo_result(algo,result,rejections,cfg['ppo_acceptance'])
            # Match the existing conditional path runner: rejected hard-KL
            # epochs are restored and LR reduced; collect a fresh batch.
            if stop=='three_consecutive_hard_rejected_batches':stop=None
            if stop:raise RuntimeError(stop)
        elif result.get('nonfinite_stop'):raise RuntimeError('nonfinite value-only optimizer')
        if update<=0:
            writer.add_scalar('warmup/value_loss',result['value'],update+warmup)
            with (root/'critic_warmup.jsonl').open('a') as f:f.write(json.dumps(dict(batch=update+warmup,policy_steps=n*steps,optimizer=result))+'\n')
            if update==0:
                save_checkpoint(root/'checkpoints',algo,0,identity,route_counters=eps.episode)
                evaluate(0)
            continue
        statistic=host[0]['diagnostics']
        for x in host[1:]:statistic=merge_statistics(statistic,x['diagnostics'])
        diagnostic=summarize_statistics(statistic,cfg)
        reward_mean=float(np.mean([np.asarray(x['reward']) for x in host]));writer.add_scalar('train/mean_step_reward',reward_mean,update)
        for name in ('value','surrogate','mean_kl','actor_grad_norm','critic_grad_norm'):
            if name in result:writer.add_scalar('optimizer/'+name,result[name],update)
        writer.add_scalar('curriculum/stage',stage+1,update)
        writer.add_scalar('train/physical_failures',sum(int(x['physical_failures']) for x in host),update);writer.add_scalar('train/domain_failures',sum(int(x['domain_failures']) for x in host),update);writer.flush()
        if update==1:
            run='engineering' if smoke else 'ppo'
            if not _verify_reward_http(root.parent,tb,run):raise RuntimeError('current run reward HTTP verification failed')
        accepted_updates+=int(result.get('accepted_epochs',1)>0)
        record=dict(update=update,accepted_updates=accepted_updates,sampling_model_update=update-1,curriculum_stage=stage+1,mean_step_reward=reward_mean,optimizer=clean_numbers(result),sample_seconds=sample_seconds,optimize_seconds=time.monotonic()-started,diagnostics=diagnostic)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(clean_numbers(record))+'\n')
        if not smoke and update==total:
            atomic_json(root/'training_status.json',dict(state='completed',completed_updates=total,accepted_updates=accepted_updates))
            notice('STTW PPO training completed',f'{total}/{total} sampling slots finished. Final evaluation and plots are still running.')
        if smoke or update%cfg['curriculum']['evaluation_every']==0 or update==total:
            cp=save_checkpoint(root/'checkpoints',algo,update,identity,route_counters=eps.episode)
            atomic_json(root/'last_completed.json',dict(update=update,checkpoint=str(cp.resolve()),stage=stage))
            if not smoke:
                summary=evaluate(update);nxt_stage,streak=promotion(stage,update-stage_start,streak,summary,cfg['curriculum'])
                if nxt_stage!=stage and update<total:
                    notice('STTW curriculum advanced',f'Stage {stage+1} to {nxt_stage+1}; update {update}/{total}.')
                    stage=nxt_stage;stage_start=update;eps=reset(j.arange(n,dtype=j.int32),eps.episode+1,j.int32(stage));current=td(observe(eps))
        status(stage='training',completed_updates=update,accepted_updates=accepted_updates,declared_updates=total,curriculum_stage=stage+1)
        print(f'PPO {update}/{total} stage={stage+1} reward={reward_mean:.6g}',flush=True)
    writer.close();atomic_json(root/'status.json',dict(state='completed',completed_updates=total,curriculum_stage=stage+1,discarded=smoke))
    if not smoke:
        lines=['# Distilled alpha curriculum results','',f'{total} PPO sampling slots complete; stage advancement and qualification are separate.','']
        for stage,record in sorted(best.items()):lines.append(f"- Stage {stage+1}: evaluated best u{record['update']}, qualified={record['qualified']}; [panel]({Path(record['validation']).relative_to(root)}/INDEX.md)")
        (root/'INDEX.md').write_text('\n'.join(lines)+'\n')
    return algo
