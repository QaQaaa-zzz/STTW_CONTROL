"""Bounded SmoothV4 repair; reuse physical environment, rollout/GAE and prepared bank."""
from pathlib import Path
import json,pickle,time,hashlib,random,os,subprocess
import numpy as np
from .direct_command_training import Campaign,write,clean_numbers,flatten_logs,notify
from .direct_command_budget import ComputeBudget,BudgetStop
from .smooth_command_config import resolve

OLD=Path('/home/qy/STTW_CONTROL/runs/worktrees/frozen-lower-upper-endpoints/runs/frozen_R196_R244_upper_endpoints_250_20261008')

class SmoothCampaign(Campaign):
    def __init__(self,output,*,fresh=False,train_wall=1800,total_wall=5400,resume_parent=None,target_updates=None,unlimited_wall=False):
        self.fresh=fresh;self.train_wall=train_wall;self.total_wall=total_wall
        self.resume_parent=Path(resume_parent).resolve() if resume_parent else None
        def configured(alpha):
            spec=resolve(alpha,fresh=fresh,train_wall=train_wall,total_wall=total_wall)
            if self.resume_parent:
                if target_updates!=250:raise ValueError('this authorized continuation ends at250')
                spec['initialization_mode']='learner_state_resume_environment_reset'
                spec['ppo'].update(default_updates=250,future_total_updates_only_after_user_approval=250,policy_updates_per_endpoint=250,validation_updates=[250])
                spec['smooth_v4']['ppo'].update(policy_updates_per_endpoint=250,validation_updates=[250])
                spec['smooth_v4']['evaluation']['stage250_cases']=list(spec['smooth_v4']['evaluation']['stage60_cases'])
                spec['budget']['default_control_transitions_upper']=2*250*512*128*4
            if unlimited_wall:
                spec['budget']['wall_limits_enabled']=False
                spec['smooth_v4']['budget']['wall_limits_enabled']=False
                spec['smooth_v4']['budget']['user_amendment']='Stop at declared250 updates, no wall-clock budget termination'
            return spec
        self.resolve=configured
        self.out=Path(output);self.spec=self.resolve(0);self.config=Path('learning/configs/r196_smooth_alpha0.json')
        if (self.out/'manifest.json').exists():raise FileExistsError('no automatic restart of immutable run')
        self.budget=ComputeBudget(self.out,self.spec);self.budget.limits.update(train0=train_wall,train1=train_wall);self.budget.recover_interrupted()
        carry=self.out/'prior_attempt_budget.json'
        if carry.exists():
            prior=json.loads(carry.read_text())
            def apply(d):
                if d.get('prior_attempt_charged'):return
                d['seconds']={k:d['seconds'].get(k,0.)+v for k,v in prior['seconds'].items()};d['prior_attempt_charged']=True
                d['events'].append(dict(kind='prior_attempt_compute_charge',source=prior['source'],seconds=prior['seconds']))
            self.budget._update(apply)
        self.status=dict(state='initializing',completed_updates={'0':0,'1':0},declared_policy_batches_per_endpoint=self.spec['ppo']['default_updates'],initialization='scratch_actor_critic_optimizer_std' if fresh else 'actor_weights_only_fresh_critic_optimizer_std',pid=os.getpid())
        if self.resume_parent:self.status.update(initialization='learner_state_resume_environment_reset',parent_run=str(self.resume_parent))
        self.seen_cases=set();self.endpoints={};self._status()
    def initialize(self):
        import jax,jax.numpy as jp,torch
        from dataclasses import asdict
        from .direct_command_env import DirectCommandEnv
        from .runtime import configure_compilation_cache
        torch.set_num_threads(2);configure_compilation_cache(self.out/'jax_cache')
        with self.budget.measure('compile','load unchanged lower, physics and existing prepared bank'):
            self.env=DirectCommandEnv(self.spec)
            source=OLD/'R196_upper0/prepared_bank.pkl'
            with source.open('rb') as f:self.bank=jax.tree.map(jp.asarray,pickle.load(f))
            self.sample=jax.tree.map(lambda x:x[3],self.bank)
            self.sample_state=self.env.reset(self.sample,jp.int32(0),jp.int32(0),jp.asarray(0.));self.env.set_log_template(self.sample_state)
            jax.block_until_ready(self.sample_state)
        old=json.loads((OLD/'R196_upper0/manifest.json').read_text())
        assert asdict(self.env.cc)==old['controller']
        assert asdict(self.env.ac)==old['actuator']
        assert self.env.physics.bundle.identity==old['model']
        write(self.out/'manifest.json',dict(implementation_parent='6cdb3ce',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          prepared_bank=str(source),prepared_bank_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),lower=self.env.lower.provenance,
          initialization='Full upper learner from parent150; physical episode reset' if self.resume_parent else 'Fresh Actor Critic Adam std; no prior policy loaded' if self.fresh else 'Actor only from alpha0@250 and alpha1@143; Critic Adam std reset',
          budget=self.spec['smooth_v4']['budget'],new_preparation_ticks=0,lower_interfaces_reused=True,command_center='preserved governed-centered ECBC plus frozen lower residual',
          policy_budget_semantics=("global endpoint target250; resume parent completed batch, no new value-only warmup" if self.resume_parent else f"maximum{self.spec['ppo']['default_updates']} new PPO batches each; accepted epochs and updates separate; two value-only rollouts excluded")))
        for a in [0,1]:write(self.out/f'alpha{a}/frozen_config.json',self.resolve(a))
        self.states,self.advance,self.observe=self.setup_batch(512)
        self.component_names=sorted(list(self.env.zero_log['raw_components'])+['upper_rate','upper_acceleration'])
    def create_endpoint(self,alpha):
        if self.resume_parent:return self.resume_endpoint(alpha)
        import torch,jax,jax.numpy as jp
        from torch.utils.tensorboard import SummaryWriter
        from .direct_command_ppo import make_algorithm
        from .direct_command_policy import export_actor
        spec=self.resolve(alpha);stage=f'train{alpha}'
        with self.budget.measure(stage,'fresh Actor Critic Adam std' if self.fresh else 'Actor-only warm start, new Critic Adam std'):
            torch.manual_seed(81);torch.cuda.manual_seed_all(81);np.random.seed(81);random.seed(81)
            state=self.states.replace(alpha=jp.full((512,),float(alpha)))
            a,c,_,_=self.observe(state);obs=self.td(a,c)
            algo=make_algorithm(obs,128,spec,'cuda');algo.temporal_spec=spec
            np.testing.assert_allclose(algo.policy.log_std.exp().detach().cpu().numpy(),spec['ppo']['initial_latent_std'],rtol=1e-6)
            if self.fresh:
                src=None;digest=None
                assert not algo.optimizer.state
                assert torch.count_nonzero(algo.policy.actor[-1].weight)==0
                assert torch.count_nonzero(algo.policy.actor[-1].bias)==0
            else:
                u=[250,143][alpha];src=OLD/f'R196_upper{alpha}/pilot/checkpoints/update_{u:04d}.pt';actor=src.with_name(f'actor_{u:04d}.pkl')
                digest=hashlib.sha256(actor.read_bytes()).hexdigest();assert digest==spec['smooth_v4']['upper'][f'alpha{alpha}_source_actor_sha256']
                loaded=torch.load(src,map_location='cuda',weights_only=False)
                critic={k:v.detach().clone() for k,v in algo.policy.critic.state_dict().items()}
                algo.policy.actor.load_state_dict({k[6:]:v for k,v in loaded['policy'].items() if k.startswith('actor.')})
                assert all(torch.equal(v,algo.policy.critic.state_dict()[k]) for k,v in critic.items())
                assert not algo.optimizer.state
                with actor.open('rb') as f:expected=pickle.load(f)
                for x,y in zip(jax.tree.leaves(export_actor(algo.policy)),jax.tree.leaves(expected)):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
            ep=dict(algo=algo,state=state,obs=obs,update=0,warmup=0,hard_rejects=0,duration=0.,stopped=None,writer=SummaryWriter(str(self.out/'tensorboard'/f'alpha{alpha}')))
            self.endpoints[alpha]=ep
            write(self.out/f'alpha{alpha}/initialization.json',dict(initialization='scratch' if self.fresh else 'actor_only',source_checkpoint=None if src is None else str(src),source_actor_sha256=digest,actor_exact=None if self.fresh else True,previous_policy_loaded=not self.fresh,critic_fresh=True,optimizer_empty=True,std=algo.policy.log_std.exp().detach().cpu().tolist(),seed=81))
            self.save(alpha,0)
    def resume_endpoint(self,alpha):
        import torch,jax,jax.numpy as jp
        from torch.utils.tensorboard import SummaryWriter
        from .direct_command_ppo import make_algorithm
        last=json.loads((self.resume_parent/f'alpha{alpha}/last_completed.json').read_text())
        source=Path(last['checkpoint'])
        saved=torch.load(source,map_location='cuda',weights_only=False)
        if not 60<=saved['update']<=150 or saved['warmup']!=2:raise ValueError('requires parent checkpoint60..150 and original warmup')
        start_update=int(saved['update'])
        spec=self.resolve(alpha)
        for key in ['reward','action','network','commands','plant','limits','reference','lower_controller','lower_reference_centered','actor_temporal_regularizer']:
            if spec.get(key)!=saved['config'].get(key):raise ValueError('resume method mismatch '+key)
        # Parent checkpoints do not serialize physical/ESO/history state. Start
        # new complete episodes with unused episode keys, never pretend exact resume.
        manifest=self.resume_parent/f'alpha{alpha}/case_manifest.jsonl'
        episode=max(json.loads(line)['episode_index'] for line in manifest.read_text().splitlines())+1
        ids=jp.arange(512,dtype=jp.int32)
        def reset_one(env_id):
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(spec['commands']['random_seed']),env_id),episode)
            index=jax.random.randint(jax.random.fold_in(key,812),(),0,8)
            snap=jax.tree.map(lambda x:x[index],self.bank)
            return self.env.reset(snap,env_id,jp.int32(episode),jp.asarray(float(alpha)))
        reset=self.compile(f'alpha{alpha} continuation episode reset',jax.vmap(reset_one),ids)
        with self.budget.measure(f'train{alpha}','restore full upper learner and reset physical episodes'):
            state=reset(ids);jax.block_until_ready(state)
            a,c,_,_=self.observe(state);obs=self.td(a,c)
            algo=make_algorithm(obs,128,spec,'cuda');algo.temporal_spec=spec
            restore_learner(algo,saved)
            restore_rng(saved)
            ep=dict(algo=algo,state=state,obs=obs,update=start_update,warmup=2,hard_rejects=0,duration=0.,stopped=None,rng=capture_rng(),writer=SummaryWriter(str(self.out/'tensorboard'/f'alpha{alpha}')))
            self.endpoints[alpha]=ep
            write(self.out/f'alpha{alpha}/initialization.json',dict(initialization='learner_state_resume_environment_reset',source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),actor_critic_optimizer_std_restored=True,rng_restored=True,accepted_policy_updates=algo.accepted_policy_updates,physical_state_restored=False,new_episode_index=episode,first_new_update=start_update+1,additional_value_only_rollouts=0))
            self.save(alpha,start_update)
        self._status(completed_updates={str(a):e['update'] for a,e in self.endpoints.items()})

    @staticmethod
    def td(a,c):
        import torch
        from tensordict import TensorDict
        return TensorDict({'policy':torch.utils.dlpack.from_dlpack(a),'critic':torch.utils.dlpack.from_dlpack(c)},batch_size=[512])
    def save(self,alpha,update):
        import torch,jax
        from .direct_command_policy import export_actor
        ep=self.endpoints[alpha];algo=ep['algo'];root=self.out/f'alpha{alpha}/checkpoints';root.mkdir(exist_ok=True,parents=True)
        path=root/f'update_{update:04d}.pt'
        torch.save(dict(policy=algo.policy.state_dict(),optimizer=algo.optimizer.state_dict(),update=update,warmup=ep['warmup'],accepted_policy_updates=algo.accepted_policy_updates,config=self.resolve(alpha),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),initialization='learner_state_resume_environment_reset' if self.resume_parent else 'scratch' if self.fresh else 'actor_only_warm_start',environment_continuation='not an exact physics resume'),path.with_suffix('.tmp'))
        os.replace(path.with_suffix('.tmp'),path)
        with path.with_name(f'actor_{update:04d}.pkl').open('wb') as f:pickle.dump(jax.device_get(export_actor(algo.policy)),f)
        write(self.out/f'alpha{alpha}/last_completed.json',dict(update=update,warmup=ep['warmup'],accepted_policy_updates=algo.accepted_policy_updates,checkpoint=str(path.resolve()),stop=ep['stopped']))
    def batch(self,alpha,warmup=False):
        import torch,jax,jax.numpy as jp
        ep=self.endpoints[alpha]
        if 'rng' in ep:restore_rng(ep['rng'])
        algo=ep['algo'];stage=f'train{alpha}';index=ep['warmup']+1 if warmup else ep['update']+1
        estimate=ep['duration']*1.1
        if self.budget.remaining(stage)<=max(1.,estimate):ep['stopped']='predicted_training_budget';return False
        self._status(stage=stage,phase='value_only' if warmup else 'PPO',current_alpha=alpha,current_batch=index)
        sums=np.zeros((4,14+2*len(self.component_names)));rewards=[];fails=ends=0;start=time.monotonic();capcounts={k:0 for k in self.component_names};capden=0;diagnostics=[]
        snapshot=algo._snapshot()
        try:
            with self.budget.measure(stage,f"{'critic_only' if warmup else 'policy'} batch {index}",estimate=estimate):
                self.budget.reserve(512*128*4,f'alpha{alpha} batch{index} warmup={warmup}')
                with torch.no_grad():
                    for t in range(128):
                        obs=ep['obs']
                        if not torch.isfinite(obs['policy']).all() or not torch.isfinite(obs['critic']).all():raise RuntimeError('nonfinite observation')
                        z=algo.act(obs)
                        if not torch.isfinite(z).all():raise RuntimeError('nonfinite latent')
                        result=self.advance(ep['state'],jax.dlpack.from_dlpack(z.detach().contiguous()))
                        state,a,c,f,r,d,stat,failed,fault,peak,diag,final_obs=result
                        if bool(jp.any(f|fault)):raise RuntimeError('policy/lower fault')
                        ep['state']=state;ep['obs']=self.td(a,c)
                        algo.process_env_step(ep['obs'],torch.utils.dlpack.from_dlpack(r),torch.utils.dlpack.from_dlpack(d),{})
                        sums+=np.asarray(stat).sum(axis=0);rewards.append(np.asarray(r));fails+=int(jp.sum(failed));ends+=int(jp.sum(d))
                        if t==0 or bool(jp.any(d)):self.case_manifest(state,f'alpha{alpha}')
                        if warmup and index==1:diagnostics.append(jax.device_get({k:v for k,v in diag.items() if not k.startswith('all_')}))
                        for k,v in diag['all_component_cap_counts'].items():capcounts[k]+=int(v)
                        capden+=int(diag['all_active_ticks'])
                    algo.compute_returns(ep['obs'])
                if diagnostics:
                    flat=flatten_logs(jax.tree.map(lambda *x:np.stack(x),*diagnostics));np.savez_compressed(self.out/f'alpha{alpha}/first_rollout_diagnostic.npz',**flat)
                sample=time.monotonic()-start
                ret=algo.storage.returns;val=algo.storage.values
                variance=float(ret.var());explained=1-float((ret-val).var())/variance if variance>1e-12 else 0.
                metrics=algo.value_only_update(8) if warmup else algo.update()
                if warmup:ep['warmup']+=1
                else:
                    ep['update']+=1
                    ep['hard_rejects']=ep['hard_rejects']+1 if metrics['hard_kl_stop'] else 0
                    if metrics['hard_kl_stop']:
                        algo.optimizer.param_groups[0]['lr']=max(1e-5,algo.optimizer.param_groups[0]['lr']/2)
                        algo.hard_kl_stop=False
                    if metrics['nonfinite_stop']:ep['stopped']='nonfinite_optimizer'
                    if ep['hard_rejects']>=3:ep['stopped']='three_consecutive_hard_rejected_batches'
                groups={}
                for label,row in zip(['all','ordinary','conflict','recovery'],sums):
                    count=max(1.,row[0]);n=len(self.component_names)
                    groups[label]=dict(count=int(row[0]),speed_rmse=float(np.sqrt(row[1]/count)),steer_rmse=float(np.sqrt(row[2]/count)),heading_rmse=float(np.sqrt(row[11]/count)),working_roll_fraction=row[3]/count,any_state_cap_fraction=row[4]/count,motor_clip=row[7]/count,final_clip=row[8]/count,reference_clip=row[9]/count,reference_slew_fraction_diagnostic_only=row[10]/count,mean_offsets=(row[12:14]/count).tolist(),raw_costs=dict(zip(self.component_names,(row[14:14+n]/count).tolist())),effective_costs=dict(zip(self.component_names,(row[14+n:]/count).tolist())))
                rec=dict(alpha=alpha,phase='value_only' if warmup else 'policy',batch=index,completed_policy_batches=ep['update'],accepted_policy_updates=algo.accepted_policy_updates,mean_step_reward=float(np.mean(rewards)),failed_episodes=fails,ended_episodes=ends,failure_fraction=fails/max(1,ends),optimizer=metrics,explained_variance=explained,groups=groups,component_cap_fraction_all_envs={k:v/max(1,capden) for k,v in capcounts.items()},sample_seconds=sample,total_seconds=time.monotonic()-start)
                rec=clean_numbers(rec)
                with (self.out/f'alpha{alpha}/metrics.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
                step=index if not warmup else index-3;writer=ep['writer'];writer.add_scalar('train/mean_step_reward',rec['mean_step_reward'],step)
                for k,v in metrics.items():
                    if isinstance(v,(int,float)) and np.isfinite(v):writer.add_scalar('ppo/'+k,v,step)
                for k in ['speed_rmse','steer_rmse','heading_rmse','working_roll_fraction']:writer.add_scalar('physical/'+k,groups['all'][k],step)
                writer.add_scalar('physical/failure_fraction',rec['failure_fraction'],step);writer.flush()
                if warmup or index%10==0 or ep['stopped']:self.save(alpha,ep['update'])
                print(json.dumps(dict(alpha=alpha,phase=rec['phase'],batch=index,reward=rec['mean_step_reward'],kl=metrics.get('mean_kl'),seconds=rec['total_seconds'],stop=ep['stopped'])),flush=True)
        except BudgetStop:
            algo._restore(snapshot);algo.storage.clear();ep['stopped']='budget_interrupted_batch';self.save(alpha,ep['update']);return False
        if self.resume_parent:ep['rng']=capture_rng()
        ep['duration']=time.monotonic()-start
        self._status(completed_updates={str(a):e['update'] for a,e in self.endpoints.items()},current_stop=ep['stopped'])
        return not ep['stopped']
    def evaluate(self,stage):
        control=self.out/'runtime_control.json'
        if control.exists() and stage in json.loads(control.read_text()).get('disabled_evaluation_stages',[]):
            write(self.out/f'evaluation{stage}_skipped.json',dict(reason='explicit user runtime amendment',stage=stage));return
        import jax,jax.numpy as jp
        from .direct_command_policy import DirectCommandActor,export_actor
        from .fixed_command_panel import load_protocol,schedules
        protocol=load_protocol();cases=self.spec['smooth_v4']['evaluation'][f'stage{stage}_cases'];env=self.env
        params=jax.tree.map(lambda *x:jp.stack(x),*[export_actor(self.endpoints[a]['algo'].policy) for a in [0,1]])
        if not hasattr(self,'eval_chunk'):
            alphas=jp.array([0.,1.]);rows=jp.asarray(schedules(protocol)[0][1],jp.float32)
            reset=lambda rows:jax.vmap(lambda a:env.reset(self.sample,jp.int32(77001),jp.int32(0),a,rows,jp.array(protocol['slew'])))(alphas)
            self.eval_reset=self.compile('matched evaluation reset',reset,rows)
            with self.budget.measure('review','evaluation reset'):states=self.eval_reset(rows);jax.block_until_ready(states)
            actor=DirectCommandActor()
            def chunk(states,params):
                def step(states,_):
                    obs,_,fault,_=jax.vmap(env.observation)(states)
                    mu=jax.vmap(lambda p,o:actor.apply(p,o))(params,obs)
                    states=states.replace(fault=states.fault|(fault&~states.physical.failed))
                    end,r,done,logs,_=jax.vmap(env.policy_step)(states,mu)
                    logs['latent_mean']=jp.repeat(mu[:,None,:],4,axis=1)
                    return end,logs
                return jax.lax.scan(step,states,None,length=50)
            self.eval_chunk=self.compile('one-second paired frozen-case evaluation',chunk,states,params)
        for case,rows in schedules(protocol):
            if case not in cases:continue
            self._status(stage=f'evaluate{stage}',case=case)
            seconds=16 if stage>=60 and case=='fast_turn' else 10
            chunks=[]
            with self.budget.measure('review',f'update{stage} {case} paired {seconds}s'):
                states=self.eval_reset(jp.asarray(rows,jp.float32))
                for second in range(seconds):
                    states,log=self.eval_chunk(states,params);jax.block_until_ready(states);chunks.append(jax.device_get(log))
                    if bool(jp.any(states.fault)):raise RuntimeError('evaluation policy fault')
                flat=flatten_logs(jax.tree.map(lambda *x:np.concatenate(x),*chunks))
                for i in [0,1]:
                    d={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()};mask=d['active_tick'];d={k:v[mask] for k,v in d.items()}
                    d['checkpoint_update']=np.asarray(self.endpoints[i]['update']);d['partial']=np.asarray(False)
                    dest=self.out/f'evaluation{stage}'/case;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'alpha{i}.npz',**d)
            print(f'evaluation{stage} {case} saved',flush=True)
        from .smooth_command_reporting import report
        with self.budget.measure('review',f'stage{stage} independent reward audit and physical figures'):
            report(self.out,stage)
        notify('STTW SmoothV4评价阶段完成',f'update{stage}：固定场景已保存')

def run(output,**kwargs):
    c=SmoothCampaign(output,**kwargs)
    try:
        c.initialize()
        for a in [0,1]:c.create_endpoint(a)
        for stage in c.spec['ppo']['validation_updates']:
            for a in [0,1]:
                ep=c.endpoints[a]
                while ep['warmup']<2 and not ep['stopped']:
                    if not c.batch(a,True):break
                while ep['update']<stage and not ep['stopped']:
                    if not c.batch(a):break
                c.save(a,ep['update']);notify('STTW SmoothV4训练阶段结束',f'alpha{a}: {ep["update"]}/{c.spec["ppo"]["default_updates"]}; {ep["stopped"] or "stage complete"}')
            c.evaluate(stage)
            if all(e['stopped'] for e in c.endpoints.values()):break
        notify('STTW SmoothV4流水线结束','声明训练与固定场景对比已保存，停止，不自动续训')
        c._status(state='complete' if all(e['update']==c.spec['ppo']['default_updates'] for e in c.endpoints.values()) else 'partial',stage='saved_and_stopped',endpoint_stops={str(a):e['stopped'] for a,e in c.endpoints.items()})
    except BaseException as exc:
        c._status(state='budget_stopped' if isinstance(exc,BudgetStop) else 'error',reason=str(exc))
        notify('STTW SmoothV4停止',str(exc));raise
    finally:
        for e in c.endpoints.values():e['writer'].close()


def capture_rng():
    import torch
    return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())


def restore_rng(saved):
    import torch
    torch.set_rng_state(saved['torch_rng'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])


def restore_learner(algo,saved):
    """Resume all learned state; never substitute physical-best for training-last."""
    import torch
    algo.policy.load_state_dict(saved['policy']);algo.optimizer.load_state_dict(saved['optimizer'])
    algo.accepted_policy_updates=int(saved['accepted_policy_updates'])
    for k,v in saved['policy'].items():
        if not torch.equal(v,algo.policy.state_dict()[k]):raise ValueError('learner restore mismatch '+k)
    if not algo.optimizer.state:raise ValueError('missing parent optimizer state')
