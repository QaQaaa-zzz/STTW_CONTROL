"""Bounded SmoothV4 repair; reuse physical environment, rollout/GAE and prepared bank."""
from pathlib import Path
import json,pickle,time,hashlib,random,os,subprocess
import numpy as np
from .direct_command_training import Campaign,write,clean_numbers,flatten_logs,notify
from .direct_command_budget import ComputeBudget,BudgetStop
from .smooth_command_config import resolve,resolve_dimensions

OLD=Path('/home/qy/STTW_CONTROL/runs/worktrees/frozen-lower-upper-endpoints/runs/frozen_R196_R244_upper_endpoints_250_20261008')

class SmoothCampaign(Campaign):
    def __init__(self,output,*,fresh=False,train_wall=1800,total_wall=5400,resume_parent=None,target_updates=None,unlimited_wall=False,preference_v5=False,preference_v51=False,smoke=False,preference_stage2_parent=None,num_envs=None,rollout_steps=None,log_mode="evaluation_full",reset_guard=False):
        if preference_stage2_parent and (not preference_v5 or fresh or resume_parent or smoke):
            raise ValueError('V5 Stage2 continuation requires only --preference-v5 and a Stage1 parent')
        if (preference_v5 or preference_v51) and not preference_stage2_parent and (not fresh or resume_parent):
            raise ValueError('V5 Stage1 requires fresh independent uppers')
        self.reset_guard=reset_guard
        self.log_mode=log_mode
        if log_mode=="training_summary" and not preference_v51:raise ValueError("summary currently validated for V5.1 only")
        self.preference_v51=preference_v51
        self.preference_v5=preference_v5 or preference_v51;self.preference_stage2_parent=Path(preference_stage2_parent).resolve() if preference_stage2_parent else None
        self.training_stage=51 if preference_v51 else 2 if self.preference_stage2_parent else 1;self.smoke=smoke
        self.fresh=fresh;self.train_wall=train_wall;self.total_wall=total_wall
        self.resume_parent=Path(resume_parent).resolve() if resume_parent else None
        def configured(alpha):
            spec=resolve(alpha,fresh=fresh,train_wall=train_wall,total_wall=total_wall,preference_v5=preference_v5,preference_v51=preference_v51,stage=self.training_stage)
            if preference_v5 or preference_v51:
                spec['prepared_high_speed_indices']=getattr(self,'high_speed_indices',[5,6,7])
                return resolve_dimensions(spec,smoke=smoke,num_envs=num_envs,rollout_steps=rollout_steps)
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
            if num_envs is not None or rollout_steps is not None:
                raise ValueError('dimension overrides require preference mode')
            return resolve_dimensions(spec,smoke=smoke)
        self.resolve=configured
        self.out=Path(output);self.spec=self.resolve(0);self.config=Path('learning/configs/r196_smooth_alpha0.json')
        self.n=self.spec['ppo']['num_envs'];self.steps=self.spec['ppo']['rollout_policy_steps']
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
        completed={'0':40,'1':40} if self.preference_stage2_parent else {'0':0,'1':0}
        initialization='learner_state_resume_environment_reset' if self.preference_stage2_parent else 'scratch_actor_critic_optimizer_std' if fresh else 'actor_weights_only_fresh_critic_optimizer_std'
        self.status=dict(state='initializing',completed_updates=completed,declared_policy_batches_per_endpoint=self.spec['ppo']['default_updates'],initialization=initialization,pid=os.getpid())
        if self.resume_parent:self.status.update(initialization='learner_state_resume_environment_reset',parent_run=str(self.resume_parent))
        if self.preference_stage2_parent:self.status.update(parent_run=str(self.preference_stage2_parent),training_stage=2,stage1_gate_passed=False,user_override_stage2=True)
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
            if self.preference_v5:
                meta=json.loads((OLD/'R196_upper0/prepared_metrics.json').read_text())
                conditions=np.asarray(meta['conditions'])
                self.high_speed_indices=np.flatnonzero(np.isclose(conditions[:,0],2.6,atol=1e-6)).tolist()
                if len(self.high_speed_indices)!=3:raise ValueError('prepared bank high-speed metadata mismatch')
                self.spec['prepared_high_speed_indices']=self.high_speed_indices
                self.high_speed_zero=int(np.flatnonzero(np.isclose(conditions[:,0],2.6,atol=1e-6)&(conditions[:,1]==0))[0])
            self.sample=jax.tree.map(lambda x:x[3],self.bank)
            self.sample_state=self.env.reset(self.sample,jp.int32(0),jp.int32(0),jp.asarray(0.));self.env.set_log_template(self.sample_state)
            jax.block_until_ready(self.sample_state)
        old=json.loads((OLD/'R196_upper0/manifest.json').read_text())
        assert asdict(self.env.cc)==old['controller']
        assert asdict(self.env.ac)==old['actuator']
        assert self.env.physics.bundle.identity==old['model']
        write(self.out/'manifest.json',dict(implementation_parent='6cdb3ce',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          execution_dimensions=self.spec['execution_dimensions'],prepared_bank=str(source),prepared_bank_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),lower=self.env.lower.provenance,
          initialization='Full upper learner from parent150; physical episode reset' if self.resume_parent else 'Full Stage1 upper learner from update40; physical episode reset' if self.preference_stage2_parent else 'Fresh Actor Critic Adam std; no prior policy loaded' if self.fresh else 'Actor only from alpha0@250 and alpha1@143; Critic Adam std reset',
          budget=self.spec['budget'] if self.preference_v5 else self.spec['smooth_v4']['budget'],new_preparation_ticks=0,lower_interfaces_reused=True,command_center='preserved governed-centered ECBC plus frozen lower residual',
          policy_budget_semantics=("global endpoint target250; resume parent completed batch, no new value-only warmup" if self.resume_parent else f"maximum{self.spec['ppo']['default_updates']} new PPO batches each; accepted epochs and updates separate; two value-only rollouts excluded")))
        for a in [0,1]:
            write(self.out/f'alpha{a}/frozen_config.json',self.resolve(a))
            if self.preference_v5:write(self.out/f'alpha{a}/frozen_config_stage{self.training_stage}.json',self.resolve(a))
        if self.preference_v5:
            manifest=json.loads((self.out/'manifest.json').read_text())
            semantics=('V5.1 scratch 16-second mixed task from first batch; stop at100; no value warmup' if self.preference_v51 else
                       'Stage2 user override after failed Stage1 gate; resume cumulative40 and stop at120; no value warmup' if self.preference_stage2_parent else
                       'Stage1 max40 each; only paired gate pass permits Stage2 cumulative120; no value warmup')
            manifest.update(implementation_parent='e1a6cc1f5a57b90b780207a69ce5bfa543cc1642',policy_budget_semantics=semantics,prepared_high_speed_indices=self.high_speed_indices,prepared_high_speed_zero_index=self.high_speed_zero,
                            parent_run=None if not self.preference_stage2_parent else str(self.preference_stage2_parent),stage1_gate_passed=None if not self.preference_stage2_parent else False,user_override_stage2=bool(self.preference_stage2_parent),priority_recovery_v51=self.preference_v51)
            write(self.out/'manifest.json',manifest)
        self.states,self.advance,self.observe=self.setup_batch(self.n,log_mode=self.log_mode,reset_guard=self.reset_guard)
        self.component_names=sorted(list(self.env.zero_log['raw_components'])+['upper_rate','upper_acceleration'])
        if self.preference_stage2_parent:
            self.stage2_parent_receipt=validate_preference_stage2_parent(self.preference_stage2_parent)
            write(self.out/'stage_transition.json',dict(**self.stage2_parent_receipt,from_stage=1,to_stage=2,actor_critic_adam_std_rng_preserved=True,physics_eso_filter_action_history_reset=True,reward_formula_unchanged=True,user_override_reason='User explicitly requested Stage2 after the declared Stage1 paired gate failed'))
    def create_endpoint(self,alpha):
        if self.preference_stage2_parent:return self.resume_preference_stage2_endpoint(alpha)
        if self.resume_parent:return self.resume_endpoint(alpha)
        import torch,jax,jax.numpy as jp
        from torch.utils.tensorboard import SummaryWriter
        from .direct_command_ppo import make_algorithm
        from .direct_command_policy import export_actor
        spec=self.resolve(alpha);stage=f'train{alpha}'
        with self.budget.measure(stage,'fresh Actor Critic Adam std' if self.fresh else 'Actor-only warm start, new Critic Adam std'):
            seed=spec['ppo']['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
            state=self.states.replace(alpha=jp.full((self.n,),float(alpha)))
            a,c,_,_=self.observe(state);obs=self.td(a,c)
            algo=make_algorithm(obs,self.steps,spec,'cuda');algo.temporal_spec=spec
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
            ep['rng']=capture_rng()
            self.endpoints[alpha]=ep
            write(self.out/f'alpha{alpha}/initialization.json',dict(initialization='scratch' if self.fresh else 'actor_only',source_checkpoint=None if src is None else str(src),source_actor_sha256=digest,actor_exact=None if self.fresh else True,previous_policy_loaded=not self.fresh,critic_fresh=True,optimizer_empty=True,std=algo.policy.log_std.exp().detach().cpu().tolist(),seed=seed))
            self.save(alpha,0)
    def resume_preference_stage2_endpoint(self,alpha):
        import torch,jax.numpy as jp
        from torch.utils.tensorboard import SummaryWriter
        from .direct_command_ppo import make_algorithm
        receipt=self.stage2_parent_receipt
        source=Path(receipt['endpoints'][str(alpha)]['checkpoint'])
        saved=torch.load(source,map_location='cuda',weights_only=False)
        spec=self.resolve(alpha)
        for key in ['reward','action','network','plant','limits','reference','lower_controller','lower_reference_centered','actor_temporal_regularizer']:
            if spec.get(key)!=saved['config'].get(key):raise ValueError('V5 Stage2 resume method mismatch '+key)
        with self.budget.measure(f'train{alpha}','restore full Stage1 upper learner and reset Stage2 physical episodes'):
            state=self.states.replace(alpha=jp.full((self.n,),float(alpha)))
            a,c,_,_=self.observe(state);obs=self.td(a,c)
            algo=make_algorithm(obs,self.steps,spec,'cuda');algo.temporal_spec=spec
            restore_learner(algo,saved)
            ep=dict(algo=algo,state=state,obs=obs,update=40,warmup=0,hard_rejects=0,duration=0.,stopped=None,rng={k:saved[k] for k in ('torch_rng','cuda_rng','numpy_rng','python_rng')},writer=SummaryWriter(str(self.out/'tensorboard'/f'alpha{alpha}')))
            self.endpoints[alpha]=ep
            write(self.out/f'alpha{alpha}/initialization.json',dict(initialization='learner_state_resume_environment_reset',source_checkpoint=str(source),source_sha256=receipt['endpoints'][str(alpha)]['sha256'],actor_critic_optimizer_std_restored=True,rng_restored_on_first_batch=True,accepted_policy_updates=algo.accepted_policy_updates,physical_state_restored=False,first_new_update=41,additional_value_only_rollouts=0,stage1_gate_passed=False,user_override_stage2=True))
            restore_rng(ep['rng'])
            self.save(alpha,40)
        self._status(completed_updates={str(a):e['update'] for a,e in self.endpoints.items()})
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
            algo=make_algorithm(obs,self.steps,spec,'cuda');algo.temporal_spec=spec
            restore_learner(algo,saved)
            restore_rng(saved)
            ep=dict(algo=algo,state=state,obs=obs,update=start_update,warmup=2,hard_rejects=0,duration=0.,stopped=None,rng=capture_rng(),writer=SummaryWriter(str(self.out/'tensorboard'/f'alpha{alpha}')))
            self.endpoints[alpha]=ep
            write(self.out/f'alpha{alpha}/initialization.json',dict(initialization='learner_state_resume_environment_reset',source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),actor_critic_optimizer_std_restored=True,rng_restored=True,accepted_policy_updates=algo.accepted_policy_updates,physical_state_restored=False,new_episode_index=episode,first_new_update=start_update+1,additional_value_only_rollouts=0))
            self.save(alpha,start_update)
        self._status(completed_updates={str(a):e['update'] for a,e in self.endpoints.items()})

    def td(self,a,c):
        import torch
        from tensordict import TensorDict
        return TensorDict({'policy':torch.utils.dlpack.from_dlpack(a),'critic':torch.utils.dlpack.from_dlpack(c)},batch_size=[self.n])
    def save(self,alpha,update):
        import torch,jax
        from .direct_command_policy import export_actor
        ep=self.endpoints[alpha];algo=ep['algo'];root=self.out/f'alpha{alpha}/checkpoints';root.mkdir(exist_ok=True,parents=True)
        path=root/f'update_{update:04d}.pt'
        torch.save(dict(policy=algo.policy.state_dict(),optimizer=algo.optimizer.state_dict(),update=update,warmup=ep['warmup'],accepted_policy_updates=algo.accepted_policy_updates,config=self.resolve(alpha),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),initialization='learner_state_resume_environment_reset' if self.resume_parent or self.preference_stage2_parent else 'scratch' if self.fresh else 'actor_only_warm_start',environment_continuation='not an exact physics resume'),path.with_suffix('.tmp'))
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
        sums=np.zeros((4,14+2*len(self.component_names)));family_sums=np.zeros((4,14+2*len(self.component_names)));rewards=[];fails=ends=0;start=time.monotonic();capcounts={k:0 for k in self.component_names};capden=0;diagnostics=[]
        snapshot=algo._snapshot()
        summary_mode=self.log_mode=='training_summary'
        if summary_mode:
            from .smooth_device_statistics import initialize,begin_rollout,accumulate
            if not hasattr(self,'accumulate_statistics'):self.accumulate_statistics=jax.jit(accumulate)
            carry=begin_rollout(ep.get('device_statistics',initialize(self.n,14+2*len(self.component_names),len(self.component_names))))
            events=[];reset_events=[]
            # The first entry is the actual starting state, before any advance.
            self.case_manifest(ep['state'],f'alpha{alpha}')
        try:
            with self.budget.measure(stage,f"{'critic_only' if warmup else 'policy'} batch {index}",estimate=estimate):
                self.budget.reserve(self.n*self.steps*4,f'alpha{alpha} batch{index} warmup={warmup}')
                with torch.no_grad():
                    for t in range(self.steps):
                        obs=ep['obs']
                        current_families=ep['state'].family if summary_mode else np.asarray(ep['state'].family)
                        if not torch.isfinite(obs['policy']).all() or not torch.isfinite(obs['critic']).all():raise RuntimeError('nonfinite observation')
                        z=algo.act(obs)
                        if not torch.isfinite(z).all():raise RuntimeError('nonfinite latent')
                        result=self.advance(ep['state'],jax.dlpack.from_dlpack(z.detach().contiguous()))
                        state,a,c,f,r,d,stat,failed,fault,peak,diag,final_obs=result
                        if bool(jp.any(f|fault)):raise RuntimeError('policy/lower fault')
                        ep['state']=state;ep['obs']=self.td(a,c)
                        algo.process_env_step(ep['obs'],torch.utils.dlpack.from_dlpack(r),torch.utils.dlpack.from_dlpack(d),{})
                        if summary_mode:
                            carry,event=self.accumulate_statistics(carry,r,d,failed,stat,current_families,
                                jp.stack([diag['all_component_cap_counts'][k] for k in self.component_names]),diag['all_active_ticks'],
                                diag['all_proposal_dv'],diag['all_governed_dv'],diag['all_valid_ticks'])
                            events.append(event);reset_events.append(diag['reset_event'])
                        else:
                            if self.preference_v5:self.record_episode_statistics(alpha,np.asarray(r),np.asarray(d),np.asarray(failed),np.asarray(stat),jax.device_get(diag))
                            stat_np=np.asarray(stat);sums+=stat_np.sum(axis=0)
                            for family_id in range(4):family_sums[family_id]+=stat_np[current_families==family_id,0].sum(axis=0)
                            rewards.append(np.asarray(r));fails+=int(jp.sum(failed));ends+=int(jp.sum(d))
                            if t==0 or bool(jp.any(d)):self.case_manifest(state,f'alpha{alpha}')
                            if warmup and index==1:diagnostics.append(jax.device_get({k:v for k,v in diag.items() if not k.startswith('all_')}))
                            for k,v in diag['all_component_cap_counts'].items():capcounts[k]+=int(v)
                            capden+=int(diag['all_active_ticks'])
                    algo.compute_returns(ep['obs'])
                if summary_mode:
                    if len(events)!=self.steps:raise RuntimeError('incomplete statistics event buffer')
                    ep['device_statistics']=carry
                    host,episode_events,resets=jax.device_get((carry,jax.tree.map(lambda *x:jp.stack(x),*events),jax.tree.map(lambda *x:jp.stack(x),*reset_events)))
                    sums=host['sums'];family_sums=host['family_sums'];rewards=[host['reward_sum']/(self.n*self.steps)]
                    fails=int(host['fails']);ends=int(host['ends']);capden=int(host['capden']);capcounts=dict(zip(self.component_names,host['capcounts'].tolist()))
                    ep['negative_covered']=host['negative_covered'];ep['coverage_completed']=int(host['coverage_completed'])
                    for t,i in zip(*np.nonzero(episode_events['done'])):
                        row=episode_events['stats'][t,i];count=max(row[0],1)
                        ep.setdefault('completed_episode_batch',[]).append(dict(return_sum=float(episode_events['return_sum'][t,i]),
                            policy_steps=int(episode_events['steps'][t,i]),physical_failure=bool(episode_events['failed'][t,i]),
                            speed_rmse=float(np.sqrt(row[1]/count)),steer_rmse=float(np.sqrt(row[2]/count)),heading_rmse=float(np.sqrt(row[11]/count)),training_stage=self.training_stage))
                    self.write_reset_events(resets,f'alpha{alpha}')
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
                def summarize(row):
                    count=max(1.,row[0]);n=len(self.component_names)
                    return dict(count=int(row[0]),speed_rmse=float(np.sqrt(row[1]/count)),steer_rmse=float(np.sqrt(row[2]/count)),heading_rmse=float(np.sqrt(row[11]/count)),working_roll_fraction=row[3]/count,any_state_cap_fraction=row[4]/count,motor_clip=row[7]/count,final_clip=row[8]/count,reference_clip=row[9]/count,reference_slew_fraction_diagnostic_only=row[10]/count,mean_offsets=(row[12:14]/count).tolist(),raw_costs=dict(zip(self.component_names,(row[14:14+n]/count).tolist())),effective_costs=dict(zip(self.component_names,(row[14+n:]/count).tolist())))
                groups={label:summarize(row) for label,row in zip(['all','ordinary','conflict','recovery'],sums)}
                families={label:summarize(row) for label,row in zip(['nominal','conflict','random','heading_recovery_start'],family_sums)}
                rec=dict(alpha=alpha,phase='value_only' if warmup else 'policy',batch=index,completed_policy_batches=ep['update'],accepted_policy_updates=algo.accepted_policy_updates,mean_step_reward=float(np.mean(rewards)),failed_episodes=fails,ended_episodes=ends,failure_fraction=fails/max(1,ends),optimizer=metrics,explained_variance=explained,groups=groups,families=families,component_cap_fraction_all_envs={k:v/max(1,capden) for k,v in capcounts.items()},sample_seconds=sample,total_seconds=time.monotonic()-start)
                if self.preference_v5:
                    rec.update(training_stage=self.training_stage,complete_episode_statistics=ep.pop('completed_episode_batch',[]),negative_request_coverage=self.coverage_summary(ep))
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
        if self.resume_parent or self.preference_v5:ep['rng']=capture_rng()
        ep['duration']=time.monotonic()-start
        self._status(completed_updates={str(a):e['update'] for a,e in self.endpoints.items()},current_stop=ep['stopped'])
        return not ep['stopped']
    def notify_once(self,event,title,body):
        key='stage_notified' if event.startswith('training_') else 'completion_notified'
        seen=self.status.get(key,[])
        if event in seen:return
        notify(title,body);self._status(**{key:seen+[event]})

    def record_episode_statistics(self,alpha,reward,done,failed,stat,diag):
        ep=self.endpoints[alpha]
        if 'episode_return' not in ep:
            ep.update(episode_return=np.zeros(self.n),episode_steps=np.zeros(self.n,int),episode_stats=np.zeros((self.n,stat.shape[-1])),negative_dwell=np.zeros((self.n,2,2),int),negative_seen=np.zeros((self.n,2,2),bool),negative_covered=np.zeros((2,2),int),coverage_completed=0)
        ep['episode_return']+=reward;ep['episode_steps']+=1;ep['episode_stats']+=stat[:,0]
        for tick in range(4):
            values=np.stack([diag['all_proposal_dv'],diag['all_governed_dv'][:,tick]],axis=1)
            mask=(values[:,:,None]<np.array([-.15,-.30]))&diag['all_valid_ticks'][:,tick,None,None]
            ep['negative_dwell']=np.where(mask,ep['negative_dwell']+1,0)
            ep['negative_seen']|=ep['negative_dwell']>=40
        for i in np.flatnonzero(done):
            row=ep['episode_stats'][i];count=max(row[0],1)
            rec=dict(return_sum=float(ep['episode_return'][i]),policy_steps=int(ep['episode_steps'][i]),physical_failure=bool(failed[i]),speed_rmse=float(np.sqrt(row[1]/count)),steer_rmse=float(np.sqrt(row[2]/count)),heading_rmse=float(np.sqrt(row[11]/count)),training_stage=self.training_stage)
            ep.setdefault('completed_episode_batch',[]).append(rec)
            ep['negative_covered']+=ep['negative_seen'][i];ep['coverage_completed']+=1
            ep['episode_return'][i]=0;ep['episode_steps'][i]=0;ep['episode_stats'][i]=0;ep['negative_dwell'][i]=0;ep['negative_seen'][i]=False

    @staticmethod
    def coverage_summary(ep):
        n=ep.get('coverage_completed',0)
        return dict(complete_episodes=n,request_and_governed_thresholds=[-.15,-.30],minimum_continuous_s=.2,covered_counts=ep.get('negative_covered',np.zeros((2,2),int)).tolist(),fraction=None if not n else (ep['negative_covered']/n).tolist())

    def switch_preference_stage2(self):
        import jax,jax.numpy as jp
        from .direct_command_env import DirectCommandEnv
        self.training_stage=2;self.spec=self.resolve(0);self.env=DirectCommandEnv(self.spec)
        self.sample_state=self.env.reset(self.sample,jp.int32(0),jp.int32(0),jp.asarray(0.));self.env.set_log_template(self.sample_state)
        self.states,self.advance,self.observe=self.setup_batch(self.n)
        for a,ep in self.endpoints.items():
            write(self.out/f'alpha{a}/stage1_partial_episodes_at_transition.json',dict(discarded_partial_episode_steps=ep.get('episode_steps',np.zeros(self.n,int)).tolist(),not_counted_as_complete=True))
            for k in ['episode_return','episode_steps','episode_stats','negative_dwell','negative_seen','negative_covered','coverage_completed']:ep.pop(k,None)
            ep['state']=self.states.replace(alpha=jp.full((self.n,),float(a)))
            x,y,_,_=self.observe(ep['state']);ep['obs']=self.td(x,y)
            ep['algo'].storage.clear();ep['algo'].temporal_spec=self.resolve(a)
            write(self.out/f'alpha{a}/frozen_config_stage2.json',self.resolve(a))
        write(self.out/'stage_transition.json',dict(from_stage=1,to_stage=2,actor_critic_adam_rng_preserved=True,accepted_updates={str(a):ep['algo'].accepted_policy_updates for a,ep in self.endpoints.items()},physics_histories_reset=True))
        self._status(training_stage=2,declared_policy_batches_per_endpoint=120)

    def evaluate_preference(self,update,case_filter=None):
        import jax,jax.numpy as jp
        from .direct_command_policy import DirectCommandActor,export_actor
        from .fixed_command_panel import load_protocol,schedules
        from .preference_command_reporting import report_stage1,report_stage2
        env=self.env;cfg=self.spec['preference_v5']['evaluation'];actor=DirectCommandActor()
        reuse_baseline=self.training_stage==1 and update==40
        methods=['alpha0','alpha1'] if reuse_baseline else ['alpha0','alpha1','B0']
        indices=[0,1] if reuse_baseline else [0,1,0]
        params=jax.tree.map(lambda *x:jp.stack(x),*[export_actor(self.endpoints[a]['algo'].policy) for a in indices])
        alphas=jp.asarray(indices,dtype=jp.float32);bypass=jp.array([m=='B0' for m in methods])
        sample=jax.tree.map(lambda x:x[self.high_speed_zero],self.bank) if self.training_stage==1 else self.sample
        def reset(rows):return jax.vmap(lambda a:env.reset(sample,jp.int32(77001),jp.int32(0),a,rows,jp.array([.5,.3])))(alphas)
        if self.training_stage==1:
            cases=[]
            for name,r in cfg['stage1_pair_cases'].items():
                rows=np.zeros((16,3));rows[:,0]=99.;rows[:len(r)]=r;cases.append((name,rows,5))
        else:
            cases=[(name,rows,16 if name=='fast_turn' else 10) for name,rows in schedules(load_protocol())]
            if self.preference_v51 and update==25:
                cases=[row for row in cases if row[0] in ('straight_hold','fast_turn')]
        if case_filter is not None:
            wanted=set(case_filter);cases=[row for row in cases if row[0] in wanted]
            found={row[0] for row in cases}
            if found!=wanted:raise ValueError(f'unknown preference evaluation cases: {sorted(wanted-found)}')
        if not cases:raise ValueError('preference evaluation requires at least one case')
        reset=self.compile(f'V5 stage{self.training_stage} paired reset',reset,jp.asarray(cases[0][1],jp.float32))
        states=reset(jp.asarray(cases[0][1],jp.float32))
        def chunk(states,params):
            def step(states,_):
                obs,_,fault,_=jax.vmap(env.observation)(states)
                mu=jax.vmap(lambda p,o:actor.apply(p,o))(params,obs);mu=jp.where(bypass[:,None],jp.zeros_like(mu),mu)
                states=states.replace(fault=states.fault|(fault&~states.physical.failed))
                end,r,done,logs,_=jax.vmap(lambda st,z:env.policy_step(st,z,False))(states,mu)
                return end,logs
            return jax.lax.scan(step,states,None,length=50)
        chunk=self.compile(f'V5 stage{self.training_stage} paired 1s physical evaluation',chunk,states,params)
        for case,rows,seconds in cases:
            self._status(stage=f'evaluate{update}',case=case)
            with self.budget.measure('review',f'update{update}/{case}'):
                states=reset(jp.asarray(rows,jp.float32));chunks=[]
                for _ in range(seconds):
                    states,logs=chunk(states,params);jax.block_until_ready(states);chunks.append(jax.device_get(logs))
                    if bool(jp.any(states.fault)):raise RuntimeError('evaluation nonfinite policy or lower')
                flat=flatten_logs(jax.tree.map(lambda *x:np.concatenate(x),*chunks))
                dest=self.out/f'evaluation{update}'/case;dest.mkdir(parents=True,exist_ok=True)
                for i,name in enumerate(methods):
                    d={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()};valid=d['active_tick'];d={k:v[valid] for k,v in d.items()}
                    d.update(checkpoint_update=np.asarray(update),partial=np.asarray(False))
                    np.savez_compressed(dest/f'{name}.npz',**d)
            if reuse_baseline:
                source=self.out/'evaluation20'/case/'B0.npz'
                if not source.exists():raise FileNotFoundError(source)
                (dest/'B0.npz').symlink_to(source.resolve())
            print(f'V5 evaluation{update} {case} saved',flush=True)
        result=report_stage1(self.out,update) if self.training_stage==1 else report_stage2(self.out,update)
        notify('STTW V5评价阶段完成',f'update{update} 图与门槛结果已保存')
        return result

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


def validate_preference_stage2_parent(parent):
    """Validate the immutable Stage1 endpoint checkpoints before an explicit override."""
    import torch
    parent=Path(parent).resolve()
    status=json.loads((parent/'status.json').read_text())
    if status.get('state')!='complete' or status.get('training_stage')!=1:
        raise ValueError('V5 Stage2 parent must be a completed Stage1 run')
    if status.get('completed_updates')!={'0':40,'1':40}:
        raise ValueError('V5 Stage2 parent endpoints must both be exactly 40 updates')
    if status.get('stage')!='stage1_gate_failed_stopped' or status.get('stage1_gate_passed') is not False or status.get('stage2_started') is not False:
        raise ValueError('V5 Stage2 override requires the preserved failed-gate terminal status')
    endpoints={}
    for alpha in (0,1):
        last=json.loads((parent/f'alpha{alpha}/last_completed.json').read_text())
        source=Path(last['checkpoint']).resolve()
        if source.parent.parent!=parent/f'alpha{alpha}' or not source.exists():
            raise ValueError(f'alpha{alpha} checkpoint does not belong to parent run')
        saved=torch.load(source,map_location='cpu',weights_only=False)
        if any(int(saved.get(k,-1))!=40 for k in ('update','accepted_policy_updates')) or int(saved.get('warmup',-1))!=0:
            raise ValueError(f'alpha{alpha} parent learner must be exactly update40 with no warmup')
        if last.get('stop') is not None or int(last.get('update',-1))!=40 or int(last.get('accepted_policy_updates',-1))!=40:
            raise ValueError(f'alpha{alpha} last-completed receipt mismatch')
        if saved.get('config',{}).get('preference_v5_stage')!=1:
            raise ValueError(f'alpha{alpha} is not a V5 Stage1 checkpoint')
        if not saved.get('policy') or not saved.get('optimizer',{}).get('state'):
            raise ValueError(f'alpha{alpha} missing full learner state')
        if any(k not in saved for k in ('torch_rng','cuda_rng','numpy_rng','python_rng')):
            raise ValueError(f'alpha{alpha} missing RNG state')
        endpoints[str(alpha)]=dict(checkpoint=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),update=40,warmup=0,accepted_policy_updates=40,optimizer_state_present=True,rng_state_present=True)
    return dict(parent_run=str(parent),parent_terminal_stage=status['stage'],stage1_gate_passed=False,user_override_required=True,endpoints=endpoints)


def run_preference(output,smoke=False):
    """Execute the user-declared gated V5 budget, never extend a failed pilot."""
    c=SmoothCampaign(output,fresh=True,preference_v5=True,smoke=smoke,unlimited_wall=True)
    try:
        c.initialize()
        for a in ([0] if smoke else [0,1]):c.create_endpoint(a)
        if smoke:
            for _ in range(2):
                if not c.batch(0):raise RuntimeError('V5 finite smoke stopped')
            c.save(0,c.endpoints[0]['update'])
            write(c.out/'smoke_result.json',dict(passed=True,environments=8,policy_steps=16,updates=2,policy_transitions=256,control_ticks=1024,scope='engineering only, not coordination evidence'))
            c._status(state='complete',stage='smoke_complete');return
        gate=None
        for update in [20,40]:
            for a in [0,1]:
                ep=c.endpoints[a]
                while ep['update']<update:
                    if not c.batch(a):raise RuntimeError(f'alpha{a} stopped: '+str(ep['stopped']))
                c.save(a,ep['update']);c.notify_once(f'training_{a}_{update}','STTW V5训练阶段结束',f'alpha{a} Stage1 {update}/40，尚待左右配对评价')
            gate=c.evaluate_preference(update)
        if not gate['passed']:
            c._status(state='complete',stage='stage1_gate_failed_stopped',training_stage=1,stage1_gate_passed=False,stage2_started=False,qualified=False)
            c.notify_once('pipeline_complete','STTW V5流水线结束','Stage1未通过左右门槛；停止扩训，不进入Stage2')
            return
        c.switch_preference_stage2()
        for update in [80,120]:
            for a in [0,1]:
                ep=c.endpoints[a]
                while ep['update']<update:
                    if not c.batch(a):raise RuntimeError(f'alpha{a} stopped: '+str(ep['stopped']))
                c.save(a,ep['update']);c.notify_once(f'training_{a}_{update}','STTW V5训练阶段结束',f'alpha{a} 累计{update}/120，尚待完整任务评价')
            c.evaluate_preference(update)
        c._status(state='complete',stage='saved_and_stopped',training_stage=2,stage1_gate_passed=True)
        c.notify_once('pipeline_complete','STTW V5流水线结束','声明120更新及任务评价完成；不自动追加')
    except BaseException as exc:
        c._status(state='error',reason=str(exc));notify('STTW V5错误停止',str(exc));raise
    finally:
        for ep in c.endpoints.values():ep['writer'].close()


def run_preference_stage2(output,parent):
    """Continue failed-gate Stage1 learners only after the user's explicit override."""
    c=SmoothCampaign(output,preference_v5=True,preference_stage2_parent=parent,unlimited_wall=True)
    try:
        c.initialize()
        for a in [0,1]:c.create_endpoint(a)
        c._status(state='running',stage='stage2_training',training_stage=2,stage1_gate_passed=False,stage2_started=True,user_override_stage2=True)
        for update in [80,120]:
            for a in [0,1]:
                ep=c.endpoints[a]
                while ep['update']<update:
                    if not c.batch(a):raise RuntimeError(f'alpha{a} stopped: '+str(ep['stopped']))
                c.save(a,ep['update']);c.notify_once(f'training_{a}_{update}','STTW V5训练阶段结束',f'alpha{a} Stage2累计{update}/120，尚待完整任务评价')
            c.evaluate_preference(update)
        c._status(state='complete',stage='saved_and_stopped',training_stage=2,stage1_gate_passed=False,stage2_started=True,user_override_stage2=True,qualified=None)
        c.notify_once('pipeline_complete','STTW V5流水线结束','Stage2声明120更新及任务评价完成；不自动追加')
    except BaseException as exc:
        c._status(state='error',reason=str(exc),stage1_gate_passed=False,user_override_stage2=True);notify('STTW V5错误停止',str(exc));raise
    finally:
        for ep in c.endpoints.values():ep['writer'].close()


def run_preference_v51(output,smoke=False):
    """Execute the supplied fresh V5.1 16-second, 100-update endpoint campaign."""
    c=SmoothCampaign(output,fresh=True,preference_v51=True,smoke=smoke,unlimited_wall=True)
    try:
        c.initialize()
        for a in ([0] if smoke else [0,1]):c.create_endpoint(a)
        if smoke:
            for _ in range(2):
                if not c.batch(0):raise RuntimeError('V5.1 finite engineering run stopped')
            c.save(0,c.endpoints[0]['update'])
            write(c.out/'smoke_result.json',dict(passed=True,environments=8,policy_steps=16,updates=2,
                  policy_transitions=256,control_ticks=1024,scope='engineering only; official learners are initialized separately'))
            c._status(state='complete',stage='engineering_check_complete');return
        c._status(state='running',stage='v51_training',training_stage=51)
        for update in [25,100]:
            for a in [0,1]:
                ep=c.endpoints[a]
                while ep['update']<update:
                    if not c.batch(a):raise RuntimeError(f'alpha{a} stopped: '+str(ep['stopped']))
                c.save(a,ep['update']);c.notify_once(f'training_{a}_{update}','STTW V5.1训练阶段结束',f'alpha{a} {update}/100，等待固定协议评价')
            result=c.evaluate_preference(update)
            audits=[payload[f'alpha{endpoint}']['reward_audit'] for payload in result.values() for endpoint in (0,1)]
            failed=[audit for audit in audits if audit.get('passed') is not True]
            if failed:raise RuntimeError(f'V5.1 independent reward reconstruction failed at update{update}: {failed}')
        c._status(state='complete',stage='saved_and_stopped',training_stage=51,qualified=None)
        c.notify_once('pipeline_complete','STTW V5.1流水线结束','声明100更新及评价完成；不自动追加')
    except BaseException as exc:
        c._status(state='error',reason=str(exc),training_stage=51);notify('STTW V5.1错误停止',str(exc));raise
    finally:
        for ep in c.endpoints.values():ep['writer'].close()


def recover_preference_v51_evaluation(parent):
    """Finish missing update100 evidence from immutable completed V5.1 learners."""
    import torch
    parent=Path(parent).resolve()
    original_status=json.loads((parent/'status.json').read_text())
    if original_status.get('completed_updates')!={'0':100,'1':100}:
        raise ValueError('V5.1 evaluation recovery requires two completed update100 learners')
    if original_status.get('training_stage')!=51:
        raise ValueError('parent is not a V5.1 run')
    receipt=parent/'evaluation100_error_receipt.json'
    if not receipt.exists():write(receipt,original_status)
    runtime=parent/f'evaluation_recovery_runtime_{int(time.time())}'
    c=SmoothCampaign(runtime,fresh=True,preference_v51=True,unlimited_wall=True)
    try:
        c.initialize()
        for alpha in (0,1):
            source=parent/f'alpha{alpha}/checkpoints/update_0100.pt'
            saved=torch.load(source,map_location='cuda',weights_only=False)
            if int(saved.get('update', -1))!=100 or int(saved.get('accepted_policy_updates', -1))!=100:
                raise ValueError(f'alpha{alpha} source is not the completed update100 learner')
            spec=c.resolve(alpha)
            for key in ['reward','action','network','plant','limits','reference','lower_controller','lower_reference_centered','actor_temporal_regularizer']:
                if spec.get(key)!=saved.get('config',{}).get(key):raise ValueError('V5.1 recovery method mismatch '+key)
            c.create_endpoint(alpha)
            c.endpoints[alpha]['algo'].policy.load_state_dict(saved['policy'])
            c.endpoints[alpha]['update']=100
        c.out=parent;c.status=original_status
        from .fixed_command_panel import load_protocol,schedules
        expected=[name for name,_ in schedules(load_protocol())]
        missing=[name for name in expected if any(not (parent/'evaluation100'/name/f'{method}.npz').exists() for method in ('alpha0','alpha1','B0'))]
        if missing:
            result=c.evaluate_preference(100,case_filter=missing)
        else:
            from .preference_command_reporting import report_stage2
            result=report_stage2(parent,100)
        audits=[payload[f'alpha{endpoint}']['reward_audit'] for payload in result.values() for endpoint in (0,1)]
        failed=[audit for audit in audits if audit.get('passed') is not True]
        if failed:raise RuntimeError(f'V5.1 independent reward reconstruction failed at update100: {failed}')
        recovery=dict(parent_run=str(parent),runtime=str(runtime),missing_cases_completed=missing,training_restarted=False,
                      checkpoints_written=False,original_error_receipt=str(receipt),completed_epoch=time.time())
        write(parent/'evaluation100_recovery.json',recovery)
        c._status(state='complete',stage='saved_and_stopped',training_stage=51,qualified=None,
                  reason=None,case=None,evaluation_recovered_after_tick_budget_fix=True,
                  recovery_receipt=str(parent/'evaluation100_recovery.json'))
        return result
    except BaseException as exc:
        c.out=parent;c.status=original_status
        c._status(state='error',reason=str(exc),training_stage=51,evaluation_recovery_failed=True)
        raise
    finally:
        for ep in c.endpoints.values():ep['writer'].close()
