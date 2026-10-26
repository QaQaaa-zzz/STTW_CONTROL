"""Fixed finite-goal Pure Pursuit comparison, not a new safety controller."""
import copy
import jax
import jax.numpy as j
import numpy as np

GOAL_ROLES = dict(goal_zero_feedback=('zero_middle_feedback',[False,False]),
                  goal_steer_feedback=('steer_only_feedback',[False,True]),
                  goal_full_feedback=('feedback_010',[True,True]))


def goal_follower(original,path,progress,pose,config,*,radius_m=.5):
    """Use the fixed goal only when ordinary lookahead passes it, still ahead."""
    from .geometric_path import at,wrap
    if radius_m != .5:raise ValueError('fixed original task radius required')
    c=config['follower'];point,_,_=at(path,path.goal);_,heading,_=at(path,progress)
    d=point-pose[:2];cs=j.cos(pose[2]);sn=j.sin(pose[2])
    body=j.stack([cs*d[0]+sn*d[1],-sn*d[0]+cs*d[1]])
    use=(progress<path.goal)&(progress+original.lookahead>path.goal)&(body[0]>c['preview_body_x_min_m'])&(j.sum(body**2)>radius_m**2)
    k=2*body[1]/j.maximum(j.sum(body**2),1e-12)
    steer=j.arctan(c['wheelbase_m_expected']*k/j.cos(j.deg2rad(c['caster_deg_expected'])))
    command=original.command.at[1].set(j.clip(steer,-c['steer_reference_max_rad'],c['steer_reference_max_rad']))
    fault=original.numerical_fault|~j.all(j.isfinite(j.concatenate([body,command])))
    invalid=(body[0]<=c['preview_body_x_min_m'])|(j.abs(wrap(heading-pose[2]))>=j.pi/2)|fault
    proposed=original.replace(command=command,preview_body=body,preview_world=point,kappa=k,
                               invalid=invalid,numerical_fault=fault)
    return jax.lax.cond(use,lambda _:proposed,lambda _:original,None),use


class GoalPreviewEnv:
    """One unchanged physical tick; replace only newly published follower state."""
    def __init__(self,base):self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def config(self):return self.base.config
    @config.setter
    def config(self,value):self.base.config=value
    def __copy__(self):
        # make_reference_step copies the environment and installs floor_config.
        # Both delegated bound methods and tick must see that same local config.
        return GoalPreviewEnv(copy.copy(self.base))

    def tick(self,state,action,path,speed,force_pulse=None,horizon=None):
        from .path_reference import publish
        end,logs=self.base.tick(state,action,path,speed,force_pulse,horizon=horizon)
        def update(_):
            _,pose,_=self.physics.observe(end.physical.data)
            follower,used=goal_follower(end.follower,path,end.projection.progress,pose,self.config)
            def replace(_):
                # Publish from INPUT raw, never from end.raw (no double slew).
                nominal,rates=publish(state.physical.raw,follower.command,self.config)
                domain=follower.invalid&~follower.numerical_fault
                revised=end.replace(follower=follower,physical=end.physical.replace(raw=nominal),
                    nominal_rates=rates,fault=end.fault|follower.numerical_fault,
                    domain_exit=state.domain_exit|domain)
                # These fields describe the newly computed next follower;
                # executed nominal/pp_target/preview logs remain unchanged.
                return revised,dict(logs,domain_exit=domain,tracking_domain_failure=domain)
            return jax.lax.cond(used,replace,lambda _:(end,logs),None)
        return jax.lax.cond(end.tick%4==0,update,lambda _:(end,logs),None)


def goal_cases(declared):
    originals=[c for c in declared if c['role']=='feedback_010']
    if len(originals)!=60 or len({c['case_id'] for c in originals})!=60:
        raise ValueError('complete sixty original events required')
    cases=[]
    for c in originals:
        if c['feedback_gain_rad']!=.1 or c['mode_index']!=1 or not c['causal_support']:
            raise ValueError('fixed feedback and original support required')
        for role,(source,mask) in GOAL_ROLES.items():
            new=copy.deepcopy(c)
            new.pop('actor_source_zero',None)
            new.update(role=role,goal_preview=True,source_role=source,source_keep_channels=mask,
                       case_id=c['case_id'].rsplit('/',1)[0]+'/'+role)
            cases.append(new)
    return cases


def make_goal_step(env,actor,*,horizon_s,spec):
    from .middle_channel_ablation import ChannelActor
    from .reference_task_feedback import make_feedback_evaluation_step
    from .reference_reentry import actor_permission
    advance=make_feedback_evaluation_step(env,ChannelActor(actor),horizon_s=horizon_s,spec=spec)
    def step(carry,parameters,path,speed,pulse,mode):
        s=carry.state
        stopped=s.physical.failed|s.fault|(s.tick>=round(horizon_s/env.cc.dt))
        enabled=actor_permission(s.domain_exit,s.follower.invalid,carry.clean_blocks,j.bool_(False),stopped)
        obs,_,_=env.observation(s,path,speed,horizon_s)
        native=actor.apply(parameters['actor']['native'],j.where(enabled,obs,j.zeros_like(obs)))
        end,logs=advance(carry,parameters,path,speed,pulse,mode)
        legal=logs['actor_action_valid'];active=logs['active_tick']
        logs.update(source_actor_latent=logs['original_actor_latent'],
            original_actor_latent=j.where(legal[:,None],j.tile(native,(4,1)),j.nan),
            source_keep_channels=j.tile(parameters['actor']['keep_channels'],(4,1)),
            goal_preview_world=j.where(active[:,None],j.tile(s.follower.preview_world,(4,1)),j.nan),
            goal_preview_body=j.where(active[:,None],j.tile(s.follower.preview_body,(4,1)),j.nan),
            goal_preview_command=j.where(active[:,None],j.tile(s.follower.command,(4,1)),j.nan))
        return end,logs
    return step


def verify_goal_log(raw,case,config,spec,path):
    """NumPy replay of block-held target selection and single publish slew."""
    from .reference_task_feedback import verify_feedback_log
    legal=np.asarray(raw['actor_action_valid'],bool);active=np.asarray(raw['active_tick'],bool)
    mask=np.asarray(GOAL_ROLES[case['role']][1],bool)
    np.testing.assert_array_equal(raw['source_keep_channels'],np.tile(mask,(len(active),1)))
    np.testing.assert_array_equal(raw['source_actor_latent'][legal],np.where(mask,raw['original_actor_latent'][legal],0.))
    verify_feedback_log(dict(raw,original_actor_latent=raw['source_actor_latent']),case,config,spec)
    c=config['follower'];dt=config['timing']['lower_dt_s'];goal=float(path['goal'])
    point=np.array([np.interp(goal,path['s'],path['xy'][:,i]) for i in range(2)])
    selected=[];starts=np.flatnonzero(active&(raw['physical_tick']%4==0))
    # The first block retains the unchanged sealed task0 follower. Subsequent
    # blocks use the previous final physics sample, not a post-tick approximation.
    if starts.size and np.any(raw['path_progress'][:4]+c['lookahead_max_m']>=goal):
        raise ValueError('first task0 block must precede finite-goal intervention')
    for start in starts[starts>0]:
        pre=start-1;progress=raw['path_progress'][pre];speed=raw['actual_forward_speed'][pre]
        look=np.clip(c['lookahead_time_s']*max(speed,c['lookahead_min_speed_m_s']),c['lookahead_min_m'],c['lookahead_max_m'])
        xy=raw['actual_xy'][pre];yaw=raw['actual_yaw'][pre]
        rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        goal_body=rotation@(point-xy)
        used=progress<goal and progress+look>goal and goal_body[0]>c['preview_body_x_min_m'] and np.dot(goal_body,goal_body)>.25
        chosen=point if used else np.array([np.interp(progress+look,path['s'],path['xy'][:,i]) for i in range(2)])
        body=rotation@(chosen-xy);k=2*body[1]/max(np.dot(body,body),1e-12)
        steer=np.clip(np.arctan(c['wheelbase_m_expected']*k/np.cos(np.deg2rad(c['caster_deg_expected']))),-c['steer_reference_max_rad'],c['steer_reference_max_rad'])
        block=np.arange(start,min(start+4,len(active)));block=block[active[block]]
        np.testing.assert_allclose(raw['goal_preview_world'][block],np.tile(chosen,(len(block),1)),atol=3e-5,rtol=0)
        np.testing.assert_allclose(raw['goal_preview_body'][block],np.tile(body,(len(block),1)),atol=3e-5,rtol=0)
        np.testing.assert_allclose(raw['goal_preview_command'][block],np.tile([case['requested_speed_m_s'],steer],(len(block),1)),atol=3e-5,rtol=0)
        if used:selected.extend(block.tolist())
    limit=np.array([c['nominal_speed_slew_m_s2'],c['nominal_steer_slew_rad_s']])*dt
    indices=np.flatnonzero(active)[1:]
    target=raw['goal_preview_command'][indices]
    prior=raw['nominal_reference'][indices-1]
    expected=prior+np.clip(target-prior,-limit,limit)
    np.testing.assert_allclose(raw['nominal_reference'][indices],expected,atol=3e-6,rtol=0)
    return dict(changed_target_ticks=len(selected),changed_target_s=len(selected)*dt,
                first_changed_target_s=(selected[0]*dt if selected else None),
                geometry_checked_after_initial_block=True)


def prepare(root,source):
    import ast
    import json
    import os
    import shutil
    import subprocess
    import traceback
    from pathlib import Path
    from .preference_best import atomic_json,digest
    from .finite_value_evaluation_report import evaluation_inputs
    from .finite_value_evaluation_run import authenticate_models
    from prepare_short_recovery_campaign import register
    root,source=Path(root).resolve(),Path(source).resolve();root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 有限目标预瞄强对照准备',root/'prepare_status.json')
    try:
        old=json.loads((source/'plan.json').read_text())
        if old['experiment']!='middle_reference_channel_feedback_v1':raise ValueError('original channel panel required')
        for file in ['status.json','analysis/status.json']:
            if json.loads((source/file).read_text())['phase']!='completed':raise ValueError('incomplete source')
        if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
            raise ValueError('changed source plan')
        for file,expected in old['sealed_files'].items():
            if digest(file)!=expected:raise ValueError('changed sealed source: '+file)
        module=Path(__file__).parent
        runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
            'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
            'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
            'reference_reentry.py','geometric_path.py','middle_channel_ablation.py']
        for name in runtime:
            if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
                raise ValueError('changed original numerical runtime: '+name)
        def body(file,symbol):
            return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
                        if isinstance(n,ast.FunctionDef) and n.name==symbol)
        for symbol in ['make_evaluation_step','rebind_path_context']:
            if body(module/'finite_value_evaluation.py',symbol)!=body(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py',symbol):
                raise ValueError('changed common execution/initialization kernel')
        cfg=json.loads((source/'common_environment_config.json').read_text())
        native=copy.deepcopy(old['models']['steer_only_feedback']);native.pop('source_keep_channels')
        authenticate_models(cfg,dict(free_GAE=native))
        declared,records,indices,_=evaluation_inputs(source,old);cases=goal_cases(declared)
        originals={v[0] for v in GOAL_ROLES.values()}
        cached=[c for c in declared if c['role'] in originals];ids={c['case_id'] for c in cached}
        if len(cached)!=180 or len(ids)!=180:raise ValueError('180 complete matched original controls required')
        byid={r['case_id']:r for r in records}
        for c in cached:
            if any(byid[c['case_id']][k]!=v for k,v in c.items()):raise ValueError('cache metadata mismatch')
        cached_records=[r for r in records if r['case_id'] in ids]
        cached_indices=[copy.deepcopy(r) for r in indices if r['case_id'] in ids]
        for item in cached_indices:
            file=(source/item['file']).resolve()
            if digest(file)!=item['sha256']:raise ValueError('cache raw changed')
            item['file']=str(file)
        for c in old['contexts'].values():
            with np.load(c['path'],allow_pickle=False) as z:
                if float(z['goal'])<=cfg['follower']['lookahead_max_m']:
                    raise ValueError('goal modification must be inactive in unchanged task0 block')
        for name in ['common_environment_config.json','source_declaration.json']:shutil.copyfile(source/name,root/name)
        models={role:dict(copy.deepcopy(native),source_keep_channels=mask) for role,(_,mask) in GOAL_ROLES.items()}
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        cache=dict(declared=cached,records=cached_records,indices=cached_indices,
            active_lower_ticks=sum(r['physical_ticks'] for r in cached_records),allocated_lower_slots=360000,
            source_run=str(source),source_plan_sha256=digest(source/'plan.json'))
        atomic_json(root/'cache_receipt.json',cache)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_goal_preview.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json','cache_receipt.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
        plan=dict(project='STTW_CONTROL',experiment='finite_goal_preview_baseline_v1',models=models,contexts=old['contexts'],
            evaluation_roles=['zero_middle_feedback','steer_only_feedback','feedback_010',*GOAL_ROLES],
            goal_preview_comparison=True,declared_cases=180,new_cases=180,cached_cases=180,analysis_cases=360,
            feedback_spec=old['feedback_spec'],horizon_s=10.,lower_dt_s=.005,optimizer_updates=0,
            goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=old['task_rule'],
            maximum_active_lower_ticks=360000,maximum_allocated_lower_slots=360000,
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=360000,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=old['initialization'],information=old['information'],
            stop='Fixed180 once, engineering error stops; no goal-radius/lookahead/gain/threshold/mask search, training or automatic retry.',
            data_role='Task-interface DEV diagnosis, not a novel controller or independent final test. Same180 historical controls, not new repeats. '
                'Goal point original and fixed; when pre-goal lookahead crosses it and it is ahead outside0.5m disk, select it; otherwise original PP. '
                'No speed change/stop, state clearing, history reset or horizon extension. Same rule for all three neural sources. '
                'Ordinary goal handling gains do not establish Transactions innovation; no gain only rejects this fixed rule, not all geometric recovery.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=180,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 有限目标预瞄180及强对照',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 有限目标预瞄180 shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
