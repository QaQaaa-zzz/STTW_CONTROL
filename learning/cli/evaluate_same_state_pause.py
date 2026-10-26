"""One fixed common-state post-pulse opportunity diagnostic; no optimization."""
import argparse
import ast
import copy
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from sttw_control.preference_best import atomic_json,digest
from sttw_control.same_state_reference_pause import inventory,clone_complete_carry,prefix_guard,pause_step


def prepare(root,source):
    from prepare_short_recovery_campaign import register
    from sttw_control.finite_value_evaluation_run import authenticate_models
    root=Path(root).resolve();source=Path(source).resolve();old=json.loads((source/'plan.json').read_text())
    if old['experiment']!='entry_history_confirmation_v1' or json.loads((source/'status.json').read_text())['phase']!='completed':
        raise ValueError('complete original entry-history source required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('source plan changed')
    for file,sha in old['sealed_files'].items():
        if digest(file)!=sha:raise ValueError('source input changed')
    module=Path('learning/src/sttw_control').resolve()
    for filename in ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py','path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py','direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py','reference_reentry.py','geometric_path.py','middle_channel_ablation.py','causal_reference_baseline.py']:
        if digest(module/filename)!=old['source_sha256'][str(module/filename)]:raise ValueError('source numeric runtime changed: '+filename)
    for filename,names in [('finite_value_evaluation.py',['make_evaluation_step','rebind_path_context']),('goal_preview_baseline.py',['goal_follower','GoalPreviewEnv','make_goal_step'])]:
        def tree(file,name):return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        for name in names:
            if tree(module/filename,name)!=tree(source/'source_at_launch/learning/src/sttw_control'/filename,name):raise ValueError('numeric control function changed')
    prefixes,cases=inventory(json.loads((source/'inventory.json').read_text())['cases'])
    cfg=json.loads((source/'common_environment_config.json').read_text());model=copy.deepcopy(old['models']['goal_full_feedback']);model.pop('source_keep_channels')
    authenticate_models(cfg,dict(free_GAE=model))
    # All source timing is aligned to the existing .02s policy clock.
    forks=np.array([c['fork_task_tick'] for c in prefixes]);prefix_blocks=int(np.ceil(forks.max()/200));suffix_blocks=int(np.ceil((2000-forks.min())/200))
    slots=prefix_blocks*200*120+suffix_blocks*200*360
    active_bound=int(np.sum(forks+3*(2000-forks)))
    root.mkdir(exist_ok=False)
    register(root/'ACTIVE_PREPARE.json','STTW同状态短暂停准备',root/'prepare_status.json')
    for name in ['common_environment_config.json','source_declaration.json']:shutil.copyfile(source/name,root/name)
    protocol=dict(question='From exactly the same complete current post-pulse state, does a fixed short pause of neural reference channels offer useful recovery opportunity?',
        roles=['full_current','pause_neural_steer','pause_neural_both'],pause_lower_ticks=80,pause_s=.4,
        fork='original declared pulse end=onset+20lower ticks; same synthetic clock for nominal',horizon_s=10.,goal_radius_m=.5,
        data_role='Outcome-inspected related DEV opportunity diagnostic, not independent confirmation, deployment detector or new method.',
        information='Privileged experiment timing only. No policy alpha/state/history/filter/ESO/actuator/gate reset; restore control law, not original action sequence.',
        stop='All120 events/three fixed branches once. No duration/cutpoint/mask expansion, optimization, witness-selected cases or automatic retry.',
        outcomes='Fixed-branch arrival/survival gain and loss, same-user cost; nominal and reached-fork eligibility separate. Posthoc branch union is opportunity only.',
        failure_semantics='Cases terminating before fork remain pre-fork terminal; no intervention executed, no replacement. Fault stops engineering run, no negative physical relabel.',
        cost_formula='Sum actual120 shared prefixes once + actual360 suffixes; logical complete trajectory reuse never recounts prefixes as physics. All fixed scan slots counted separately.')
    for name,value in [('protocol.json',protocol),('prefix_inventory.json',dict(cases=prefixes)),('inventory.json',dict(cases=cases)),('authenticated_model.json',model)]:atomic_json(root/name,value)
    files=list(module.glob('*.py'))+[Path(__file__).resolve()]
    sources={str(f):digest(f) for f in files}
    for file in files:
        dest=root/'source_at_launch'/file.relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    sealed={str(root/name):digest(root/name) for name in ['common_environment_config.json','source_declaration.json','protocol.json','prefix_inventory.json','inventory.json','authenticated_model.json']}
    sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
    plan=dict(project='STTW_CONTROL',experiment='same_state_reference_pause_v1',source_reference_run=str(source),source_plan_sha256=digest(source/'plan.json'),
        contexts=old['contexts'],model=model,feedback_spec=old['feedback_spec'],logical_complete_trajectories=360,prefixes=120,branches_per_prefix=3,
        prefix_blocks=prefix_blocks,suffix_blocks=suffix_blocks,maximum_active_lower_ticks=active_bound,maximum_allocated_lower_slots=slots,
        sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_updates=0)
    atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,completed_cases=0,optimizer_updates=0))
    atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),new_physics_steps=0,optimizer_updates=0))
    for name,title,status in [('ACTIVE_RUN.json','STTW同状态120前缀360尾段',root/'status.json'),('ACTIVE_SHAPE.json','STTW同状态分支shape',root/'shape/status.json')]:register(root/name,title,status)
    print(json.dumps(dict(prepared=str(root),prefixes=120,suffixes=360,actual_bound=active_bound,slots_bound=slots,new_physics_steps=0)))


def execute(root,shape_only=False):
    root=Path(root).resolve();previous=json.loads((root/'status.json').read_text())
    if previous['phase']!='prepared':raise ValueError('fresh prepared run required; no implicit replay/restart')
    out=root/('shape' if shape_only else 'physical');out.mkdir(exist_ok=False)
    status_file=out/'status.json' if shape_only else root/'status.json'
    status=dict(previous,phase='running',pid=os.getpid(),stage='authenticating')
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(status_file,status)
    update()
    try:
        import jax
        import jax.numpy as j
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.goal_preview_baseline import GoalPreviewEnv,make_goal_step
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.causal_recovery_relay import CausalRelayState,initial_gate as initial_relay
        from sttw_control.causal_reference_baseline import initial_gate
        from sttw_control.geometric_path import Path as GeometricPath
        from sttw_control.finite_value_evaluation import rebind_path_context
        from sttw_control.finite_value_evaluation_run import authenticate_models
        from sttw_control.evaluation_clock import physical_origin,physical_pulse
        from sttw_control.anticipatory_run import flatten_ticks,physical_outcome
        from sttw_control.runtime import configure_compilation_cache
        p=json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('run plan changed')
        for file,sha in {**p['sealed_files'],**p['source_sha256']}.items():
            if digest(file)!=sha:raise ValueError('sealed input changed: '+file)
        cfg=json.loads((root/'common_environment_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        authenticate_models(cfg,dict(free_GAE=p['model']))
        with Path(p['model']['actor']).open('rb') as f:native=jax.tree.map(j.asarray,pickle.load(f))
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2');env=GoalPreviewEnv(env)
        prefixes=json.loads((root/'prefix_inventory.json').read_text())['cases'];cases=json.loads((root/'inventory.json').read_text())['cases']
        origins={};paths={}
        for context,c in p['contexts'].items():
            with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
            original=jax.tree.map(lambda x:j.asarray(x[0]),bank)
            if int(original.tick)!=0 or int(original.physical.physical_tick)!=physical_origin(c) or bool(original.physical.failed|original.fault|original.domain_exit):raise ValueError('invalid original complete entry state')
            with np.load(c['path']) as z:path=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
            paths[context]=path
            rebound=rebind_path_context(original,path,j.float32(c['speed_m_s']),env)
            retained=[n for n in original.__dataclass_fields__ if n not in ['projection','follower']]
            if not all(getattr(original,n) is getattr(rebound,n) for n in retained):raise ValueError('unexpected initial state field mutation')
            origins[context]=rebound
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        states=[origins[c['context']].replace(alpha=j.float32(c['alpha'])) for c in prefixes]
        carry=CausalRelayState(stack(states),stack([initial_gate() for _ in prefixes]),j.zeros(120,j.int32),stack([initial_relay() for _ in prefixes]))
        batch_paths=stack([paths[c['context']] for c in prefixes]);speeds=j.asarray([c['requested_speed_m_s'] for c in prefixes]);modes=j.asarray([c['mode_index'] for c in prefixes],j.int32)
        pulses=j.asarray([physical_pulse(c,p['contexts'][c['context']]) for c in prefixes],j.float32);forks=j.asarray([c['fork_task_tick'] for c in prefixes],j.int32)
        params=dict(actor=dict(native=native,keep_channels=j.array([True,True])),feedback_gain_rad=j.float32(.1))
        parameters=stack([params for _ in prefixes]);env.set_log_template(states[0],paths[prefixes[0]['context']],speeds[0])
        advance=make_goal_step(env,make_path_actor(cfg),horizon_s=10.,spec=p['feedback_spec'])
        first=lambda x:jax.tree.map(lambda a:a[0],x)
        log_shapes=jax.eval_shape(advance,first(carry),params,first(batch_paths),speeds[0],pulses[0],modes[0])[1]
        zeros=jax.tree.map(lambda s:j.zeros(s.shape,s.dtype),log_shapes)
        prefix_one=prefix_guard(advance,zeros);suffix_one=pause_step(advance)
        @jax.jit
        def prefix_chunk(state):
            return jax.lax.scan(lambda st,_:jax.vmap(prefix_one)(st,parameters,batch_paths,speeds,pulses,modes,forks),state,None,length=50)
        repeated=lambda x:jax.tree.map(lambda a:j.repeat(a,3,axis=0),x)
        branch_paths=repeated(batch_paths);branch_speeds=j.repeat(speeds,3);branch_pulses=j.repeat(pulses,3,axis=0);branch_modes=j.repeat(modes,3)
        suffix_parameters=stack([dict(params,fork_task_tick=j.int32(c['fork_task_tick']),pause_mask=j.asarray(c['pause_mask'],bool)) for c in cases])
        @jax.jit
        def suffix_chunk(state):
            return jax.lax.scan(lambda st,_:jax.vmap(suffix_one)(st,suffix_parameters,branch_paths,branch_speeds,branch_pulses,branch_modes),state,None,length=50)
        if shape_only:
            a=jax.eval_shape(prefix_chunk,carry);b=jax.eval_shape(suffix_chunk,clone_complete_carry(carry))
            assert a[1]['active_tick'].shape==(50,120,4) and b[1]['active_tick'].shape==(50,360,4)
            atomic_json(out/'receipt.json',dict(prefix_lower_shape=[50,120,4],suffix_lower_shape=[50,360,4],new_physics_steps=0,optimizer_updates=0));update(phase='completed',stage='shape_complete');return
        if json.loads((root/'shape/status.json').read_text())['phase']!='completed' or not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('completed shape and actual GPU required')
        prefix_logs=[];suffix_logs=[];start=time.monotonic()
        def account(data,n):
            status['active_lower_ticks']+=int(data['active_tick'].sum());status['allocated_lower_slots']+=200*n
            if status['active_lower_ticks']>p['maximum_active_lower_ticks'] or status['allocated_lower_slots']>p['maximum_allocated_lower_slots']:raise ValueError('declared cost exceeded')
        for block in range(p['prefix_blocks']):
            carry,logs=prefix_chunk(carry);jax.block_until_ready(carry);host,data=jax.device_get((carry,logs));data=flatten_ticks(data);prefix_logs.append(data)
            np.savez_compressed(out/f'prefix_chunk_{block:02d}.npz',**data);account(data,120);update(stage='prefix',completed_prefix_blocks=block+1,elapsed_s=time.monotonic()-start)
            if np.any(host.state.fault):raise RuntimeError('prefix engineering fault retained; no automatic retry')
        host=jax.device_get(carry);reached=(np.asarray(host.state.tick)==np.asarray(forks))&~np.asarray(host.state.physical.failed)
        if not np.all(reached|np.asarray(host.state.physical.failed)):raise ValueError('prefix did not reach declared fork or real contact')
        with (out/'common_fork_carry.pkl').open('wb') as f:pickle.dump(host,f)
        branch=clone_complete_carry(carry);cloned=jax.device_get(branch)
        for a,b in zip(jax.tree.leaves(host),jax.tree.leaves(cloned)):np.testing.assert_array_equal(b,np.repeat(a,3,axis=0))
        atomic_json(out/'fork_receipt.json',dict(prefixes=120,branches=360,reached_fork=reached.tolist(),fork_ticks=np.asarray(forks).tolist(),actual_state_ticks=np.asarray(host.state.tick).tolist(),pre_fork_contacts=np.asarray(host.state.physical.failed).tolist(),all_carry_leaves_copied=True,clock_reset=False))
        prefix_impulses=[]
        for lane,c in enumerate(prefixes):
            impulse=sum((d['applied_force_impulse'][:,lane][d['active_tick'][:,lane]].sum(axis=0,dtype=np.float64) for d in prefix_logs),start=np.zeros(3))
            expected=np.asarray(c['world_direction'])*c['force_N']*c['pulse_duration_s']
            prefix_impulses.append(dict(case_id=c['case_id'],eligible=bool(reached[lane]),measured_prefix_impulse=impulse.tolist(),expected_complete_pulse_impulse=expected.tolist(),full_exposure_with_original_numeric_tolerance=bool(np.allclose(impulse,expected,atol=.025,rtol=.05))))
        atomic_json(out/'scheduled_pulse_end_audit.json',dict(scope='Declared cutpoint never moved using realized data. Prefix impulse is measured, including pre-fork contact/censoring. Full exposure uses the original evaluation tolerance.',cases=prefix_impulses))
        prefix_actual=status['active_lower_ticks'];prefix_slots=status['allocated_lower_slots']
        for block in range(p['suffix_blocks']):
            branch,logs=suffix_chunk(branch);jax.block_until_ready(branch);end,data=jax.device_get((branch,logs));data=flatten_ticks(data);suffix_logs.append(data)
            np.savez_compressed(out/f'suffix_chunk_{block:02d}.npz',**data);account(data,360);update(stage='suffix',completed_suffix_blocks=block+1,elapsed_s=time.monotonic()-start)
            if np.any(end.state.fault):raise RuntimeError('suffix engineering fault retained; no automatic retry')
        with (out/'final_branch_carry.pkl').open('wb') as f:pickle.dump(end,f)
        pref={k:np.concatenate([v[k] for v in prefix_logs]) for k in prefix_logs[0]};suff={k:np.concatenate([v[k] for v in suffix_logs]) for k in suffix_logs[0]}
        (out/'cases').mkdir();records=[];index=[]
        for lane,c in enumerate(cases):
            source=c['prefix_index'];pa=pref['active_tick'][:,source];sa=suff['active_tick'][:,lane]
            raw={k:np.concatenate([pref[k][:,source][pa],suff[k][:,lane][sa]]) for k in pref}
            for k in set(suff)-set(pref):
                before=np.full(int(pa.sum()),c['fork_task_tick'],dtype=suff[k].dtype) if k=='branch_fork_task_tick' else np.zeros(int(pa.sum()),dtype=suff[k].dtype)
                raw[k]=np.concatenate([before,suff[k][:,lane][sa]])
            ticks=len(raw['active_tick']);np.testing.assert_array_equal(raw['physical_tick'],np.arange(ticks))
            outcome=physical_outcome(raw['active_tick'],raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            if outcome=='unknown':raise ValueError('unknown branch preserved; no negative relabel')
            file=out/'cases'/f'case_{lane:03d}.npz';np.savez_compressed(file,**raw)
            records.append(dict(c,physical_outcome=outcome,physical_ticks=ticks,reached_fork=bool(reached[source]),prefix_physical_ticks=int(pa.sum()),suffix_physical_ticks=int(sa.sum()),pause_requested_active_lower_ticks=int(np.sum(raw['active_tick']&raw['branch_pause_active'])),pause_legal_active_lower_ticks=int(np.sum(raw['active_tick']&raw['branch_pause_active']&raw['actor_action_valid']))))
            pose=np.asarray(env.physics.helpers.pose(states[source].physical.data))
            index.append(dict(case_id=c['case_id'],file=str(file.relative_to(root)),sha256=digest(file),initial_xy=pose[:2].tolist(),initial_yaw=float(pose[2])))
        atomic_json(out/'results.json',records);atomic_json(out/'case_index.json',index)
        atomic_json(out/'cost_receipt.json',dict(prefix_actual_lower_ticks=prefix_actual,prefix_allocated_slots=prefix_slots,suffix_actual_lower_ticks=status['active_lower_ticks']-prefix_actual,suffix_allocated_slots=status['allocated_lower_slots']-prefix_slots,logical_complete_trajectory_ticks=sum(c['physical_ticks'] for c in records),new_physics_ticks=status['active_lower_ticks'],warning='Logical complete traces repeat the same prefix in each branch; do not count that repeated data as new physics.'))
        update(phase='completed',stage='physical_complete_analysis_pending',completed_cases=360,elapsed_s=time.monotonic()-start)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--phase',choices=['prepare','shape','execute'],required=True);parser.add_argument('--source',type=Path,default=Path('runs/entry_history_confirmation_v1'))
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.run,args.source)
    else:execute(args.run,shape_only=args.phase=='shape')
