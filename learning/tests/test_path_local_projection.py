"""Compare local geometry with the frozen full-segment definition."""
import json
from pathlib import Path as DiskPath
import jax
import jax.numpy as j
import numpy as np
from sttw_control import geometric_path as gp
CFG=json.loads((DiskPath(__file__).parents[1]/'configs/STTW_Path_Conflict_Exposure.json').read_text())

def reference(path,xy,previous,distance):
    c=CFG['path'];lo=j.maximum(0.,previous-c['projection_backward_m'])
    hi=j.minimum(path.s[-1],previous+j.maximum(c['projection_forward_floor_m'],c['projection_forward_motion_multiplier']*distance+c['projection_forward_padding_m']))
    a=path.xy[:-1];v=path.xy[1:]-a;span=path.s[1:]-path.s[:-1]
    u=j.sum((xy-a)*v,axis=-1)/j.sum(v*v,axis=-1)
    u=j.clip(u,j.clip((lo-path.s[:-1])/span,0,1),j.clip((hi-path.s[:-1])/span,0,1))
    ss=path.s[:-1]+u*span;points=a+u[:,None]*v
    d2=j.where((path.s[1:]>=lo)&(path.s[:-1]<=hi),j.sum((xy-points)**2,axis=-1),j.inf)
    idx=j.argmin(j.where(d2<=j.min(d2)+1e-10,j.abs(ss-previous),j.inf))
    h=j.interp(ss[idx],path.s,path.heading);delta=xy-points[idx]
    return gp.Projection(j.maximum(previous,ss[idx]),ss[idx],points[idx],-j.sin(h)*delta[0]+j.cos(h)*delta[1],h)


def test_uniform_interpolation_matches_original_including_endpoints():
    for name in ['straight','left90_R2','compact_RL']:
        path=gp.build_path(name,[.2,-.3,.5],CFG)
        values=j.concatenate([j.array([-10.,0.,.02,1000.]),path.s[::57],path.s[::57]+1.e-6])
        expected=jax.vmap(lambda s:(j.stack([j.interp(s,path.s,path.xy[:,0]),j.interp(s,path.s,path.xy[:,1])]),j.interp(s,path.s,path.heading),j.interp(s,path.s,path.curvature)))(values)
        actual=jax.jit(jax.vmap(lambda s:gp.at(path,s)))(values)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_allclose(a,b,atol=2e-6,rtol=1e-6)
        # Uniform-grid interpolation requires no search loop.
        assert 'scan[' not in str(jax.make_jaxpr(lambda s:gp.at(path,s))(j.array(1.)))


def test_local_projection_matches_full_for_short_and_arbitrarily_wide_windows():
    for name in ['straight','left90_R2','compact_LR','right180_R1p8']:
        path=gp.build_path(name,[.1,.2,-.4],CFG)
        rng=np.random.default_rng(719)
        previous=j.asarray(np.r_[0.,.02,1.28,float(path.s[-1])-.01,np.linspace(0,float(path.s[-1])-.1,70)])
        points=jax.vmap(lambda s:gp.at(path,s)[0])(previous)+j.asarray(rng.normal(0,.3,(len(previous),2)))
        distances=j.asarray(rng.choice([0.,.01,.04,.8,3.,25.],len(previous)))
        expected=jax.jit(jax.vmap(lambda xy,p,d:reference(path,xy,p,d)))(points,previous,distances)
        actual=jax.jit(jax.vmap(lambda xy,p,d:gp.project(path,xy,p,d,CFG)))(points,previous,distances)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_allclose(a,b,atol=3e-6,rtol=2e-6)
        text=str(jax.make_jaxpr(lambda xy:gp.project(path,xy,j.array(1.),j.array(.01),CFG))(j.zeros(2)))
        assert f'f32[{len(path.s)-1},2]' not in text


@__import__('pytest').mark.skipif(__import__('os').environ.get('STTW_PATH_CLEANUP_GPU')!='1',reason='bounded explicit GPU check')
def test_gpu_cleanup_matches_original_training_batch(tmp_path):
    import sys,types,subprocess,time,os
    from sttw_control import path_command_training as tr
    from sttw_control import path_command_statistics as statistics
    from sttw_control.runtime import configure_compilation_cache
    configure_compilation_cache()
    root=DiskPath(__file__).parents[2]
    def baseline_module(name,file):
        module=types.ModuleType('sttw_control.'+name);module.__package__='sttw_control'
        sys.modules[module.__name__]=module
        text=subprocess.check_output(['git','show','83f0873:learning/src/sttw_control/'+file],cwd=root,text=True)
        exec(compile(text,file,'exec'),module.__dict__)
        return module
    old_geometry=baseline_module('_cleanup_old_geometry','geometric_path.py')
    old_reference=baseline_module('_cleanup_old_reference','path_reference.py');old_reference.at=old_geometry.at
    old_env_module=baseline_module('_cleanup_old_environment','path_command_env.py')
    old_env_module.at=old_geometry.at;old_env_module.project=old_geometry.project;old_env_module.pursuit=old_reference.pursuit
    plan,cfg,source=tr.load_plan(root/'learning/configs/path_conflict_exposure_run.json')
    env,snapshot=tr.load_environment(cfg,source,'terminal_contract_v2')
    original=old_env_module.PathCommandEnv(cfg,env.plant_spec,env.lower,'terminal_contract_v2')
    batches=[tr.PathBatch(e,snapshot,cfg,plan['seed'],None) for e in (original,env)]
    ids=j.array([0,17,102,103,306,307,410,411]);states=[];executors=[];compilation=[]
    for variant,(e,batch) in enumerate(zip((original,env),batches)):
        ep=jax.vmap(lambda eid:batch.reset(eid,j.int32(0),j.int32(81)))(ids)
        first=jax.tree.map(lambda x:x[0],ep);e.set_log_template(first.state,first.route.path,first.route.speed)
        states.append(ep)
        def advance(ep,z,batch=batch,variant=variant):
            if variant==0:return jax.vmap(lambda s,a:batch.advance_one(s,a,j.int32(81)))(ep,z)
            ended,_,reward,flags,final,stats=jax.vmap(lambda s,a:batch.advance_one(s,a,j.int32(81),auto_reset=False))(ep,z)
            nxt=tr.reset_finished_batch(ended,flags,lambda eid,k:batch.reset(eid,k,j.int32(81)))
            resets=jax.vmap(statistics.reset_statistics)(nxt.route,flags['done']&~flags['engineering_fault'])
            stats['diagnostics']=statistics.merge_statistics(stats['diagnostics'],resets)
            return nxt,jax.vmap(batch.observation)(nxt),reward,flags,final,stats
        started=time.perf_counter();execute=jax.jit(advance).lower(ep,j.zeros((8,2))).compile()
        compilation.append(time.perf_counter()-started);executors.append(execute)
    error=0.;reward_error=0.;timings=[[],[]];leaf_errors={}
    for step in range(4):
        outputs=[]
        for variant,execute in enumerate(executors):
            # Exercise a real full closed-loop reset on the last call as well.
            if step==3:states[variant]=states[variant].replace(state=states[variant].state.replace(domain_exit=states[variant].state.domain_exit.at[0].set(True)))
            started=time.perf_counter();result=execute(states[variant],j.ones((8,2))*j.array([.1,-.2]));jax.block_until_ready(result)
            timings[variant].append(time.perf_counter()-started);states[variant]=result[0];outputs.append(jax.device_get(result))
        for (key,a),b in zip(jax.tree.leaves_with_path(outputs[0][0]),jax.tree.leaves(outputs[1][0])):
            label=jax.tree_util.keystr(key)
            if np.asarray(a).dtype.kind in 'fc' and np.size(a):
                difference=float(np.max(np.abs(a-b)));error=max(error,difference)
                leaf_errors[label]=max(leaf_errors.get(label,0.),difference)
            # MJX force/acceleration work arrays amplify float32 rounding;
            # retain tighter bounds on actual state, sensors and commands.
            derived='.data.' in label and not any(label.endswith('.'+field) for field in ('qpos','qvel','ctrl','sensordata','act'))
            if not derived:
                np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-5,err_msg=label)
        for left,right in zip(outputs[0][1:],outputs[1][1:]):
            for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-5)
        reward_error=max(reward_error,float(np.max(np.abs(outputs[0][2]-outputs[1][2]))))
    receipt=dict(maximum_lower_ticks=256,ppo_updates=0,variants=['83f0873_geometry_and_vmap_reset','local_geometry_and_guarded_batch_reset'],
        max_state_absolute_error=error,max_reward_absolute_error=reward_error,leaf_max_absolute_errors=leaf_errors,compile_seconds=compilation,step_seconds=timings,
        reset_on_last_call=True,throughput_benchmark=False,
        comparison_scope='kinematics/sensors/controller/history/commands/observations/reward/flags/statistics at2e-5; derived MJX work arrays measured separately')
    out=DiskPath(os.environ.get('STTW_PATH_CLEANUP_RECEIPT',str(tmp_path/'cleanup_gpu.json')))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
