"""Matched real-physics check of extracted versus prior inline hold continuation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan',type=Path,required=True)
    plan_path = p.parse_args().plan.resolve(); plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    from sttw_control.preference_best import atomic_json
    status = dict(phase='running',stage='loading',pid=os.getpid(),training_steps=0,
                  active_lower_ticks=0,maximum_lower_ticks=plan['maximum_lower_ticks'])
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(root/'status.json',status)
    update()
    try:
        import numpy as np
        import jax
        import jax.numpy as j
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.balance_probe import make_lower_tick
        from sttw_control.physical_continuation import make_hold_last_tick,lower_risk_observation
        from sttw_control.runtime import configure_compilation_cache
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        for path,sha in plan['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
                raise ValueError('source identity changed: '+path)
        decl=json.loads(Path(plan['declaration']).read_text())
        env,_=load_environment_contact_only(decl['config'],decl['source'],'terminal_contract_v2')
        initial=[]
        for row in plan['cases']:
            state=jax.tree.map(j.asarray,pickle.loads(Path(row['snapshot']).read_bytes()))
            if not bool(state.domain_exit) or bool(state.fault|state.physical.failed):
                raise ValueError('Expected actual domain-held, live, finite source state')
            initial.extend([state,state])
        stack=lambda states:jax.tree.map(lambda *xs:j.stack(xs),*states)
        states=stack(initial)
        references=j.stack([s.physical.governor.current_reference for s in initial])
        lower=make_lower_tick(env,references[0],contact_only=True)
        extracted=make_hold_last_tick(lower,dt=env.cc.dt)
        pulse=j.zeros(5)
        def prior(state,t,ref):
            physical,local,log=lower(state.physical,state.lower,pulse,t,ref)
            end=state.replace(physical=physical,lower=local,tick=state.tick+1,
                fault=state.fault|log['lower_fault']|log['nonfinite'],
                yaw_rate=(physical.yaw_unwrapped-state.physical.yaw_unwrapped)/env.cc.dt)
            return end,log
        keys=['actual_xy','actual_forward_speed','phi','phi_dot','actual_delta',
              'contact_failure','nonfinite','lower_fault','reference','applied_force_impulse']
        def one(state,new,ref,step):
            def active(state):
                t=step*env.cc.dt
                def fresh(s):
                    end,log=extracted(s,pulse,t,ref)
                    return end,{k:log[k] for k in keys}
                def old(s):
                    end,log=prior(s,t,ref)
                    return end,{k:log[k] for k in keys}
                end,log=jax.lax.cond(new,fresh,old,state)
                measurement,pose,speed=env.physics.observe(end.physical.data)
                view=lower_risk_observation(end,measurement=measurement,pose=pose,
                    true_speed=speed,held_reference=ref,
                    remaining_s=(plan['ticks_per_branch']-step-1)*env.cc.dt)
                log.update(active_tick=j.bool_(True),risk_view_valid=view['valid'],
                    upper_active=view['upper_action_valid'])
                return end,log
            template=jax.eval_shape(active,state)[1]
            inactive=jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),template)
            return jax.lax.cond(state.physical.failed|state.fault,
                lambda s:(s,inactive),active,state)
        modes=j.asarray([True,False]*len(plan['cases']))
        @jax.jit
        def execute(states):
            return jax.lax.scan(lambda st,i:jax.vmap(one)(st,modes,references,
                j.full((len(initial),),i)),states,j.arange(plan['ticks_per_branch']))
        update(stage='compiling_and_integrating')
        end,logs=execute(states);jax.block_until_ready(end)
        logs=jax.device_get(logs);end=jax.device_get(end)
        count=int(logs['active_tick'].sum())
        # Always retain costly integration before any validation assertion.
        np.savez_compressed(root/'raw_logs.npz',**logs)
        with (root/'raw_final_states.pkl').open('wb') as f:pickle.dump(end,f)
        update(stage='verifying',active_lower_ticks=count)
        assert not np.asarray(end.fault).any()
        assert not logs['nonfinite'].any() and not logs['lower_fault'].any()
        rows=[]
        for i,row in enumerate(plan['cases']):
            a,b=2*i,2*i+1
            errors=[]
            integer_mismatches=0
            for leaf in jax.tree.leaves(end):
                aa,bb=np.asarray(leaf[a]),np.asarray(leaf[b])
                if aa.dtype.kind in 'biu': integer_mismatches+=int(np.count_nonzero(aa!=bb))
                else:
                    if aa.size:errors.append(float(np.max(np.abs(aa-bb))))
            # User accepts small simulator differences; keep numeric diagnostics,
            # while exact event/clock/reference contracts remain explicit checks.
            for k in ['active_tick','contact_failure','nonfinite','lower_fault']:
                np.testing.assert_array_equal(logs[k][:,a],logs[k][:,b])
            assert int(end.tick[a])==int(end.tick[b])
            assert int(end.physical.physical_tick[a])==int(end.physical.physical_tick[b])
            for original,now in zip(jax.tree.leaves(initial[a].history),jax.tree.leaves(end.history)):
                np.testing.assert_array_equal(np.asarray(original),now[a])
            assert not logs['upper_active'][:,a].any()
            np.testing.assert_array_equal(logs['reference'][:,a][logs['active_tick'][:,a]],
                np.broadcast_to(np.asarray(references[a]),(int(logs['active_tick'][:,a].sum()),2)))
            for branch,index in [('extracted',a),('prior_inline',b)]:
                np.savez_compressed(root/f'case_{i}_{branch}.npz',
                    **{k:v[:,index] for k,v in logs.items()},
                    initial_xy=np.asarray(initial[index].physical.data.qpos[:2]),
                    initial_yaw=float(initial[index].physical.yaw_unwrapped))
            rows.append(dict(source=row,active_ticks_each=int(logs['active_tick'][:,a].sum()),
                maximum_final_state_error=max(errors),
                final_integer_mismatches=integer_mismatches,
                maximum_xy_difference_m=float(np.max(np.abs(logs['actual_xy'][:,a]-logs['actual_xy'][:,b]))),
                maximum_roll_difference_rad=float(np.max(np.abs(logs['phi'][:,a]-logs['phi'][:,b]))),
                maximum_speed_difference_m_s=float(np.max(np.abs(logs['actual_forward_speed'][:,a]-logs['actual_forward_speed'][:,b]))),
                final_contact=bool(logs['contact_failure'][:,a].any()),
                task_tick_increase=int(end.tick[a]-initial[a].tick),
                physical_tick_increase=int(end.physical.physical_tick[a]-initial[a].physical.physical_tick),
                source_snapshot_unmodified=hashlib.sha256(Path(row['snapshot']).read_bytes()).hexdigest()==plan['source_sha256'][row['snapshot']]))
        assert count<=plan['maximum_lower_ticks']
        assert all(r['source_snapshot_unmodified'] for r in rows)
        atomic_json(root/'receipt.json',dict(cases=rows,active_lower_ticks=count,
            training_steps=0,scope='Engineering identity check, zero new force; not task or robustness improvement.'))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(rows),figsize=(6*len(rows),5),squeeze=False,layout='constrained')
        for i,row in enumerate(rows):
            ax=axes[0,i]
            for branch in ['extracted','prior_inline']:
                with np.load(root/f'case_{i}_{branch}.npz') as d:
                    active=d['active_tick'].astype(bool);xy=d['actual_xy'][active]
                    ax.plot(xy[:,0],xy[:,1],label=branch,linestyle='-' if branch=='extracted' else '--')
                    ax.scatter(*d['initial_xy'],marker='s',color='black')
                    ax.scatter(*xy[-1],marker='x' if row['final_contact'] else 'o')
            ax.set_title(row['source']['name']+'; zero added force')
            ax.set(xlabel='world x (m)',ylabel='world y (m)');ax.set_aspect('equal');ax.legend()
        fig.savefig(root/'xy_comparison.png',dpi=160);plt.close(fig)
        update(phase='completed',stage='contract_checked_numeric_differences_recorded',completed_cases=len(rows),
               active_lower_ticks=count)
        print(json.dumps(dict(cases=rows,active_lower_ticks=count),indent=2))
    except BaseException as exc:
        update(phase='error',stage='engineering_error',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__': main()
