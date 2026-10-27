"""Complete closed-loop batches and audited objective for reference-bias tools."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
from .preference_best import atomic_json, digest


def batch_size(cases, vectors, allowed=(8,16)):
    n=len(cases)
    if n not in allowed or not set(allowed)<={8,16,64} or np.asarray(vectors).shape!=(n,6):
        raise ValueError('explicitly allowed real lanes required; no implicit padding')
    return n


def nominal_objective(*, physical_outcome, goal_arrived, cost):
    if physical_outcome=='unknown' or goal_arrived is None or not np.isfinite(cost):
        raise ValueError('unknown engineering outcome cannot become infeasible control')
    if physical_outcome not in ['survived','contact']:raise ValueError('explicit physical outcome required')
    return float(cost) if physical_outcome=='survived' and goal_arrived else float('inf')


def encode_objective(row):
    """Strict JSON: known infeasible is explicit; unknown never becomes null."""
    result=dict(row);score=float(result['objective'])
    if np.isnan(score) or np.isneginf(score):raise ValueError('unknown objective cannot be encoded as infeasible')
    feasible=bool(np.isfinite(score))
    result.update(nominal_feasible=feasible,objective=score if feasible else None)
    return result


def score_raw(raw, case, config, path, *, targets=None, positions=None, feedback_spec=None):
    from .anticipatory_run import physical_outcome
    from .anticipatory_analysis import cost_accounting
    from .survival_endpoint_task_metrics import goal_arrival_metrics
    from .preference_trace_rescore import rescore_trace_alpha
    from .finite_value_evaluation import verify_gate_log
    from .reference_bias_float32_audit import verify_bias_log
    active=np.asarray(raw['active_tick'],bool)
    outcome=physical_outcome(active,raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
    if targets is not None:
        verify_bias_log(raw,positions=positions,targets=targets,case=case,config=config,spec=feedback_spec,path=path)
        verify_gate_log(raw,case)
    point=[np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in range(2)]
    goals={str(radius):goal_arrival_metrics(raw,goal_xy=point,radius_m=radius,dt=.005,horizon_s=10.) for radius in [.5,.75,1.]}
    costs=cost_accounting(raw,scale=config['path_reward']['scale'],dt=.005)
    own=rescore_trace_alpha(raw,alpha=case['alpha'],config=config)
    np.testing.assert_allclose(own['scored_reward'],raw['scored_tick_reward'],rtol=2e-5,atol=2e-6)
    crossed={str(a):rescore_trace_alpha(raw,alpha=a,config=config)['scored_cost'] for a in [0.,.5,1.]}
    row=dict(case,physical_outcome=outcome,physical_ticks=int(active.sum()),goals=goals,
             cross_user_costs=crossed,**costs)
    row['objective']=nominal_objective(physical_outcome=outcome,goal_arrived=goals['1.0']['goal_arrived'],cost=costs['scored_cost'])
    return row


class BiasRuntime:
    """GPU vectorized full-state rollouts; no policy optimizer or event selector."""
    def __init__(self, root):
        import jax
        import jax.numpy as j
        from .contact_only_environment import load_environment_contact_only
        from .goal_preview_baseline import GoalPreviewEnv
        from .path_command_policy import make_path_actor
        from .reference_bias_optimization import make_bias_step
        from .finite_value_evaluation import rebind_path_context
        from .geometric_path import Path as GeometricPath
        from .runtime import configure_compilation_cache
        self.root=Path(root).resolve();p=json.loads((self.root/'plan.json').read_text());self.plan=p
        if digest(self.root/'plan.json')!=json.loads((self.root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('changed plan')
        for file,sha in {**p['sealed_files'],**p['source_sha256']}.items():
            if digest(file)!=sha:raise ValueError('changed frozen source/input '+file)
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        self.cfg=json.loads((self.root/'common_environment_config.json').read_text());decl=json.loads((self.root/'source_declaration.json').read_text())
        env,_=load_environment_contact_only(self.cfg,decl['source'],'terminal_contract_v2');self.env=GoalPreviewEnv(env)
        self.paths={};self.path_arrays={};self.origins={}
        for name,c in p['contexts'].items():
            with np.load(c['path']) as z:self.path_arrays[name]={k:z[k] for k in z.files}
            path=GeometricPath(*(j.asarray(self.path_arrays[name][k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
            with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
            state=jax.tree.map(lambda x:j.asarray(x[0]),bank)
            if int(state.tick)!=0 or int(state.physical.physical_tick)!=c['initial_physical_tick']:raise ValueError('complete initial clock changed')
            self.origins[name]=rebind_path_context(state,path,j.float32(c['speed_m_s']),self.env);self.paths[name]=path
        with Path(p['model']['actor']).open('rb') as f:self.native=jax.tree.map(j.asarray,pickle.load(f))
        first=next(iter(self.origins));self.env.set_log_template(self.origins[first],self.paths[first],j.float32(p['contexts'][first]['speed_m_s']))
        one=make_bias_step(self.env,make_path_actor(self.cfg),horizon_s=10.,spec=p['feedback_spec'])
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        @jax.jit
        def chunk(carry,parameters,paths,speeds,pulses,modes):
            return jax.lax.scan(lambda c,_:jax.vmap(one)(c,parameters,paths,speeds,pulses,modes),carry,None,length=50)
        self.chunk=chunk

    def rollout(self,cases,vectors,directory,*,allowed_real_batch_sizes=(8,16)):
        import jax
        import jax.numpy as j
        from .causal_reference_baseline import initial_gate
        from .causal_recovery_relay import CausalRelayState,initial_gate as initial_relay
        from .reference_bias_optimization import bias_targets
        from .evaluation_clock import physical_pulse
        from .anticipatory_run import flatten_ticks
        out=Path(directory)
        if (out/'audit_recovery.json').exists():
            receipt=json.loads((out/'audit_recovery.json').read_text())
            inventory=json.loads((out/'inventory.json').read_text())
            if inventory['cases']!=cases or not np.array_equal(inventory['vectors'],vectors):
                raise ValueError('recovery cannot reuse different candidate identities')
            if digest(out/'results.json')!=receipt['results_sha256']:
                raise ValueError('recovered scores changed')
            rows=json.loads((out/'results.json').read_text())
            if len(rows)!=len(cases):raise ValueError('incomplete recovered batch')
            for row in rows:
                if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('recovered raw changed')
            return rows
        out.mkdir(parents=True,exist_ok=False)
        n=batch_size(cases,vectors,allowed_real_batch_sizes)
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        states=[self.origins[c['context']].replace(alpha=j.float32(c['alpha'])) for c in cases]
        targets=[bias_targets(v) for v in vectors]
        parameters=stack([dict(actor=dict(native=self.native,keep_channels=j.array([True,True])),feedback_gain_rad=j.float32(.1),
                               bias_positions=j.asarray(self.plan['contexts'][c['context']]['bias_positions_m']),bias_targets=j.asarray(t)) for c,t in zip(cases,targets)])
        carry=CausalRelayState(stack(states),stack([initial_gate() for _ in cases]),j.zeros(n,j.int32),stack([initial_relay() for _ in cases]))
        with (out/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        paths=stack([self.paths[c['context']] for c in cases]);speeds=j.asarray([c['requested_speed_m_s'] for c in cases],j.float32)
        pulses=j.asarray([physical_pulse(c,self.plan['contexts'][c['context']]) for c in cases],j.float32);modes=j.ones(n,j.int32)
        data=[];active_ticks=0;allocated=0;start=time.monotonic()
        atomic_json(out/'inventory.json',dict(cases=cases,vectors=np.asarray(vectors).tolist()))
        for block in range(10):
            carry,logs=self.chunk(carry,parameters,paths,speeds,pulses,modes);jax.block_until_ready(carry)
            host,logs=jax.device_get((carry,logs));logs=flatten_ticks(logs);data.append(logs)
            np.savez_compressed(out/f'chunk_{block:02d}.npz',**logs)
            active_ticks+=int(logs['active_tick'].sum());allocated+=200*n
            atomic_json(out/'status.json',dict(phase='running',blocks=block+1,active_lower_ticks=active_ticks,allocated_lower_slots=allocated,elapsed_s=time.monotonic()-start))
            if np.any(host.state.fault):raise RuntimeError('engineering fault; complete chunk retained, no implicit retry')
            if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
        with (out/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
        combined={k:np.concatenate([x[k] for x in data]) for k in data[0]}
        rows=[];(out/'cases').mkdir()
        atomic_json(out/'status.json',dict(phase='physical_completed',stage='auditing',completed_physical_cases=n,
            active_lower_ticks=active_ticks,allocated_lower_slots=allocated,elapsed_s=time.monotonic()-start))
        for lane in range(n):
            np.savez_compressed(out/'cases'/f'case_{lane:02d}.npz',**{k:v[:,lane] for k,v in combined.items()})
        for lane,(case,t) in enumerate(zip(cases,targets)):
            raw={k:v[:,lane] for k,v in combined.items()};file=out/'cases'/f'case_{lane:02d}.npz'
            row=score_raw(raw,case,self.cfg,self.path_arrays[case['context']],targets=t,
                          positions=self.plan['contexts'][case['context']]['bias_positions_m'],feedback_spec=self.plan['feedback_spec'])
            row.update(raw_file=str(file.resolve()),raw_sha256=digest(file),vector=np.asarray(vectors[lane]).tolist())
            # JSON remains strict; infeasibility is explicit, not a null unknown.
            rows.append(encode_objective(row))
        atomic_json(out/'results.json',rows)
        atomic_json(out/'status.json',dict(phase='completed',completed_cases=n,active_lower_ticks=active_ticks,allocated_lower_slots=allocated,elapsed_s=time.monotonic()-start,optimizer_updates=0))
        return rows
