"""Prepare/execute one four-episode frozen stochastic sampling check; no optimizer."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

from sttw_control.preference_best import atomic_json, config_hash, digest


def prepare(root):
    import torch
    from prepare_short_recovery_campaign import register
    root.mkdir(exist_ok=False, parents=True)
    prior = Path('runs/causal_reference_full_task_v1_attempt2/bank3/plan.json').resolve()
    old = json.loads(prior.read_text())
    declaration = Path(old['source_declaration'])
    assert digest(declaration) == old['source_declaration_sha256']
    decl = json.loads(declaration.read_text())
    model = decl['policies']['H75']
    for kind in ['actor', 'checkpoint']:
        assert digest(model[kind]) == model[kind+'_sha256']
    checkpoint = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
    log_std = checkpoint['policy']['log_std'].tolist()
    assert decl['source']['bank_index'] == 3 and decl['config']['training_future_phase_C']['episode_s'] == 10.
    cases = []
    for context, (route, speed, force) in enumerate([('left90_R1p1', 1.8, 32.), ('right90_R1p1', 1.5, -32.)]):
        for f in [0., force]:
            row = next(x for x in old['cases'] if (x['route'], x['speed'], x['force_N'], x['arm']) == (route, speed, f, 'original'))
            cases.append({k: row[k] for k in ['route', 'speed', 'alpha', 'force_N',
                'pulse_onset_task_tick', 'pulse_duration_s', 'world_direction']})
            cases[-1]['rng_fold_in'] = context
    files = list(Path('learning/src/sttw_control').rglob('*.py'))+[Path('learning/cli/collect_survival_full_prefix.py'),
        Path('docs/superpowers/plans/2026-10-24-fresh-survival-collector.md')]
    sources = {}
    worktree = Path.cwd().resolve()
    for file in files:
        path = file.resolve()
        sources[str(path)] = digest(path)
        archived = root/'source_at_launch'/path.relative_to(worktree)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(path.read_bytes())
    policy = dict(actor_sha256=model['actor_sha256'], checkpoint_sha256=model['checkpoint_sha256'],
        log_std=log_std, action='raw_diagonal_Gaussian_before_tanh',
        config_sha256=config_hash(decl['config']), survival_baseline='constant_probability_0.5')
    plan = dict(project='STTW_CONTROL', purpose='Fresh task0 stochastic collector engineering, not training/effect evaluation',
        source_declaration=str(declaration), source_declaration_sha256=digest(declaration),
        reference_plan=str(prior), reference_plan_sha256=digest(prior),
        path_evidence_directory=old['path_evidence_directory'], model=model,
        sampling=policy, collection_policy_id=config_hash(policy), seed=2026102407,
        cases=cases, declared_cases=4, horizon_s=10., maximum_lower_ticks=8000,
        training_steps=0, optimizer_updates=0, code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources, acceleration=['vmap4', 'scan50', 'persistent_JAX_cache'],
        stop='One four-episode engineering run. Keep all failures/exposure; no std/seed/force tuning or automatic retry.')
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'status.json', dict(phase='prepared', declared_cases=4, completed_cases=0,
        active_lower_ticks=0, training_steps=0, optimizer_updates=0))
    register(root/'ACTIVE_RUN.json', 'STTW_CONTROL 完整前缀随机采样工程检查', root/'status.json')
    print('prepared4 episodes,maximum8000 lower ticks,0optimizer updates', flush=True)


def execute(root):
    state = dict(phase='running', stage='loading', pid=os.getpid(), declared_cases=4,
                 completed_cases=0, active_lower_ticks=0, training_steps=0, optimizer_updates=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(root/'status.json', state)
    update()
    try:
        import numpy as np
        import jax
        import jax.numpy as j
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
        from sttw_control.geometric_path import build_path
        from sttw_control.runtime import configure_compilation_cache
        from sttw_control.survival_collection import SurvivalCollectionState, make_survival_collection_step
        plan = json.loads((root/'plan.json').read_text())
        for p, expected in plan['source_sha256'].items():
            if digest(p) != expected:
                raise ValueError('sampling source changed '+p)
        assert digest(plan['source_declaration']) == plan['source_declaration_sha256']
        assert digest(plan['reference_plan']) == plan['reference_plan_sha256']
        decl = json.loads(Path(plan['source_declaration']).read_text())
        cfg, source = decl['config'], decl['source']
        model = plan['model']
        for kind in ['actor', 'checkpoint']:
            assert digest(model[kind]) == model[kind+'_sha256']
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env, initial = load_environment_contact_only(cfg, source, 'terminal_contract_v2')
        with Path(model['actor']).open('rb') as stream:
            params = jax.tree.map(j.asarray, pickle.load(stream))
        validate_actor_parameters(params, cfg)
        log_std = j.asarray(plan['sampling']['log_std'], dtype=j.float32)
        pose = np.asarray(env.physics.helpers.pose(initial.data))
        paths = {}
        for route in decl['routes']:
            path = build_path(route['builder'], pose, cfg)
            with np.load(Path(plan['path_evidence_directory'])/(route['name']+'_path.npz')) as z:
                np.testing.assert_allclose(path.xy, z['xy'], atol=1e-4, rtol=0.)
            paths[route['name']] = path
            np.savez_compressed(root/(route['name']+'_path.npz'), xy=np.asarray(path.xy), s=np.asarray(path.s), goal=np.asarray(path.goal))
        cases = plan['cases']
        initial_states = [env.reset(initial, paths[c['route']], j.float32(c['speed']), j.float32(c['alpha'])) for c in cases]
        assert all(int(s.tick) == 0 and not bool(s.fault|s.domain_exit|s.physical.failed) for s in initial_states)
        stack = lambda xs:jax.tree.map(lambda *x:j.stack(x), *xs)
        carry = stack([SurvivalCollectionState(s, jax.random.fold_in(jax.random.PRNGKey(plan['seed']), c['rng_fold_in']),
                                               j.zeros(2), j.bool_(False)) for s, c in zip(initial_states, cases)])
        batch_paths = stack([paths[c['route']] for c in cases])
        speeds = j.asarray([c['speed'] for c in cases], dtype=j.float32)
        absolute_start = float(initial.physical_tick)*env.cc.dt
        pulses = j.asarray([[absolute_start+c['pulse_onset_task_tick']*.005,
                             absolute_start+c['pulse_onset_task_tick']*.005+c['pulse_duration_s'],
                             *[c['force_N']*d for d in c['world_direction']]] for c in cases], dtype=j.float32)
        env.set_log_template(initial_states[0], paths[cases[0]['route']], speeds[0])
        one = make_survival_collection_step(env, make_path_actor(cfg), horizon_s=plan['horizon_s'])
        @jax.jit
        def chunk(carry):
            return jax.lax.scan(lambda c,_:jax.vmap(one, in_axes=(0,None,None,0,0,0))(
                c,params,log_std,batch_paths,speeds,pulses), carry, None, length=50)
        atomic_json(root/'manifest.json', dict(plan=plan, config=cfg, source=source,
            policy_params_frozen=True, physical_identity=env.physics.bundle.identity,
            terminal_contract='contact_only + irreversible domain-hold, task0 to10s',
            original_cost_formulas=True, old_roll_terminal_training_reproduction=False,
            physical_value_baseline='constant0.5, untrained', optimizer_created=False))
        with (root/'initial_carry.pkl').open('wb') as stream:
            pickle.dump(jax.device_get(carry), stream)
        started = time.monotonic()
        decisions, logs = [], []
        update(stage='compile_and_collect')
        for block in range(11):
            carry, (decision, log) = chunk(carry)
            jax.block_until_ready(carry)
            decision, log = jax.device_get((decision, log))
            flat = {k:np.swapaxes(v,1,2).reshape((200,len(cases))+v.shape[3:]) for k,v in log.items()}
            decisions.append(decision)
            logs.append(flat)
            np.savez_compressed(root/f'chunk_{block:02d}.npz', **{'decision_'+k:v for k,v in decision.items()},
                                **{'lower_'+k:v for k,v in flat.items()})
            state['active_lower_ticks'] += int(flat['active_tick'].sum())
            assert state['active_lower_ticks'] <= plan['maximum_lower_ticks']
            update(stage='collecting', completed_blocks=block+1, declared_maximum_blocks=11,
                   elapsed_collection_s=time.monotonic()-started)
            if bool(j.any(carry.state.fault)):
                raise RuntimeError('collector engineering fault; saved block retained, no update/retry')
            stopped = carry.state.physical.failed|(carry.state.tick>=2000)
            if bool(j.all(stopped)):
                break
        assert bool(j.all(carry.state.physical.failed|(carry.state.tick>=2000)))
        physical = {k:np.concatenate([x[k] for x in logs], axis=0) for k in logs[0]}
        decision_data = {k:np.concatenate([x[k] for x in decisions], axis=0) for k in decisions[0]}
        for i in range(len(cases)):
            np.savez_compressed(root/f'case_{i:03d}.npz', **{k:v[:,i] for k,v in physical.items()},
                **{'decision_'+k:v[:,i] for k,v in decision_data.items()})
        with (root/'final_carry.pkl').open('wb') as stream:
            pickle.dump(jax.device_get(carry), stream)
        for kind in ['actor', 'checkpoint']:
            assert digest(model[kind]) == model[kind+'_sha256']
        update(stage='all_physics_complete', completed_cases=4)
        analyze(root, plan, cfg)
        update(phase='completed', stage='targets_provenance_and_XY_ready')
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


def analyze(root, plan, cfg):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sttw_control.path_command_policy import make_torch_path_network
    from sttw_control.path_command_reward import configured_cost_keys
    from sttw_control.survival_credit import resolve_survival_credit
    from sttw_control.balance_probe import classify_contact_balance
    import jax
    import jax.numpy as j
    from sttw_control.survival_collection import sample_gaussian_action
    checkpoint = torch.load(plan['model']['checkpoint'], map_location='cpu', weights_only=False)
    actor = make_torch_path_network(351, 2, cfg)
    actor.load_state_dict({k[len('actor.'):]:v for k,v in checkpoint['policy'].items() if k.startswith('actor.')}, strict=True)
    actor.eval()
    std = checkpoint['policy']['log_std'].exp()
    with (root/'initial_carry.pkl').open('rb') as stream:
        initial_carry=pickle.load(stream)
    with (root/'final_carry.pkl').open('rb') as stream:
        final_carry=pickle.load(stream)
    rows = []
    fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')
    for i, case in enumerate(plan['cases']):
        path = root/f'case_{i:03d}.npz'
        with np.load(path, allow_pickle=False) as z:
            data = {k:z[k] for k in z.files}
        active = data['active_tick']
        valid = data['decision_valid']
        assert data['physical_tick'][active][0] == 0
        action_mask = np.zeros_like(active)
        action_mask[np.flatnonzero(valid)*4] = True
        credit = resolve_survival_credit(active_tick=active, actor_action_valid=action_mask,
            contact_failure=data['contact_failure'], engineering_fault=data['nonfinite']|data['lower_fault']|data['policy_fault'],
            horizon_reached=data['horizon_reached'], physical_tick=data['physical_tick'],
            window_start_tick=0, window_ticks=2000, collection_policy_id=plan['collection_policy_id'],
            learner_policy_id=plan['collection_policy_id'])
        physical = classify_contact_balance(data, horizon_s=10.)
        assert credit['outcome'] == ('contact' if physical['event_contact'] else 'survived')
        assert credit['physical_ticks'] == int(active.sum())
        assert int(final_carry.state.physical.physical_tick[i]-initial_carry.state.physical.physical_tick[i]) == int(active.sum())
        assert not np.any(action_mask & data['fallback_active'])
        assert np.array_equal(data['decision_task_tick'][valid], data['physical_tick'][np.flatnonzero(action_mask)])
        obs = torch.as_tensor(data['decision_observation'][valid])
        action = torch.as_tensor(data['decision_raw_action'][valid])
        assert torch.isfinite(obs).all() and torch.isfinite(action).all()
        with torch.no_grad():
            means = actor(obs)
            logp = torch.distributions.Normal(means,std).log_prob(action).sum(-1).numpy()
        mean_error = float(np.max(np.abs(means.numpy()-data['decision_mean'][valid])))
        logp_error = float(np.max(np.abs(logp-data['decision_log_prob'][valid])))
        assert mean_error < 1e-4 and logp_error < 1e-3
        recreated, _, _ = jax.jit(jax.vmap(sample_gaussian_action,in_axes=(0,None,0)))(
            j.asarray(data['decision_mean'][valid]), j.asarray(plan['sampling']['log_std'],dtype=j.float32),
            j.asarray(data['decision_rng_key'][valid]))
        sampling_error=float(np.max(np.abs(np.asarray(recreated)-data['decision_raw_action'][valid])))
        assert sampling_error < 2e-5
        assert np.any(data['decision_raw_action'][valid] != data['decision_mean'][valid])
        assert np.all(data['decision_physical_value'][valid] == .5)
        np.savez_compressed(root/f'case_{i:03d}_credit.npz', actor_mask=credit['actor_mask'],
                            physical_targets=credit['targets'], decision_targets=credit['targets'][::4])
        expected = case['force_N']*case['pulse_duration_s']*np.asarray(case['world_direction'])
        impulse = data['applied_force_impulse'][active].sum(0, dtype=np.float64)
        full = np.linalg.norm(impulse-expected) <= max(2e-4, .01*np.linalg.norm(expected))
        rows.append(dict(case=i, **case, **physical, outcome=credit['outcome'],
            actor_decisions=credit['actor_actions'], hold_ticks=int(data['fallback_active'][active].sum()),
            mean_Torch_JAX_max_error=mean_error, log_probability_max_error=logp_error,
            saved_RNG_action_max_error=sampling_error,
            task_scored_return=float(data['scored_tick_reward'][active].sum(dtype=np.float64)),
            impulse_Ns=impulse.tolist(), expected_impulse_Ns=expected.tolist(), full_pulse_exposure=bool(full),
            raw_sha256=digest(path), collection_policy_id=plan['collection_policy_id']))
        ax=axes[i//2]
        xy=data['actual_xy'][active]
        ax.plot(xy[:,0],xy[:,1],label=f"F={case['force_N']:g}N; {credit['outcome']}",ls='-' if i%2==0 else '--')
        ax.scatter(*xy[-1],marker='x' if physical['event_contact'] else 'o',s=45)
        hold=np.flatnonzero(data['fallback_active']&active)
        if hold.size:
            ax.scatter(*data['actual_xy'][hold[0]],marker='s',s=40)
        if i%2==0:
            with np.load(root/(case['route']+'_path.npz')) as z:
                xy_ref=z['xy'][z['s']<=float(z['goal'])+2.]
                goal=np.array([np.interp(float(z['goal']),z['s'],z['xy'][:,k]) for k in range(2)])
            ax.plot(xy_ref[:,0],xy_ref[:,1],color='black',alpha=.4,label='fixed reference (through goal+2m)')
            ax.scatter(*goal,marker='+',color='black',s=70,label='original goal section')
            ax.set_title(f"{case['route']}, v={case['speed']:g}, alpha0")
        ax.set(xlabel='world X (m)',ylabel='world Y (m)')
        ax.set_aspect('equal',adjustable='datalim')
        ax.legend(fontsize=8)
    fig.suptitle('Fresh frozen stochastic sampling; x=contact, square=domain-hold, circle=window end')
    fig.savefig(root/'all_xy.png',dpi=170)
    plt.close(fig)
    for i, row in enumerate(rows):
        with np.load(root/f'case_{i:03d}.npz') as z:
            active=z['active_tick'];task=z['task_reward_valid']&active
            fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
            axes[0].plot(z['time'][task],z['scored_tick_reward'][task],label='task scored reward')
            for k,name in enumerate(configured_cost_keys(cfg)):
                axes[1].plot(z['time'][task],-cfg['path_reward']['scale']*.005*z['effective_costs'][task,k],label=name)
            axes[2].plot(z['time'][active],z['actual_forward_speed'][active],label='true body speed')
            axes[2].plot(z['time'][active],z['wheel_speed_proxy'][active],label='wheel-speed proxy')
            axes[2].plot(z['time'][active],z['reference'][active,0],label='governed speed reference')
            axes[2].axhline(row['speed'],color='black',ls='--',label='original requested speed')
            for ax in axes:
                ax.axvspan(row['pulse_onset_task_tick']*.005,row['pulse_onset_task_tick']*.005+.1,color='gray',alpha=.2)
                ax.set_xlabel('task time (s)');ax.legend(fontsize=7,ncol=3)
            axes[0].set_ylabel('scored reward');axes[1].set_ylabel('signed running components');axes[2].set_ylabel('speed (m/s)')
            fig.suptitle(f"case{i}, F={row['force_N']:g}N: {row['outcome']}; hold has no task reward; failure penalty separate")
            fig.savefig(root/f'case_{i:03d}_rewards.png',dpi=150);plt.close(fig)
    atomic_json(root/'results.json',dict(cases=rows, collection_policy_id=plan['collection_policy_id'],
        optimizer_updates=0, actor_parameters_changed=False,
        source_sha256={str(Path(__file__).resolve()):digest(__file__)},
        scope='Four correlated engineering episodes; no policy improvement or independent robustness inference.'))
    lines=['# Fresh full-prefix stochastic collection check', '', '[All XY](all_xy.png)', '',
        'Four fresh task0 episodes, fixed H75/local300, no optimizer. Physical-value baseline is untrained constant.5.',
        'Original V3 cost formulas with explicit contact/domain diagnostic termination; not old roll-terminal training reproduction.',
        'Common noise seed within force pairs. Fixed world pulse directions/times, not body-aligned to each stochastic onset.', '',
        '|Case|Force N|Physical outcome|Lower ticks|Actor decisions|Hold ticks|Full pulse|',
        '|---|---|---|---|---|---|---|']
    lines += [f"|{r['case']}|{r['force_N']}|{r['outcome']}|{r['active_ticks']}|{r['actor_decisions']}|{r['hold_ticks']}|{r['full_pulse_exposure']}|" for r in rows]
    lines += ['', 'All actual states/logs and decision observations/raw actions/joint log densities are retained. '
        'Targets resolved only after contact or the full physical window. No historical data used for PPO. '
        'Dual-objective optimizer, cross-rollout learner storage and training benefits are not implemented by this run.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
