"""Fixed terminal DEV evaluation. No training, best-checkpoint choice or adoption.

Raw chunk logs and complete states precede assertions; collection completion is
separate from full figure/report delivery. Failed runs are never auto-restarted.
"""
import copy
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
from .preference_best import atomic_json, digest


def _checked(path, sha):
    if digest(path)!=sha:raise ValueError('changed sealed input: '+str(path))


def prepare_run(protocol, output):
    protocol,output=Path(protocol).resolve(),Path(output).resolve()
    plan=json.loads((protocol/'plan.json').read_text())
    names={'cases.json':'cases_sha256','common_environment_config.json':'common_config_sha256',
           'nominal_geometry.json':'nominal_geometry_sha256'}
    for name,key in names.items():_checked(protocol/name,plan[key])
    for p,sha in plan['source_sha256'].items():_checked(p,sha)
    cases=json.loads((protocol/'cases.json').read_text())
    if len(cases)!=plan['declared_actual_episodes'] or len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('sealed case matrix count or identity mismatch')
    # Archive bytes as well as hashes; includes local uncommitted dependencies.
    package=Path(__file__).parent
    sources=list(package.glob('*.py'))+[Path(__file__).parents[2]/'cli'/'evaluate_survival_endpoint.py']
    sources=[p for p in sources if p.is_file()]
    output.mkdir(parents=True,exist_ok=False)
    for name in names:(output/name).write_bytes((protocol/name).read_bytes())
    archived=output/'source_snapshot';archived.mkdir()
    hashes={}
    for index,p in enumerate(sorted(sources)):
        hashes[str(p.resolve())]=digest(p)
        (archived/f'{index:03d}_{p.name}').write_bytes(p.read_bytes())
    atomic_json(output/'plan.json',dict(protocol=str(protocol),protocol_plan_sha256=digest(protocol/'plan.json'),
        protocol_plan=plan,source_sha256=hashes,optimizer_updates=0,
        max_chunks_per_group=11,steps_per_chunk=50,
        allocated_slots_note='Padding is counted separately; actual horizon remains2000 lower ticks per case.'))
    atomic_json(output/'status.json',dict(phase='prepared',completed_cases=0,declared_cases=len(cases),
        active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,endpoint_authenticated=False,
        figures_complete=False,checked_unix=time.time()))
    return output


def flatten_chunk(decisions, physical):
    """scan × environment × lower_tick -> chronological lower_tick × environment."""
    return decisions,{k:np.swapaxes(v,1,2).reshape((-1,v.shape[1])+v.shape[3:]) for k,v in physical.items()}


def stack_case_chunks(chunks,index):
    return tuple({k:np.concatenate([part[which][k][:,index] for part in chunks],axis=0)
                  for k in chunks[0][which]} for which in (0,1))


def paired_outcomes(rows,reference,candidate):
    def selected(role):
        out={}
        for r in rows:
            if r['role']!=role:continue
            key=(r['panel'],r['context_index'],r['alpha'],r['replicate'])
            if key in out:raise ValueError('duplicate paired case')
            out[key]=r['outcome']
        return out
    a,b=selected(reference),selected(candidate)
    if a.keys()!=b.keys():raise ValueError('unmatched comparison conditions')
    counts=dict(reference=reference,candidate=candidate,paired=len(a),gain=0,loss=0,
                both_safe=0,both_contact=0,unknown=0)
    for key,x in a.items():
        y=b[key]
        if x not in ('contact','survived') or y not in ('contact','survived'):name='unknown'
        elif x==y:name='both_safe' if x=='survived' else 'both_contact'
        else:name='gain' if y=='survived' else 'loss'
        counts[name]+=1
    return counts


def execute_run(output):
    """Collect only after authenticating both completed64 native checkpoints."""
    import jax
    import jax.numpy as j
    from .runtime import configure_compilation_cache
    from .contact_only_environment import load_environment_contact_only
    from .path_command_policy import make_path_actor
    from .geometric_path import build_path
    from .survival_collection import SurvivalCollectionState
    from .survival_endpoint_models import authenticate_endpoint
    from .survival_endpoint_collection import evaluation_config,make_endpoint_collection_step
    from .survival_endpoint_batch import partition_cases,make_endpoint_chunk,simulation_alpha
    from .survival_endpoint_analysis import summarize_case,rescore_v3
    output=Path(output).resolve()
    plan=json.loads((output/'plan.json').read_text());status=json.loads((output/'status.json').read_text())
    if status['phase']!='prepared':raise ValueError('fresh prepared run required; no implicit retry/resume')
    protocol=Path(plan['protocol']);p=plan['protocol_plan']
    _checked(protocol/'plan.json',plan['protocol_plan_sha256'])
    for name,key in [('cases.json','cases_sha256'),('common_environment_config.json','common_config_sha256'),
                     ('nominal_geometry.json','nominal_geometry_sha256')]:_checked(output/name,p[key])
    for path,sha in {**p['source_sha256'],**plan['source_sha256']}.items():_checked(path,sha)
    # Rejection of an unfinished training run leaves this prepared artifact reusable.
    authenticated=authenticate_endpoint(p['training_run'])
    cfg=json.loads((output/'common_environment_config.json').read_text())
    if cfg!=authenticated['config']:raise ValueError('common config differs from authenticated endpoint')
    atomic_json(output/'authenticated_models.json',authenticated)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(output/'status.json',status)
    update(phase='running',stage='loading_environment',pid=os.getpid(),endpoint_authenticated=True)
    try:
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        train=json.loads((Path(p['training_run'])/'plan.json').read_text())
        declaration=json.loads(Path(train['source_declaration']).read_text())
        env,initial=load_environment_contact_only(cfg,declaration['source'],'terminal_contract_v2')
        cases=json.loads((output/'cases.json').read_text())
        nominal=json.loads((output/'nominal_geometry.json').read_text())
        configs={'physical':cfg,'nominal':dict(copy.deepcopy(cfg),path=nominal['path'])}
        pose=np.asarray(env.physics.helpers.pose(initial.data));paths={};routefiles={}
        (output/'reference_paths').mkdir()
        for geometry,routes in [('physical',declaration['routes']),('nominal',nominal['routes'])]:
            for route in routes:
                key=(geometry,route['name']);path=build_path(route['builder'],pose,configs[geometry]);paths[key]=path
                name=f'{geometry}_{route["name"]}.npz';routefiles['/'.join(key)]=name
                np.savez_compressed(output/'reference_paths'/name,**{k:np.asarray(getattr(path,k)) for k in
                                    ('xy','s','heading','curvature','goal','turn_end')})
        atomic_json(output/'reference_paths'/'index.json',routefiles)
        groups=partition_cases(cases,{k:tuple(v.s.shape) for k,v in paths.items()})
        atomic_json(output/'groups.json',groups)
        models=authenticated['models'];parameters={}
        for role,model in models.items():
            if model['actor'] is not None:
                with Path(model['actor']).open('rb') as stream:parameters[role]=jax.tree.map(j.asarray,pickle.load(stream))
        rows=[];utilities=[];caseindex=[];began=time.monotonic()
        (output/'cases').mkdir()
        stack=lambda items:jax.tree.map(lambda *x:j.stack(x),*items)
        for group in groups:
            directory=output/group['group_id'];directory.mkdir();indices=group['case_indices']
            selected=[cases[i] for i in indices];geometry=group['geometry']
            local=copy.copy(env);local.config=evaluation_config(configs[geometry],speed_floor=group['speed_reference_floor_m_s'])
            states=[local.reset(initial,paths[(geometry,c['route'])],j.float32(c['speed']),j.float32(simulation_alpha(c))) for c in selected]
            if any(int(s.tick)!=0 or bool(s.fault|s.domain_exit|s.physical.failed) for s in states):
                raise ValueError('invalid common full-state episode initialization')
            local.set_log_template(states[0],paths[(geometry,selected[0]['route'])],j.float32(selected[0]['speed']))
            keys=[jax.random.PRNGKey(p['seed']) if c['rng_fold_in'] is None else
                  jax.random.fold_in(jax.random.PRNGKey(p['seed']),c['rng_fold_in']) for c in selected]
            carry=stack([SurvivalCollectionState(s,k,j.zeros(2),j.bool_(False)) for s,k in zip(states,keys)])
            zero=group['inference_mode']=='zero'
            params=None if zero else stack([parameters[c['role']] for c in selected])
            std=j.zeros((len(selected),2)) if zero else j.asarray([models[c['role']]['log_std'] for c in selected])
            batchpath=stack([paths[(geometry,c['route'])] for c in selected]);speeds=j.asarray([c['speed'] for c in selected])
            start=float(initial.physical_tick)*local.cc.dt
            pulses=j.asarray([[start+c.get('pulse_onset_task_tick',0)*local.cc.dt,
                start+c.get('pulse_onset_task_tick',0)*local.cc.dt+c.get('pulse_duration_s',0.),
                *(c.get('force_N',0.)*np.asarray(c.get('world_direction',[0.,0.,0.])))] for c in selected],dtype=j.float32)
            one=make_endpoint_collection_step(local,None if zero else make_path_actor(cfg),horizon_s=p['horizon_s'],mode=group['inference_mode'])
            chunk=make_endpoint_chunk(one,steps_per_chunk=plan['steps_per_chunk'],zero_correction=zero)
            with (directory/'initial.pkl').open('wb') as stream:pickle.dump(jax.device_get(carry),stream)
            chunks=[];ticks=0;groupstart=time.monotonic()
            for block in range(plan['max_chunks_per_group']):
                update(stage='collecting',current_group=group['group_id'],group_cases=len(selected),current_block=block)
                carry,(d,physical)=chunk(carry,params,std,batchpath,speeds,pulses);jax.block_until_ready(carry)
                host,d,physical=jax.device_get((carry,d,physical));d,physical=flatten_chunk(d,physical)
                # Retain all evidence before outcome/accounting assertions.
                np.savez_compressed(directory/f'chunk_{block:02d}.npz',**{'decision_'+k:v for k,v in d.items()},
                                    **{'lower_'+k:v for k,v in physical.items()})
                with (directory/'latest_state.pkl').open('wb') as stream:pickle.dump(host,stream)
                chunks.append((d,physical));count=int(physical['active_tick'].sum());ticks+=count
                update(active_lower_ticks=status['active_lower_ticks']+count,
                    allocated_lower_slots=status['allocated_lower_slots']+plan['steps_per_chunk']*4*len(selected),
                    elapsed_s=time.monotonic()-began)
                if status['active_lower_ticks']>p['maximum_lower_ticks']:raise ValueError('actual physics limit exceeded')
                if np.any(host.state.fault):raise ValueError('engineering fault; raw trajectories retained, unknown not negative')
                done=host.state.physical.failed|(host.state.tick>=round(p['horizon_s']/local.cc.dt))
                if np.all(done):break
            else:raise ValueError('complete-window collection missing after declared padding allowance')
            grouprows=[]
            for position,(index,c) in enumerate(zip(indices,selected)):
                d,physical=stack_case_chunks(chunks,position);name=f'case_{index:04d}.npz'
                np.savez_compressed(output/'cases'/name,**{'decision_'+k:v for k,v in d.items()},**physical)
                identity=models[c['role']].get('actor_sha256','exact_zero_correction')
                row=summarize_case(c,physical,d,horizon_s=p['horizon_s'],dt_s=local.cc.dt,policy_id=identity)
                rows.append(row);grouprows.append(row)
                caseindex.append(dict(index=index,case_id=c['case_id'],file='cases/'+name,
                    sha256=digest(output/'cases'/name),group=group['group_id'],policy_id=identity,
                    reference='reference_paths/'+routefiles[geometry+'/'+c['route']]))
                if geometry=='nominal':
                    for alpha in (0.,.25,.5,.75,1.):
                        score=rescore_v3(physical,user_alpha=alpha,config=cfg);score.pop('per_lower_tick_reward')
                        utilities.append(dict(case_id=c['case_id'],**score))
            atomic_json(directory/'summary.json',dict(rows=grouprows,active_lower_ticks=ticks,elapsed_s=time.monotonic()-groupstart))
            atomic_json(output/'case_index.json',caseindex);atomic_json(output/'results.json',rows)
            atomic_json(output/'nominal_counterfactual_utilities.json',utilities)
            update(completed_cases=len(rows),completed_groups=groups.index(group)+1)
        if len(rows)!=p['declared_actual_episodes']:raise ValueError('incomplete actual episode matrix')
        if any(r['outcome'] not in ('contact','survived') for r in rows):raise ValueError('unresolved physical outcome')
        comparisons=[paired_outcomes(rows,a,b) for a,b in [('parent','task_only'),('parent','physical_credit'),
                    ('task_only','physical_credit'),('parent','parent_speed_floor')]]
        atomic_json(output/'paired_physical_outcomes.json',comparisons)
        for path,sha in {**p['source_sha256'],**plan['source_sha256']}.items():_checked(path,sha)
        update(phase='completed',stage='collection_and_numeric_analysis_complete_figures_pending',
               figures_complete=False,elapsed_s=time.monotonic()-began)
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise
