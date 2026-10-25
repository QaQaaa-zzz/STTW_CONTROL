"""Trace all eight actual batch layouts with original parent weights; zero physics.

Parent weights fill learned-role slots ONLY to check shape compatibility. This
never authenticates or evaluates a missing trained endpoint.
"""
import argparse,copy,json,os,pickle,time
from pathlib import Path
from sttw_control.preference_best import atomic_json,digest


def main(protocol,output):
    output.mkdir(exist_ok=False,parents=True)
    status=dict(phase='running',pid=os.getpid(),stage='loading',physics_steps=0,optimizer_steps=0,completed_groups=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(output/'status.json',status)
    update()
    from prepare_short_recovery_campaign import register
    register(output/'ACTIVE_RUN.json','STTW_CONTROL 终点评测八组真实批量零积分核验',output/'status.json')
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        import torch
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.geometric_path import build_path
        from sttw_control.survival_collection import SurvivalCollectionState
        from sttw_control.survival_endpoint_collection import evaluation_config,make_endpoint_collection_step
        from sttw_control.survival_endpoint_batch import partition_cases,simulation_alpha,make_endpoint_chunk
        p=json.loads((protocol/'plan.json').read_text())
        for key,sha in p['source_sha256'].items():
            if digest(key)!=sha:raise ValueError('sealed protocol source changed: '+key)
        training=Path(p['training_run']);train=json.loads((training/'plan.json').read_text())
        for key,sha in train['source_sha256'].items():
            if digest(key)!=sha:raise ValueError('active training source changed: '+key)
        cfg=json.loads((protocol/'common_environment_config.json').read_text())
        nominal=json.loads((protocol/'nominal_geometry.json').read_text());cases=json.loads((protocol/'cases.json').read_text())
        for name,key in [('cases.json','cases_sha256'),('common_environment_config.json','common_config_sha256'),('nominal_geometry.json','nominal_geometry_sha256')]:
            if digest(protocol/name)!=p[key]:raise ValueError('sealed protocol input changed: '+name)
        declaration=json.loads(Path(train['source_declaration']).read_text())
        sources=dict(train['source_sha256'])
        for path in [Path(__file__),*[Path('learning/src/sttw_control')/name for name in
            ['survival_endpoint_batch.py','survival_endpoint_collection.py','survival_endpoint_protocol.py']]]:
            sources[str(path.resolve())]=digest(path)
        atomic_json(output/'plan.json',dict(project='STTW_CONTROL',purpose=__doc__,protocol=str(protocol),
            training_plan_sha256=digest(training/'plan.json'),source_sha256=sources,
            physics_steps=0,optimizer_steps=0,endpoint_authenticated=False,expected_groups=8))
        env,initial=load_environment_contact_only(cfg,declaration['source'],'terminal_contract_v2')
        configs={'physical':cfg,'nominal':dict(copy.deepcopy(cfg),path=nominal['path'])}
        pose=np.asarray(env.physics.helpers.pose(initial.data));paths={}
        for geometry,routes in [('physical',declaration['routes']),('nominal',nominal['routes'])]:
            for route in routes:paths[(geometry,route['name'])]=build_path(route['builder'],pose,configs[geometry])
        groups=partition_cases(cases,{key:tuple(path.s.shape) for key,path in paths.items()})
        with Path(train['model']['actor']).open('rb') as stream:params=jax.tree.map(j.asarray,pickle.load(stream))
        cp=torch.load(train['model']['checkpoint'],map_location='cpu',weights_only=False)
        std=j.asarray(cp['policy']['log_std'].numpy())
        for kind in ('actor','checkpoint'):
            if digest(train['model'][kind])!=train['model'][kind+'_sha256']:raise ValueError('parent changed')
        stack=lambda items:jax.tree.map(lambda *x:j.stack(x),*items)
        cache={};rows=[]
        for group in groups:
            update(stage='trace_'+group['group_id'])
            selected=[cases[i] for i in group['case_indices']];geometry=group['geometry']
            local=copy.copy(env);local.config=evaluation_config(configs[geometry],speed_floor=group['speed_reference_floor_m_s'])
            states=[]
            for c in selected:
                key=(geometry,c['route'],c['speed'],simulation_alpha(c),group['speed_reference_floor_m_s'])
                if key not in cache:cache[key]=local.reset(initial,paths[(geometry,c['route'])],j.float32(c['speed']),j.float32(simulation_alpha(c)))
                states.append(cache[key])
            local.set_log_template(states[0],paths[(geometry,selected[0]['route'])],j.float32(selected[0]['speed']))
            carries=[SurvivalCollectionState(s,jax.random.PRNGKey(p['seed']) if c['rng_fold_in'] is None else
                jax.random.fold_in(jax.random.PRNGKey(p['seed']),c['rng_fold_in']),j.zeros(2),j.bool_(False)) for s,c in zip(states,selected)]
            carry=stack(carries);zero=group['inference_mode']=='zero';n=len(selected)
            one=make_endpoint_collection_step(local,None if zero else make_path_actor(cfg),horizon_s=p['horizon_s'],mode=group['inference_mode'])
            chunk=make_endpoint_chunk(one,steps_per_chunk=50,zero_correction=zero)
            start=float(initial.physical_tick)*local.cc.dt
            pulses=j.asarray([[start+c.get('pulse_onset_task_tick',0)*local.cc.dt,
                start+c.get('pulse_onset_task_tick',0)*local.cc.dt+c.get('pulse_duration_s',0.),
                *(c.get('force_N',0.)*np.asarray(c.get('world_direction',[0.,0.,0.])))] for c in selected],dtype=j.float32)
            abstract=jax.eval_shape(chunk,carry,None if zero else stack([params]*n),j.zeros((n,2)) if zero else j.repeat(std[None],n,axis=0),
                stack([paths[(geometry,c['route'])] for c in selected]),j.asarray([c['speed'] for c in selected]),pulses)
            d,logs=abstract[1]
            assert d['raw_action'].shape==(50,n,2) and logs['actual_xy'].shape==(50,n,4,2)
            assert np.all(np.asarray(carry.state.tick)==0)
            rows.append(dict(group,actual_batch_size=n,abstract_decision_shape=list(d['raw_action'].shape),
                abstract_xy_shape=list(logs['actual_xy'].shape),traced=True,physics_steps=0))
            atomic_json(output/'partial.json',rows);update(completed_groups=len(rows))
        assert len(rows)==8 and sum(r['actual_batch_size'] for r in rows)==488
        for path,sha in sources.items():
            if digest(path)!=sha:raise ValueError('source changed during abstract audit: '+path)
        atomic_json(output/'receipt.json',dict(groups=rows,physics_steps=0,optimizer_steps=0,endpoint_authenticated=False,
            scope='Real full-state batch tracing only, original parent weights in all learned slots; no compiled integration or trained endpoint efficacy.'))
        update(phase='completed',stage='eight_real_batch_layouts_traced')
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.protocol.resolve(),a.output.resolve())
