"""Fixed new-context goal/physical-response evaluation on sealed native code.

Load this orchestration by file in a fresh process. It never trains, silently
restarts, or changes old runs. Physics is available only through execute.
"""
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def file_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def inventory():
    events=[dict(event='zero',onset_s=1.6,force_N=0.)]
    events += [dict(event=f't{t:g}_F{f:+g}',onset_s=t,force_N=f)
               for t,f in itertools.product((1.6,2.6),(-32.,32.))]
    return [dict(case_id=f'{side}70_R2p1_v2p2/a{a:g}/{mode}/{e["event"]}',
                 route=side+'70_R2p1',builder=side+'90_R2',speed=2.2,alpha=a,
                 mode=mode,mode_index=i,duration_s=.1,**e)
            for side,a,(i,mode),e in itertools.product(
                ('left','right'),(0.,1.),enumerate(('H75','current','zero','equilibrium')),events)]



def response_protocol():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/h75_channel_response.json').read_text())


def response_inventory(protocol=None):
    p=response_protocol() if protocol is None else protocol
    validate_response_protocol(p)
    events=[dict(event='zero',onset_s=p['zero_onset_s'],force_N=0.)]
    events += [dict(event=f't{t:g}_F{f:+g}',onset_s=t,force_N=f)
               for t,f in itertools.product(p['onsets_s'],p['forces_N'])]
    rows=[]
    for route,(sa,ta),event in itertools.product(p['routes'],p['mappings'],events):
        mode='H75' if sa==ta else 'hybrid';alpha=sa if sa==ta else 1.
        rows.append(dict(case_id=f'{route["route"]}/a{alpha:g}_{mode}/{event["event"]}',
            **route,alpha=alpha,mode=mode,mode_index=0,speed_alpha=sa,steer_alpha=ta,
            duration_s=p['duration_s'],**event))
    if len(rows)!=p['expected_cases']:raise ValueError('response inventory size mismatch')
    return rows


def validate_response_protocol(p):
    # Bounded authorized contract; descriptive metadata may be enriched freely.
    required=dict(horizon_s=10.,expected_cases=54,maximum_lower_ticks=108000,optimizer_updates=0,
        force_frame='worldY',initial_bank=3,duration_s=.1,goal_radius_m=1.,replanning_radius_m=1.8,
        departure_m=1.,maximum_replans=1,replan_cadence_s=1.,planner_support_m=100.,planner_spacing_m=.02,planner_points=5001)
    for key,value in required.items():
        if p.get(key)!=value:raise ValueError('response contract mismatch '+key)
    if (p.get('forces_N')!=[-64.,-32.,32.,64.] or p.get('onsets_s')!=[1.6,4.] or
        p.get('mappings')!=[[.5,.5],[1.,1.],[1.,.5]] or p.get('zero_onset_s')!=1.6 or
        {(r['route'],r['speed']) for r in p.get('routes',[])}!={('S_left_90_R1p8',2.8),('S_right_90_R1p8',2.8)}):
        raise ValueError('response route/mapping/event contract mismatch')





def switching_protocol():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/h75_channel_switching.json').read_text())


def switching_inventory(p):
    base=dict(p,expected_cases=54,maximum_lower_ticks=108000,forces_N=[-64.,-32.,32.,64.],onsets_s=[1.6,4.])
    validate_response_protocol(base)
    if (p['expected_cases']!=36 or p['maximum_lower_ticks']!=72000 or
        p['forces_N']!=[-64.,64.] or p['onsets_s']!=[1.6] or p['switch_times_s']!=[1.,1.4,1.6] or
        p['destination_pair']!=[.5,.5] or p['capture_task_s']!=1. or
        p['origins']!=[dict(origin='fixed1',speed_alpha=1.,steer_alpha=1.),dict(origin='hybrid',speed_alpha=1.,steer_alpha=.5)]):
        raise ValueError('bounded switching contract mismatch')
    if p['prefix_comparison'].get('absolute_tolerance')!=1e-5 or p['prefix_comparison'].get('relative_tolerance')!=1e-6:
        raise ValueError('switch cached-prefix tolerance contract mismatch')
    events=[dict(event='zero',force_N=0.),*[dict(event=f't1.6_F{f:+g}',force_N=f) for f in p['forces_N']]]
    return [dict(**route,**origin,**event,case_id=f'{route["route"]}/{origin["origin"]}/switch{t:g}/{event["event"]}',
        mode='switch_'+origin['origin'],mode_index=0,alpha=1.,switch_s=t,onset_s=p['onsets_s'][0],duration_s=p['duration_s'])
        for route,origin,t,event in itertools.product(p['routes'],p['origins'],p['switch_times_s'],events)]




def online_inventory(p):
    base=dict(p,expected_cases=54,maximum_lower_ticks=108000,mappings=[[.5,.5],[1.,1.],[1.,.5]])
    rows=response_inventory(base)
    if p['expected_cases']!=18 or p['maximum_lower_ticks']!=36000 or p['rule']!=dict(preview_m=1.5,engage_accel_m_s2=3.,release_accel_m_s2=2.7,active_alpha=.5):
        raise ValueError('bounded online scalar contract mismatch')
    return [dict(c,mode='online_scalar',case_id=c['case_id'].replace('/a1_H75/','/online_scalar/')) for c in rows if c['alpha']==1. and c['mode']=='H75']


def prepare_online(source,root,protocol):
    return prepare_switching(source,root,protocol,online=True)



def samestate_inventory(p):
    if (p['expected_cases']!=72 or p['horizon_s']!=10. or p['new_horizon_s']!=9. or p['maximum_lower_ticks']!=129600 or
        p['optimizer_updates']!=0 or p['source_task_s']!=1. or p['candidate_alphas']!=[0.,.5,1.] or
        p['preparation_s']!=[0.,.2,.6] or p['forces_N']!=[-64.,64.] or p['duration_s']!=.1 or
        p['primary_after_pulse_end_s']!=3. or [r['source_index'] for r in p['source_rows']]!=[0,9,18,27]):
        raise ValueError('bounded same-state72 contract mismatch')
    rows=[]
    for src,a,prep,force in itertools.product(p['source_rows'],p['candidate_alphas'],p['preparation_s'],p['forces_N']):
        onset=p['source_task_s']+prep
        rows.append(dict(route=src['route'],speed=2.8,source_index=src['source_index'],source_origin=src['origin'],source_fingerprint=src['fingerprint'],
            case_id=f'source{src["source_index"]}/alpha{a:g}/prep{prep:g}/F{force:+g}',mode='samestate_scalar',mode_index=0,
            alpha=1.,candidate_alpha=a,query_alpha=a,preparation_s=prep,onset_s=onset,duration_s=p['duration_s'],force_N=force,
            primary_window_start_s=p['source_task_s'],primary_window_end_s=onset+p['duration_s']+p['primary_after_pulse_end_s']))
    return rows



def gather_source_carry(saved,indices):
    """Copy existing complete state, without reconstructing/resetting any leaf."""
    import jax
    import jax.numpy as j
    import numpy as np
    result=jax.tree.map(lambda x:j.asarray(x)[j.asarray(indices)],saved)
    aa=jax.tree.leaves(jax.device_get(result));bb=jax.tree.leaves(saved)
    if len(aa)!=len(bb) or any(a.dtype!=np.asarray(b).dtype or any(not np.array_equal(a[i],np.asarray(b)[index],equal_nan=True) for i,index in enumerate(indices)) for a,b in zip(aa,bb)):
        raise ValueError('complete source-copy leaves changed')
    return result

def prepare_samestate(source,root,protocol):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_switching_continuation' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed same-state source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=36:raise ValueError('completed36 source required')
    identity={}
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('same-state archived source changed')
        identity[str(archived)]=sha
    for f,sha in p['input_sha256'].items():
        if digest(f)!=sha:raise ValueError('same-state source input changed '+f)
        identity[f]=sha
    proposal=json.loads(Path(protocol).read_text());rows=samestate_inventory(proposal)
    original=json.loads((source/'inventory.json').read_text())
    for c in proposal['source_rows']:
        old=original[c['source_index']]
        if old['route']!=c['route'] or old['origin']!=c['origin'] or old['force_N']!=0. or old['switch_s']!=1.:
            raise ValueError('fixed metadata source selection mismatch')
    import numpy as np
    with np.load(source/'evaluation/task1_host_state.npz') as host:
        indices=[c['source_index'] for c in proposal['source_rows']]
        if np.any(host['replanned'][indices]) or np.any(host['ever_arrived'][indices]):raise ValueError('source original-path/unarrived contract differs')
    files=['plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/task1_complete_carry.pkl','evaluation/task1_host_state.npz','evaluation/reused_source_prefix.npz']
    files += ['evaluation/'+r+'_path.npz' for r in sorted({c['route'] for c in rows})]
    for name in files:
        f=source/name;identity[str(f)]=digest(f)
    prepare_switching(p['stay_source'],root,protocol,samestate_source=dict(source=str(source),identity=identity))

def confirmation_inventory(p):
    base=dict(p,expected_cases=18,maximum_lower_ticks=36000,forces_N=[-64.,-32.,32.,64.],onsets_s=[1.6,4.],zero_onset_s=1.6)
    online_inventory(base)
    if (p['expected_cases']!=100 or p['maximum_lower_ticks']!=200000 or p['bank_indices']!=[0,1] or
        p['forces_N']!=[-64.,64.] or p['onsets_s']!=[2.2,3.4] or p['zero_onset_s']!=2.2 or
        p['arms']!=['online_scalar','fixed0','fixed_half','fixed1','zero_with_existing_speed_floor'] or p['existing_speed_floor_m_s']!=1.5):
        raise ValueError('bounded confirmation100 contract mismatch')
    events=[dict(event='zero',force_N=0.,onset_s=p['zero_onset_s'])]
    events += [dict(event=f't{t:g}_F{f:+g}',force_N=f,onset_s=t) for t,f in itertools.product(p['onsets_s'],p['forces_N'])]
    return [dict(**route,**event,bank_index=b,initial_fingerprint=p['bank_state_fingerprints'][str(b)],
        path_key=f'bank{b}_'+route['route'],case_id=f'bank{b}/{route["route"]}/{arm}/{event["event"]}',
        mode=arm,mode_index=i,alpha=1.,query_alpha=(0. if i==1 else .5 if i==2 else 1.),duration_s=p['duration_s'])
        for b,route,event,(i,arm) in itertools.product(p['bank_indices'],p['routes'],events,enumerate(p['arms']))]


def snapshot_fingerprint(snapshot):
    import jax
    import numpy as np
    sha=hashlib.sha256()
    for path,value in jax.tree_util.tree_flatten_with_path(snapshot)[0]:
        a=np.asarray(value);sha.update(jax.tree_util.keystr(path).encode());sha.update(str(a.dtype).encode());sha.update(a.tobytes())
    return sha.hexdigest()


def existing_floor_latent(nominal,config,floor=1.5):
    """Exact existing profile7 zero-neural support mapping, native scales only."""
    import jax.numpy as j
    desired=j.maximum(0.,floor-nominal[0])
    scale=config['residual_interface']['speed_positive_scale_m_s']
    bounded=j.clip(desired/scale,-1.+1e-6,1.-1e-6)
    latent=j.stack([j.arctanh(bounded),j.asarray(0.)])
    realized=bounded*scale
    return latent,dict(support_floor=j.asarray(floor),support_requested_offset=desired,
        support_realized_target_offset=realized,support_active=realized>0.)


def case_path_key(case):
    return case.get('path_key',case['route'])



def load_confirmation_environment(cfg,source):
    """Native constructor/reset inputs, one bank read; no physical integration."""
    import jax
    import jax.numpy as j
    import numpy as np
    from dataclasses import asdict
    from sttw_control.path_command_env import PathCommandEnv
    from sttw_control.frozen_local_lower import FrozenLocalLower
    from sttw_control.lower_command import SCALES
    from sttw_control.path_command_training import config_hash
    candidate=source['candidate']
    if digest(candidate['config'])!=candidate['config_file_sha256']:raise ValueError('lower config changed')
    np.testing.assert_array_equal(np.asarray(SCALES),np.asarray(candidate['normalizer']['scales'],np.float32))
    for name in ('actor','checkpoint'):
        if digest(candidate[name])!=candidate[name+'_sha256']:raise ValueError('lower model identity changed')
    with Path(candidate['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
    spec=json.loads((Path(source['upper_run'])/'alpha0/frozen_config_stage52.json').read_text())
    env=PathCommandEnv(cfg,spec,FrozenLocalLower(params),'terminal_contract_v2')
    for key,live in [('model',env.physics.bundle.identity),('controller',asdict(env.cc)),('actuator',asdict(env.ac))]:
        if config_hash(live)!=config_hash(source[key]):raise ValueError('native physical identity mismatch '+key)
    if digest(source['prepared_bank'])!=source['prepared_bank_sha256']:raise ValueError('bank identity changed')
    with Path(source['prepared_bank']).open('rb') as f:bank=pickle.load(f)
    return env,bank

def prepare_confirmation(source,root,protocol):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_online_scalar' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed completedonline18 source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=18:raise ValueError('completedonline18 source required')
    identity=dict(p['input_sha256'])
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('online archived source mismatch')
        identity[str(archived)]=sha
    for f,sha in identity.items():
        if digest(f)!=sha:raise ValueError('online source input mismatch '+f)
    for name in ('plan.json','seal.json','evaluation/status.json'):
        f=source/name;identity[str(f)]=digest(f)
    confirmation_inventory(json.loads(Path(protocol).read_text()))
    # Preserve the authenticated native ancestor chain, without reusing trajectories.
    prepare_switching(p['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity))

def continuation_inventory(p):
    original=dict(p,maximum_lower_ticks=72000)
    rows=switching_inventory(original)
    if (p['maximum_lower_ticks']!=64800 or p['total_campaign_lower_ticks']!=79200 or
        p['consumed_source_lower_ticks']!=14400 or p['continuation_task_s']!=1. or p['new_horizon_s']!=9. or
        p['prefix_comparison'].get('field_absolute_tolerance')!={'phi_dot':1e-4} or not p['prefix_comparison'].get('floating_descriptive_only')):
        raise ValueError('bounded continuation amendment mismatch')
    return rows


def prefix_lane_divergence(logs,cases):
    import numpy as np
    groups={};maxima={}
    for i,c in enumerate(cases):groups.setdefault((c['route'],c['origin']),[]).append(i)
    for indices in groups.values():
        for name,value in logs.items():
            a=np.asarray(value)[:,indices];b=np.asarray(value)[:,indices[0],None]
            if np.issubdtype(a.dtype,np.number):maxima[name]=max(maxima.get(name,0.),float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0.)))
    return maxima


def audit_scheduled_pairs(logs,cases):
    import numpy as np
    for i,c in enumerate(cases):
        active=np.asarray(logs['active_tick'])[:,i]
        upper_tick=(np.rint(np.asarray(logs['time'])[:,i]/.005).astype(int)//4)*4
        switched=upper_tick>=round(c['switch_s']/.005)
        pair=np.where(switched[:,None],np.array([.5,.5]),np.array([c['speed_alpha'],c['steer_alpha']]))
        if not np.array_equal(logs['channel_alpha_pair'][active,i],pair[active]) or not np.array_equal(logs['switch_active'][active,i],switched[active]):
            raise ValueError('scheduled channel pair/timing mismatch')
        if 'applied_force_impulse' in logs:
            impulse=np.asarray(logs['applied_force_impulse'])[:,i]
            force=c['force_N'];time=np.asarray(logs['time'])[:,i]
            if np.any(impulse[:,[0,2]]!=0.) or (force==0. and np.any(impulse!=0.)):
                raise ValueError('worldY pulse vector mismatch')
            if force:
                nonzero=np.any(impulse!=0.,axis=-1)
                # Native substep clock may straddle one lower boundary.
                inside=(time>=c['onset_s']-.005-1e-6)&(time<c['onset_s']+c['duration_s']+.005+1e-6)
                if np.any(nonzero&~inside) or np.any(impulse[:,1]*force<0.) or np.any(np.abs(impulse[:,1])>abs(force)*.005+1e-5):
                    raise ValueError('pulse sign/timing/impulse mismatch')
    return dict(actual_pairs_and_switch_timing_verified=True,worldY_force_vector_verified='applied_force_impulse' in logs)


def prepare_continuation(source,root,protocol,*,charged_source=None):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_channel_switching':raise ValueError('failed switching source required')
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('continuation source seal mismatch')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='error' or status.get('active_lower_ticks')!=7200 or status.get('allocated_lower_slots')!=7200:raise ValueError('exact7200 consumed source required')
    identity=dict(p['input_sha256'])
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('continuation archived source mismatch')
        identity[str(archived)]=sha
    for f,sha in identity.items():
        if digest(f)!=sha:raise ValueError('continuation source input mismatch '+f)
    for name in ('plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/task1_complete_carry.pkl','evaluation/task1_host_state.npz','evaluation/chunk_00.npz'):
        f=source/name;identity[str(f)]=digest(f)
    if charged_source is None:raise ValueError('prior failed continuation accounting source required')
    charged_source=Path(charged_source).resolve();charged=json.loads((charged_source/'plan.json').read_text())
    charged_status=json.loads((charged_source/'evaluation/status.json').read_text())
    if (digest(charged_source/'plan.json')!=json.loads((charged_source/'seal.json').read_text())['plan_sha256'] or
        charged.get('mode')!='h75_switching_continuation' or Path(charged['continuation_source']['source']).resolve()!=source or
        charged_status.get('phase')!='error' or charged_status.get('active_lower_ticks')!=7200 or charged_status.get('allocated_lower_slots')!=7200):
        raise ValueError('charged failed continuation identity/count mismatch')
    for f,sha in charged['input_sha256'].items():
        if digest(f)!=sha:
            # The explicit amended protocol supersedes this one configuration
            # input. Authenticate its old bytes from the sealed embedded JSON.
            embedded=(json.dumps(charged['protocol'],indent=2)+'\n').encode()
            if Path(f).resolve()!=Path(protocol).resolve() or hashlib.sha256(embedded).hexdigest()!=sha:
                raise ValueError('charged continuation input drift '+f)
    for f,sha in charged['source_sha256'].items():
        archived=charged_source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('charged continuation archived source drift')
        identity[str(archived)]=sha
    for name in ('plan.json','seal.json','evaluation/status.json','evaluation/chunk_01.npz'):
        f=charged_source/name;identity[str(f)]=digest(f)
    proposed=json.loads(Path(protocol).read_text());rows=continuation_inventory(proposed)
    if rows!=json.loads((source/'inventory.json').read_text()):raise ValueError('original36 continuation inventory differs')
    import numpy as np
    with np.load(source/'evaluation/chunk_00.npz') as raw:
        logs={k:raw[k] for k in raw.files}
    audit_scheduled_pairs(logs,rows)
    if logs['active_tick'].shape!=(200,36) or not np.all(logs['active_tick']) or np.any(logs['switch_active']) or np.any(logs['applied_force_impulse']):
        raise ValueError('source prefix active/switch/force contract mismatch')
    for name in ('physical_failure','contact_failure','nonfinite','policy_fault','lower_fault','continuation_geometry_fault'):
        if np.any(logs[name]):raise ValueError('source prefix fault/physical event '+name)
    prepare_switching(p['stay_source'],root,protocol,continuation_source=dict(source=str(source),identity=identity,prefix_lane_maximum_absolute_difference=prefix_lane_divergence(logs,rows),charged_source=str(charged_source)))


def switch_mapping(task_tick,origin_speed,origin_steer,switch_tick,destination):
    """A query mapping change only; never edits any closed-loop state leaf."""
    import jax.numpy as j
    active=task_tick>=switch_tick
    return j.where(active,destination[0],origin_speed),j.where(active,destination[1],origin_steer),active


def assert_shared_switch_carry(carry,cases,*,lanes_per_group=9):
    import jax
    import numpy as np
    leaves=jax.tree.leaves(jax.device_get(carry));groups={}
    for i,c in enumerate(cases):groups.setdefault((c['route'],c['origin']),[]).append(i)
    if len(groups)!=4 or any(len(v)!=lanes_per_group for v in groups.values()):raise ValueError('switch capture groups mismatch')
    for indices in groups.values():
        for leaf in leaves:
            if any(not np.array_equal(leaf[indices[0]],leaf[i],equal_nan=True) for i in indices[1:]):
                raise ValueError('complete task1.0 carry differs within origin group')
    return dict(groups=4,lanes_per_group=lanes_per_group,leaf_count=len(leaves),all_leaves_exact=True,task_s=1.)


def audit_stay_prefix(logs,cases,cached,protocol,*,own_switch=False):
    import numpy as np
    matched={};maxima={};counts=[];exceeded={}
    contract=protocol['prefix_comparison']
    atol=contract.get('absolute_tolerance',1e-5);rtol=contract.get('relative_tolerance',1e-6)
    for i,c in enumerate(cases):
        key=(c['route'],c['origin'],c['force_N'])
        record=next(r for r in cached if (r['route'],r['origin'],r['force_N'])==key)
        count=round(c['switch_s']/.005) if own_switch else len(logs['time'])
        if len(logs['time'])<count:raise ValueError('insufficient own-switch prefix')
        counts.append(count)
        with np.load(record['raw_file'],allow_pickle=False) as old:
            for name in contract['fields']:
                a=np.asarray(logs[name])[:count,i];b=old[name][:count]
                floating=np.issubdtype(a.dtype,np.floating)
                difference=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0.)) if a.shape==b.shape else float('inf')
                maxima[name]=max(maxima.get(name,0.),difference)
                equal=(np.allclose(a,b,atol=contract.get('field_absolute_tolerance',{}).get(name,atol),rtol=rtol,equal_nan=False) if floating else np.array_equal(a,b)) if a.shape==b.shape else False
                if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    raise ValueError('nonfinite stay-origin prefix '+name)
                if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('stay prefix dtype/shape mismatch '+name)
                if not equal:
                    exceeded[name]=exceeded.get(name,0)+1
                    if not (floating and contract.get('floating_descriptive_only',False)):
                        raise ValueError('stay-origin prefix mismatch '+str(key)+'/'+name)
        matched[key]=record['raw_file']
    return dict(matched_stay_cases=len(matched),new_lanes=len(cases),prefix_lower_ticks=counts,
        own_switch=own_switch,all_fields_within_tolerance=not bool(exceeded),floating_descriptive_only=contract.get('floating_descriptive_only',False),exceeded_lane_field_counts=exceeded,discrete_exact=True,
        absolute_tolerance=atol,relative_tolerance=rtol,field_absolute_tolerance=contract.get('field_absolute_tolerance',{}),maximum_absolute_difference=maxima)


def prepare_switching(source,root,protocol,*,continuation_source=None,online=False,confirmation_source=None,samestate_source=None):
    source=Path(source).resolve();stay=json.loads((source/'plan.json').read_text())
    if stay.get('mode')!='h75_channel_response':raise ValueError('sealed54 response source required')
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('stay source seal mismatch')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=54:raise ValueError('complete54 stay source required')
    identity=dict(stay['input_sha256'])
    for f,sha in stay['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('stay archived module mismatch')
        identity[str(archived)]=sha
    for f,sha in identity.items():
        if digest(f)!=sha:raise ValueError('stay input mismatch '+f)
    records=json.loads((source/'evaluation/case_index.json').read_text());selected=[]
    for r in records:
        if online:
            if r['mode']!='H75' or r['alpha'] not in (.5,1.):continue
        elif r['alpha']!=1. or r['mode'] not in ('H75','hybrid') or r['force_N'] not in (0.,-64.,64.) or r['onset_s']!=1.6:continue
        r=dict(r,origin=('fixed'+format(r['alpha'],'g')) if online else 'fixed1' if r['mode']=='H75' else 'hybrid')
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('stay raw mismatch')
        selected.append(r);identity[r['raw_file']]=r['raw_sha256']
    if online:
        if len(selected)!=36 or len({(r['route'],r['alpha'],r['event']) for r in selected})!=36:raise ValueError('exact36 fixed-alpha cache required')
    elif len(selected)!=12 or len({(r['route'],r['origin'],r['force_N']) for r in selected})!=12:raise ValueError('exact12 matching stays required')
    for name in ('plan.json','seal.json','evaluation/status.json','evaluation/case_index.json'):
        f=source/name;identity[str(f)]=digest(f)
    prepare_response(stay['source'],root,protocol,stay_source=dict(source=str(source),plan=stay,records=[] if confirmation_source is not None or samestate_source is not None else selected,identity=identity,online=online),continuation_source=continuation_source,confirmation_source=confirmation_source,samestate_source=samestate_source)

def extend_straight_support(path,target_points,spacing_m):
    """Keep all original geometry samples; extend only a straight terminal ray."""
    import jax.numpy as j
    import numpy as np
    extra=target_points-len(path.s)
    if extra<0:raise ValueError('source path exceeds planner support')
    if not extra:return path
    if not np.allclose(np.asarray(path.curvature)[-2:],0.,atol=1e-8):
        raise ValueError('only straight source support may extend')
    tail=j.arange(1,extra+1)*spacing_m
    return path.replace(s=j.concatenate([path.s,path.s[-1]+tail]),
        xy=j.concatenate([path.xy,path.xy[-1]+tail[:,None]*j.array([j.cos(path.heading[-1]),j.sin(path.heading[-1])])]),
        heading=j.concatenate([path.heading,j.full(extra,path.heading[-1])]),
        curvature=j.concatenate([path.curvature,j.zeros(extra)]))

def force_pulses(cases,physical_start):
    """Archived teleop pulse is a world vector, never rotated with route/body."""
    return [[physical_start+c['onset_s'],physical_start+c['onset_s']+c['duration_s'],
             0.,c['force_N'],0.] for c in cases]

def channel_queries(obs,apply,speed_alpha,steer_alpha):
    """Same observation except native alpha336; one composed publication."""
    import jax.numpy as j
    half=apply(obs.at[336].set(.5));full=apply(obs.at[336].set(1.))
    return j.stack([j.where(speed_alpha==.5,half[0],full[0]),
                    j.where(steer_alpha==.5,half[1],full[1])]),half,full


def prepare_response(source,root,protocol,*,stay_source=None,continuation_source=None,confirmation_source=None,samestate_source=None):
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve();storage_guard(root)
    old=json.loads((source/'plan.json').read_text())
    def authenticate(directory,plan,expected_cases):
        if digest(directory/'plan.json')!=json.loads((directory/'seal.json').read_text())['plan_sha256']:
            raise ValueError('source seal changed')
        if json.loads((directory/'evaluation/status.json').read_text()).get('phase')!='completed' or json.loads((directory/'evaluation/status.json').read_text()).get('completed_cases')!=expected_cases:
            raise ValueError('completed source required')
        for f,sha in {**plan['input_sha256'],**plan['native_source_sha256']}.items():
            if digest(f)!=sha:raise ValueError('source identity changed '+f)
        for f,sha in plan['runner_sha256'].items():
            archived=directory/'source'/Path(f).name
            if digest(archived)!=sha:raise ValueError('archived runner changed '+str(archived))
    authenticate(source,old,8)
    if old.get('mode')!='native_channel_factorial_v1':raise ValueError('native channel factorial source required')
    cached=Path(old['cached_source']);native=json.loads((cached/'plan.json').read_text());authenticate(cached,native,16)
    if native.get('mode')!='native_capability_v1':raise ValueError('native S capability required')
    proposal=json.loads(protocol.read_text())
    online=stay_source is not None and stay_source.get('online',False)
    if samestate_source is not None:samestate_inventory(proposal)
    elif confirmation_source is not None:confirmation_inventory(proposal)
    elif online:online_inventory(proposal)
    elif stay_source is None:validate_response_protocol(proposal)
    elif continuation_source is None:switching_inventory(proposal)
    else:continuation_inventory(proposal)
    if old['config']!=native['config'] or old['initial_source']!=native['initial_source'] or old['h75']!=native['h75']:
        raise ValueError('native source/config/initial/model mismatch')
    if old['initial_source']['bank_index']!=3:raise ValueError('native bank3 required')
    files={**old['input_sha256'],**native['input_sha256'],**old['native_source_sha256']}
    for directory in (source,cached):
        for name in ('plan.json','seal.json','evaluation/status.json','evaluation/initial_carry.pkl'):
            f=directory/name;files[str(f)]=digest(f)
    for c in native['contexts']:
        f=cached/'evaluation'/(c['route']+'_path.npz');files[str(f)]=digest(f)
    for owner in (old['h75'],old['initial_source']['candidate']):
        for key in ('actor','checkpoint'):
            if digest(owner[key])!=owner[key+'_sha256']:raise ValueError('model changed')
            files[owner[key]]=owner[key+'_sha256']
    folder=Path(__file__).resolve().parent
    modules=[folder/n for n in ('native_goal_run.py','native_goal_continuation.py','native_equilibrium_rule.py','contact_only_environment.py','goal_replanning.py','_vendor/pythonrobotics/dubins.py','_vendor/pythonrobotics/angle.py')]
    modules.append(Path('learning/cli/run_native_goal_confirmation.py').resolve())
    if online:modules.append(folder/'geometric_preference_schedule.py')
    if stay_source is not None:
        if any(stay_source['plan'][k]!=v for k,v in [('config',old['config']),('initial_source',old['initial_source']),('models',{'H75':old['h75']})]):raise ValueError('stay source full interface identity differs')
        if stay_source['plan']['native_runtime_root']!=old['native_runtime_root'] or stay_source['plan']['native_expected_sources']!=old['native_source_sha256']:
            raise ValueError('stay native runtime identity differs')
        files.update(stay_source['identity'])
    if continuation_source is not None:files.update(continuation_source['identity'])
    if confirmation_source is not None:files.update(confirmation_source['identity'])
    if samestate_source is not None:files.update(samestate_source['identity'])
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in modules:(root/'source'/f.name).write_bytes(f.read_bytes())
    if stay_source is not None:
        save(root/'cached_stay_records.json',stay_source['records']);files[str(root/'cached_stay_records.json')]=digest(root/'cached_stay_records.json')
    save(root/'inventory.json',samestate_inventory(proposal) if samestate_source is not None else confirmation_inventory(proposal) if confirmation_source is not None else online_inventory(proposal) if online else continuation_inventory(proposal) if continuation_source is not None else response_inventory(proposal) if stay_source is None else switching_inventory(proposal));save(root/'effective_config.json',old['config'])
    files.update({str(f.resolve()):digest(f) for f in (protocol,root/'inventory.json',root/'effective_config.json')})
    plan=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode='h75_samestate_tolerance72' if samestate_source is not None else 'h75_online_confirmation100' if confirmation_source is not None else 'h75_online_scalar' if online else 'h75_switching_continuation' if continuation_source is not None else 'h75_channel_response' if stay_source is None else 'h75_channel_switching',
        source=str(source),cached_source=str(cached),native_runtime_root=old['native_runtime_root'],
        native_expected_sources=old['native_source_sha256'],initial_source=old['initial_source'],
        models={'H75':old['h75']},config=old['config'],contexts=native['contexts'],cases=proposal['expected_cases'],horizon_s=proposal['horizon_s'],goal_radius_m=proposal['goal_radius_m'],
        chunks=int(proposal.get('new_horizon_s',proposal['horizon_s'])/proposal['replan_cadence_s']),steps_per_chunk=50,maximum_active_lower_ticks=proposal['maximum_lower_ticks'],maximum_allocated_lower_slots=proposal['maximum_lower_ticks'],
        optimizer_updates=0,protocol=proposal,samestate_source=samestate_source,confirmation_source=confirmation_source,continuation_source=continuation_source,stay_source=None if stay_source is None else stay_source['source'],data_role=proposal['data_role'],question=proposal['question'],
        replanning=dict(period_s=proposal['replan_cadence_s'],radius_m=proposal['replanning_radius_m'],departure_m=proposal['departure_m'],maximum_per_case=proposal['maximum_replans'],information=proposal['replanning'],ordering=proposal['ordering']),
        scoring=proposal['scoring'],stop=proposal['stop_rule'],normalization=proposal['normalization'],initialization=proposal.get('initialization','Authenticated original native prepared bank/reset'),
        input_sha256=files,source_sha256={str(f.resolve()):digest(f) for f in modules})
    save(root/'plan.json',plan);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def storage_guard(root):
    mount=Path('/media/qy/6A7E3DB07E3D75C3')
    if not os.path.ismount(mount) or not Path(root).resolve().is_relative_to(mount):
        raise ValueError('large evaluation outputs require the mounted data volume')
    stat=os.statvfs('/home/qy/STTW_CONTROL')
    if stat.f_bavail*stat.f_frsize<50*1024**3:
        raise ValueError('system volume must retain at least50GiB available')


def prepare(source,root):
    source=Path(source).resolve();root=Path(root).resolve();storage_guard(root)
    previous=json.loads((source/'plan.json').read_text())
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('native zero source plan identity changed')
    for f,sha in previous['input_sha256'].items():
        if digest(f)!=sha:raise ValueError('native source metadata changed '+f)
    status=json.loads((source/'evaluation/status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=8:
        raise ValueError('completed native zero source required')
    native_source=Path(previous['source'])
    original=json.loads((native_source/'plan.json').read_text())
    registry=Path('learning/configs/frozen_middle_registry.json').resolve()
    h75=json.loads(registry.read_text())['records']['STTW_MIDDLE_FULLTAIL_PPO75_U77']
    models=dict(H75={k:h75[k] for k in ('actor','actor_sha256','checkpoint','checkpoint_sha256')},
                current=original['current_model'])
    files=[source/'plan.json',source/'seal.json',source/'evaluation/status.json',native_source/'plan.json',registry]
    for model in models.values():
        for key in ('actor','checkpoint'):
            if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen model changed '+key)
            files.append(Path(model[key]))
    cfg=copy.deepcopy(previous['config'])
    cfg['training_future_phase_C']['episode_s']=20.
    cfg['path'].update(turn_radius_m=2.1,turn_angle_rad=math.radians(70))
    end=cfg['path']['straight_prefix_m']+cfg['path']['curvature_transition_length_m']+math.radians(70)*2.1
    cfg['path']['straight_tail_m']=80.-end
    if cfg['path_reward'].get('tolerance',{}).get('weight',0):raise ValueError('old analytic shaping must remain off')
    if cfg['residual_interface']['speed_reference_bounds_m_s']!=[1.5,3.5]:raise ValueError('native permission mismatch')
    own=Path(__file__).resolve();folder=own.parent
    modules=[own,folder/'native_goal_continuation.py',folder/'native_equilibrium_rule.py',folder/'contact_only_environment.py',
             folder/'goal_replanning.py',folder/'_vendor/pythonrobotics/dubins.py',folder/'_vendor/pythonrobotics/angle.py',
             Path('learning/cli/run_native_goal_confirmation.py').resolve()]
    files += [folder/'_vendor/pythonrobotics/LICENSE',folder/'_vendor/pythonrobotics/UPSTREAM.json']
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in modules:(root/'source'/f.name).write_bytes(f.read_bytes())
    save(root/'inventory.json',inventory());save(root/'effective_config.json',cfg)
    files += [root/'inventory.json',root/'effective_config.json']
    plan=dict(project='STTW_CONTROL',experiment_id='STTW-NATIVE-GOAL-CONFIRMATION-V1',owner='/root',
        question='Do frozen preference policies retain useful task or physical-response gains beyond zero and a fixed rule under a common goal-continuation contract?',
        source=str(source),native_source=str(native_source),native_runtime_root=previous['native_runtime_root'],
        native_expected_sources=previous['native_source_sha256'],initial_source=previous['initial_source'],models=models,
        config=cfg,cases=80,horizon_s=20.,goal_radius_m=1.,steps_per_chunk=50,chunks=20,
        maximum_active_lower_ticks=320000,maximum_allocated_lower_slots=320000,optimizer_updates=0,
        initialization='weights_only inference; original complete prepared bank snapshot plus native task reset, no synthetic physics state',
        normalization='Original native frame/context scales; archived Actor architecture, no learned observation normalizer',
        data_role='Prospective fixed related DEV contexts, shared complete ancestor; not independent training repeats or final TEST',
        rule=dict(budget_rad=cfg['path_reward']['working_roll_start_rad'],grid_points=65),
        replanning=dict(period_s=1.,radius_m=1.8,departure_m=1.,maximum_per_case=1,
            information='Own current geometry at ordinary boundaries; no event schedule, force or pulse-end input'),
        domain_contract='Native follower command50Hz; readonly geometry validity200Hz. Same mask on all nonzero targets. Reentry after16 complete clean upper blocks. Zero targets use original filters; no history/offset reset or extra speed support.',
        scoring='Original worldgoal1m before contact/fault at10/20s; whole-window contact separately. Original-route errors, original requested speed/time and reference cost separately. Current-path V3 is diagnostic after replanning.',
        stop='Fixed80 only; contact/fault/20s. Errors preserve raw partials and cost, no implicit restart, force/radius/time/threshold sweep or model adoption. All-pass is uninformative for binary response, not permission to escalate.',
        input_sha256={str(f.resolve()):digest(f) for f in files},source_sha256={str(f.resolve()):digest(f) for f in modules})
    save(root/'plan.json',plan);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def verify(root):
    root=Path(root).resolve();storage_guard(root)
    p=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('changed plan')
    for f,sha in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(f)!=sha:raise ValueError('changed sealed source/input '+f)
    if json.loads((root/'inventory.json').read_text())!=(samestate_inventory(p['protocol']) if p.get('mode')=='h75_samestate_tolerance72' else confirmation_inventory(p['protocol']) if p.get('mode')=='h75_online_confirmation100' else online_inventory(p['protocol']) if p.get('mode')=='h75_online_scalar' else continuation_inventory(p['protocol']) if p.get('mode')=='h75_switching_continuation' else switching_inventory(p['protocol']) if p.get('mode')=='h75_channel_switching' else response_inventory(p['protocol']) if p.get('mode')=='h75_channel_response' else inventory()):raise ValueError('fixed inventory changed')
    return p


def load(root):
    root=Path(root).resolve();p=verify(root);native=Path(p['native_runtime_root']).resolve()
    for name,module in tuple(sys.modules.items()):
        if name=='sttw_control' or name.startswith('sttw_control.'):
            file=getattr(module,'__file__',None)
            if file and not Path(file).resolve().is_relative_to(native):raise ValueError('modern control package already imported')
    sys.path.insert(0,str(native))
    import jax
    import jax.numpy as j
    import numpy as np
    import sttw_control.geometric_path as geometry
    from sttw_control.path_reference import pursuit
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.direct_command_policy import q_ratio
    from sttw_control.path_command_reward import configured_cost_keys,failure_reward
    from sttw_control.runtime import configure_compilation_cache
    folder=Path(__file__).resolve().parent
    adapter=file_module('_sttw_goal_continuation',folder/'native_goal_continuation.py')
    contact=file_module('_sttw_goal_contact',folder/'contact_only_environment.py')
    rule=file_module('_sttw_goal_rule',folder/'native_equilibrium_rule.py')
    # Only planner/vendor modules live in this separate namespace. Bind their
    # Path dependency to the actual native type; do not extend sttw_control.
    from types import ModuleType
    package=ModuleType('_sttw_planner_only');package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    sys.modules[package.__name__+'.geometric_path']=geometry
    planner=file_module(package.__name__+'.goal_replanning',folder/'goal_replanning.py')
    configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
    cfg=p['config'];confirmation=p.get('mode')=='h75_online_confirmation100'
    if confirmation:
        env,prepared=load_confirmation_environment(cfg,p['initial_source'])
        snapshot=jax.tree.map(lambda x:j.asarray(x[p['protocol']['bank_indices'][0]]),prepared)
    else:env,snapshot=load_environment(cfg,p['initial_source'],'terminal_contract_v2')
    env=adapter.NativeGoalContinuation(env,contact_only_physics=contact.ContactOnlyPhysics,
                                        project=geometry.project,pursuit=pursuit)
    cases=json.loads((root/'inventory.json').read_text());pose=np.asarray(env.physics.helpers.pose(snapshot.data))
    samestate=p.get('mode')=='h75_samestate_tolerance72'
    confirmation=p.get('mode')=='h75_online_confirmation100'
    online=p.get('mode') in ('h75_online_scalar','h75_online_confirmation100')
    branch=p.get('mode')=='h75_switching_continuation'
    switching=p.get('mode') in ('h75_channel_switching','h75_switching_continuation')
    response=p.get('mode') in ('h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar','h75_online_confirmation100','h75_samestate_tolerance72')
    if samestate:
        bank={}
        for route in sorted({c['route'] for c in cases}):
            with np.load(Path(p['samestate_source']['source'])/'evaluation'/(route+'_path.npz')) as rawpath:
                bank[route]=geometry.Path(*(j.asarray(rawpath[k]) for k in ('s','xy','heading','curvature','turn_end','goal')))
    elif confirmation:
        snapshots={b:jax.tree.map(lambda x:j.asarray(x[b]),prepared) for b in p['protocol']['bank_indices']}
        for b,snap in snapshots.items():
            if snapshot_fingerprint(snap)!=p['protocol']['bank_state_fingerprints'][str(b)]:raise ValueError('new prepared snapshot fingerprint differs')
            if bool(snap.failed):raise ValueError('new prepared snapshot failed')
        from sttw_control.path_command_scenarios import _integrate_route
        bank={}
        for b,snap in snapshots.items():
            ownpose=np.asarray(env.physics.helpers.pose(snap.data))
            for c in p['contexts']:
                if c['route'] not in {r['route'] for r in cases}:continue
                path=_integrate_route(ownpose,cfg['path'],2,c['radius_m'],np.deg2rad(c['angle_deg']),c['speed'],c['direction'],0.,0.).path
                bank[f'bank{b}_'+c['route']]=extend_straight_support(path,p['protocol']['planner_points'],p['protocol']['planner_spacing_m'])
    elif response:
        from sttw_control.path_command_scenarios import _integrate_route
        bank={}
        for c in p['contexts']:
            if c['route'] not in {r['route'] for r in cases}:continue
            path=_integrate_route(pose,cfg['path'],2,c['radius_m'],np.deg2rad(c['angle_deg']),c['speed'],c['direction'],0.,0.).path
            with np.load(Path(p['cached_source'])/'evaluation'/(c['route']+'_path.npz')) as old:
                for field in ('s','xy','heading','curvature','turn_end','goal'):
                    if not np.array_equal(np.asarray(getattr(path,field)),old[field]):raise ValueError('original S path mismatch '+field)
            path=extend_straight_support(path,p['protocol']['planner_points'],p['protocol']['planner_spacing_m'])
            bank[c['route']]=path
    else:bank={c['route']:geometry.build_path(c['builder'],pose,cfg) for c in cases}
    target_points=p['protocol']['planner_points'] if response else 4001
    if any(len(path.s)!=target_points for path in bank.values()):raise ValueError('initial/replanned geometry support shape mismatch')
    stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
    paths=stack([bank[case_path_key(c)] for c in cases]);speeds=j.asarray([c['speed'] for c in cases]);alphas=j.asarray([c['alpha'] for c in cases])
    if samestate:
        with (Path(p['samestate_source']['source'])/'evaluation/task1_complete_carry.pkl').open('rb') as f:saved=pickle.load(f)
        for c in p['protocol']['source_rows']:
            selected=jax.tree.map(lambda x:np.asarray(x)[c['source_index']],saved)
            if snapshot_fingerprint(selected)!=c['fingerprint']:raise ValueError('complete source fingerprint changed')
        initial=gather_source_carry(saved,[c['source_index'] for c in cases])
        aa=jax.tree.leaves(jax.device_get(initial));bb=jax.tree.leaves(saved)
        if any(a.dtype!=b.dtype or any(not np.array_equal(a[i],np.asarray(b)[c['source_index']],equal_nan=True) for i,c in enumerate(cases)) for a,b in zip(aa,bb)):raise ValueError('same-state copy changed source leaves')
        if not bool(j.all((initial.state.tick==200)&(initial.state.alpha==1.))):raise ValueError('source clock/useralpha mismatch')
        initial_identity=dict(all_source_copy_leaves_exact=True,native_leaf_count=len(aa),source_groups=4,copies_per_source=18,source_fingerprints={str(c['source_index']):c['fingerprint'] for c in p['protocol']['source_rows']})
    elif confirmation:
        lane_snapshots=stack([snapshots[c['bank_index']] for c in cases])
        initial=jax.jit(jax.vmap(lambda snap,path,v,a:env.initial(env.reset(snap,path,v,a))))(lane_snapshots,paths,speeds,alphas)
    else:initial=jax.jit(jax.vmap(lambda path,v,a:env.initial(env.reset(snapshot,path,v,a))))(paths,speeds,alphas)
    jax.block_until_ready(initial)
    if response and not confirmation and not samestate:
        cached=Path(p['cached_source'])
        with (cached/'evaluation/initial_carry.pkl').open('rb') as f:archived=pickle.load(f)
        old_cases=json.loads((cached/'plan.json').read_text())['cases']
        indices=[next(i for i,c in enumerate(old_cases) if c['route']==case['route'] and c['speed']==case['speed'] and c['policy']=='H75' and c['alpha']==case['alpha']) for case in cases]
        expected=jax.tree.map(lambda x:np.asarray(x)[indices],archived)
        # Only the documented continuation domain latch is adapted; full native reset identity is exact.
        raw=jax.jit(jax.vmap(lambda path,v,a:env.reset(snapshot,path,v,a)))(paths,speeds,alphas)
        aa,ta=jax.tree.flatten(jax.device_get(raw));bb,tb=jax.tree.flatten(expected)
        if ta!=tb or len(aa)!=len(bb) or any(a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True) for a,b in zip(aa,bb)):
            raise ValueError('cached source native reset mismatch; no compatibility weakening')
        initial_identity=dict(all_native_leaves_exact=True,native_leaf_count=len(aa),dtypes_exact=True,original_paths_exact=True)

    if confirmation:
        groups={}
        for i,c in enumerate(cases):groups.setdefault((c['bank_index'],c['route'],c['event']),[]).append(i)
        leaves=jax.tree.leaves(jax.device_get(initial))
        if len(groups)!=20 or any(len(ids)!=5 for ids in groups.values()):raise ValueError('newbank reset grouping mismatch')
        if any(any(not np.array_equal(x[ids[0]],x[i],equal_nan=True) for i in ids[1:]) for ids in groups.values() for x in leaves):raise ValueError('five-arm complete physical reset differs')
        initial_identity=dict(new_bank_fingerprints=p['protocol']['bank_state_fingerprints'],all_five_arm_reset_leaves_exact=True,native_leaf_count=len(leaves),groups=20,original_paths_anchored_to_own_pose=True,no_old_trajectory_reuse=True)
    if branch:
        with (Path(p['continuation_source']['source'])/'evaluation/task1_complete_carry.pkl').open('rb') as f:saved=pickle.load(f)
        initial=jax.tree.map(j.asarray,saved)
        aa,ta=jax.tree.flatten(jax.device_get(initial));bb,tb=jax.tree.flatten(saved)
        if ta!=tb or len(aa)!=len(bb) or any(a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True) for a,b in zip(aa,bb)):
            raise ValueError('own saved continuation leaf changed during loading')
        initial_identity['continuation_snapshot']=dict(all_saved_leaves_preserved=True,common_origin_state='approximate; user-authorized numerical amendment',prefix_lane_maximum_absolute_difference=p['continuation_source']['prefix_lane_maximum_absolute_difference'])
        if not bool(j.all(initial.state.tick==200)):raise ValueError('continuation task1 tick mismatch')

    if bool(j.any(initial.state.fault|initial.state.physical.failed)):
        raise ValueError('initial native task is invalid; not a physical failure result')
    env.set_log_template(jax.tree.map(lambda x:x[0],initial.state),bank[case_path_key(cases[0])],speeds[0])
    if online:
        schedule=file_module('_sttw_online_scalar',folder/'geometric_preference_schedule.py')
        initial=schedule.OnlineCarry(initial,j.zeros(len(cases),bool))
    actors={}
    for name,model in p['models'].items():
        with Path(model['actor']).open('rb') as f:actors[name]=jax.tree.map(j.asarray,pickle.load(f))
    actor=make_path_actor(cfg)
    for params in actors.values():
        if jax.eval_shape(actor.apply,params,j.zeros(351)).shape!=(2,):raise ValueError('native Actor shape mismatch')
    params=stack([actors[c['mode'] if c['mode'] in actors else 'H75'] for c in cases])
    n=len(cases);horizon=p['horizon_s'];speed_alpha=j.asarray([c.get('query_alpha',c.get('speed_alpha',c['alpha'])) for c in cases]);steer_alpha=j.asarray([c.get('query_alpha',c.get('steer_alpha',c['alpha'])) for c in cases])
    modes=j.asarray([c['mode_index'] for c in cases],j.int32)
    physical_start=float(snapshot.physical_tick)*env.cc.dt
    pulses=j.asarray(force_pulses(cases,physical_start))
    keys=list(configured_cost_keys(cfg))
    keep=('active_tick','time','actual_xy','reference_xy','actual_forward_speed','wheel_speed_proxy','v_user',
          'path_cross_track','path_heading_error','path_progress','phi','phi_dot','actual_delta','peak_roll',
          'physical_failure','contact_failure','nonfinite','domain_exit','policy_fault','lower_fault',
          'governed','nominal','applied_force_impulse','scored_tick_reward','latent_z','offsets','filtered_offset','target_offset',
          'reference_target_permitted','cumulative_domain_exit','clean_blocks_at_publication','clean_blocks_at_end',
          'continuation_geometry_invalid','continuation_geometry_fault')
    switch_ticks=j.asarray([round(c.get('switch_s',1e9)/env.cc.dt) for c in cases],j.int32) if switching else j.zeros(n,j.int32)
    def one(carry,param,path,v,pulse,mode,sa,ta,st):
        if online:
            alpha_exec,latched,decision=schedule.advance_scalar(carry.latched,v,path.s,path.curvature,carry.state.projection.progress,p['protocol']['rule'])
            if confirmation:
                alpha_exec=j.where(mode==0,alpha_exec,sa);latched=j.where(mode==0,latched,False)
                decision=dict(decision,rule_alpha_target=alpha_exec)
            carry=carry.native
        origin_pair=j.stack([sa,ta])
        if switching:sa,ta,switched=switch_mapping(carry.state.tick,sa,ta,st,p['protocol']['destination_pair'])
        if response:
            obs,_,_=env.observation(carry.state,path,v,horizon)
            if online:
                queried=obs.at[336].set(alpha_exec)
                z=jax.lax.cond(mode==4,lambda _:j.zeros(2),lambda _:actor.apply(param,queried),None) if confirmation else actor.apply(param,queried)
                half=full=j.zeros_like(z)
                if confirmation:
                    support_z,support=existing_floor_latent(carry.state.physical.raw,cfg,p['protocol']['existing_speed_floor_m_s'])
                    z=j.where(mode==4,support_z,z)
            elif samestate:
                queried=obs.at[336].set(sa);z=actor.apply(param,queried);half=full=j.zeros_like(z)
            else:z,half,full=channel_queries(obs,lambda x:actor.apply(param,x),sa,ta)
        def target(obs,state):
            if response:return z
            return jax.lax.cond(mode==3,
                lambda _:rule.target(state.physical.raw,state.alpha,cfg,lambda x:q_ratio(x,env.cc),p['rule']['budget_rad'],p['rule']['grid_points'])[0],
                lambda _:actor.apply(param,obs),None)
        end,log=env.policy_step(carry,path,v,horizon,target,pulse,zero_mode=False if confirmation or samestate else mode==2)
        if response:
            ticks=lambda x:j.broadcast_to(x,log['active_tick'].shape+j.asarray(x).shape)
            log.update(queried_latent_alpha_half=ticks(half),queried_latent_alpha_one=ticks(full),channel_alpha_pair=ticks(j.stack([sa,ta])),composed_latent=ticks(z),query_observation=ticks(obs),pre_upper_yaw=ticks(env.physics.observe(carry.state.physical.data)[1][2]))
        if samestate:log.update(query_observation=ticks(queried),channel_alpha_pair=ticks(j.stack([sa,sa])),alpha_user=ticks(carry.state.alpha),alpha_exec=ticks(sa))
        if confirmation:
            log.update(neural_query_active=ticks(mode!=4),support_nominal_reference=ticks(carry.state.physical.raw),rule_pre_upper_progress=ticks(carry.state.projection.progress),**{k:ticks(j.where(mode==4,x,j.zeros_like(x))) for k,x in support.items()})
        if online:
            log.update(alpha_user=ticks(j.asarray(1.)),alpha_exec=ticks(alpha_exec),rule_latched=ticks(latched),query_observation=ticks(queried),channel_alpha_pair=ticks(j.stack([alpha_exec,alpha_exec])),**{k:ticks(x) for k,x in decision.items()})
            end=schedule.OnlineCarry(end,latched)
        if switching:log.update(origin_channel_alpha_pair=ticks(origin_pair),switch_active=ticks(switched),switch_tick=ticks(st))
        return end,dict({k:log[k] for k in keep},**({k:log[k] for k in ('alpha_user','alpha_exec')} if samestate else {}),**({k:log[k] for k in ('neural_query_active','support_nominal_reference','rule_pre_upper_progress','support_floor','support_requested_offset','support_realized_target_offset','support_active')} if confirmation else {}),**({k:log[k] for k in ('alpha_user','alpha_exec','rule_latched','rule_preview_max_curvature','rule_accel_estimate','rule_alpha_target')} if online else {}),**({k:log[k] for k in ('origin_channel_alpha_pair','switch_active','switch_tick')} if switching else {}),**({k:log[k] for k in (('channel_alpha_pair','composed_latent','query_observation','pre_upper_yaw','reference_yaw_unwrapped','e_psi_unwrapped') if online or samestate else ('queried_latent_alpha_half','queried_latent_alpha_one','channel_alpha_pair','composed_latent','query_observation','pre_upper_yaw','reference_yaw_unwrapped','e_psi_unwrapped'))} if response else {}),effective_costs=j.stack([log['effective_components'][k] for k in keys],axis=-1))
    @jax.jit
    def chunk(carry,paths):
        return jax.lax.scan(lambda c,_:jax.vmap(one)(c,params,paths,speeds,pulses,modes,speed_alpha,steer_alpha,switch_ticks),carry,None,length=50)
    @jax.jit
    def measurements(carry):
        return jax.vmap(lambda d:env.physics.observe(d)[1])(carry.state.physical.data)
    @jax.jit
    def bind(carry,paths,mask):
        if online:
            native=jax.vmap(env.rebind_geometry)(carry.native,paths,speeds)
            new=schedule.OnlineCarry(native,carry.latched)
        else:new=jax.vmap(env.rebind_geometry)(carry,paths,speeds)
        return jax.tree.map(lambda a,b:j.where(mask.reshape((n,)+(1,)*(a.ndim-1)),a,b),new,carry)
    consumed={}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.py' and Path(file).resolve().is_relative_to(native):
            file=str(Path(file).resolve());expected=p['native_expected_sources'].get(file)
            if expected is None or digest(file)!=expected:raise ValueError('native consumed source drift '+file)
            consumed[file]=expected
    # This is a diagnostic consistency check, not a new reward/criterion.
    terminal20=float(failure_reward(1000,cfg));terminal10=float(failure_reward(500,cfg))
    return dict(plan=p,config=cfg,cases=cases,env=env,initial=initial,paths=paths,bank=bank,speeds=speeds,
                chunk=chunk,bind=bind,measurements=measurements,planner=planner,keys=keys,
                physical_start_s=physical_start,consumed=consumed,terminal20=terminal20,terminal10=terminal10,initial_identity=initial_identity if response else None)


def flatten(logs):
    import numpy as np
    return {k:np.asarray(v).transpose((0,2,1)+tuple(range(3,v.ndim))).reshape((-1,v.shape[1])+v.shape[3:]) for k,v in logs.items()}


def run(root,*,shape_only=False):
    root=Path(root).resolve();p=verify(root)
    if not shape_only and json.loads((root/'status.json').read_text())['phase']!='prepared':raise ValueError('prepared only; no implicit rerun')
    directory=root/('shape' if shape_only else 'evaluation');directory.mkdir(exist_ok=False)
    started=time.monotonic();status=dict(phase='initializing',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0)
    def update(**kw):
        status.update(kw,elapsed_s=time.monotonic()-started)
        if p.get('mode')=='h75_switching_continuation':
            status.update(prior_consumed_lower_ticks=p['protocol']['consumed_source_lower_ticks'],campaign_active_lower_ticks=p['protocol']['consumed_source_lower_ticks']+status.get('active_lower_ticks',0))
        save(directory/'status.json',status)
        if not shape_only:save(root/'status.json',status)
    update()
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        n=p['cases'];runtime=load(root);carry=runtime['initial'];paths=runtime['paths'];env=runtime['env']
        shaped=jax.eval_shape(runtime['chunk'],carry,paths)
        jax.eval_shape(runtime['bind'],carry,paths,j.zeros(n,bool))
        receipt=dict(active_shape=list(shaped[1]['active_tick'].shape),real_lanes=n,padding=0,physics_steps=0,
                     consumed_native_sources=runtime['consumed'],physical_start_s=runtime['physical_start_s'],
                     terminal_cost20=runtime['terminal20'],terminal_cost10=runtime['terminal10'],
                     log_template_after_wrapper=True,planner_path_type='Explicit native Path alias; only planner/vendor namespace isolated')
        if runtime['initial_identity'] is not None:receipt.update(runtime['initial_identity'])
        save(directory/'shape_receipt.json',receipt)
        if shape_only:update(phase='completed',physics_steps=0);return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU required for numerical evaluation')
        cases=runtime['cases'];path_list=[runtime['bank'][case_path_key(c)] for c in cases]
        goals=np.array([[np.interp(float(path.goal),np.asarray(path.s),np.asarray(path.xy)[:,axis]) for axis in (0,1)] for path in path_list])
        for name,path in runtime['bank'].items():np.savez_compressed(directory/(name+'_path.npz'),**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
        save(directory/'manifest.json',dict(config=p['config'],source=p['initial_source'],cases=cases,models=p['models'],
            original_goals_xy=goals.tolist(),reward_components=runtime['keys'],physics_identity=env.physics.bundle.identity,
            plan_sha256=digest(root/'plan.json'),clock='time is task pre-tick, actual_xy is post-tick; pulse uses task time plus physical_start_s',
            data_role=p['data_role'],physical_start_s=runtime['physical_start_s'],initialization=p.get('initialization'),prepared_bank_indices=p['protocol'].get('bank_indices') if 'protocol' in p else None,prepared_state_fingerprints=p['protocol'].get('bank_state_fingerprints') if 'protocol' in p else None))
        with (directory/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        all_logs=[];decisions=[];changed=np.zeros(n,bool);arrived=np.zeros(n,bool);actual=0
        samestate=p.get('mode')=='h75_samestate_tolerance72'
        branch=p.get('mode') in ('h75_switching_continuation','h75_samestate_tolerance72')
        if branch:
            source=Path(p['samestate_source']['source'] if samestate else p['continuation_source']['source']);indices=[c['source_index'] for c in cases] if samestate else list(range(n))
            with np.load(source/'evaluation/reused_source_prefix.npz' if samestate else source/'evaluation/chunk_00.npz') as old:
                prefix={k:old[k][:,indices] for k in old.files}
            if samestate:
                # Prefix is display evidence only; retain suffix-only case logs/schema.
                observations=jax.jit(jax.vmap(lambda state,path,v:env.observation(state,path,v,p['horizon_s'])[0]))(carry.state,paths,runtime['speeds'])
                np.savez_compressed(directory/'source_features.npz',native_observation_user1=np.asarray(jax.device_get(observations)))
            else:all_logs.append(prefix)
            np.savez_compressed(directory/'reused_source_prefix.npz',**prefix)
            with np.load(source/'evaluation/task1_host_state.npz') as host:
                changed=host['replanned'][indices].copy();arrived=host['ever_arrived'][indices].copy()
                if not np.array_equal(goals,host['original_goals_xy'][indices]):raise ValueError('continuation original host goals differ')
            if not samestate:
                prefix_audit=audit_stay_prefix(prefix,cases,json.loads((root/'cached_stay_records.json').read_text()),p['protocol'])
                save(directory/'source_prefix_audit.json',prefix_audit)
            save(directory/'prefix_provenance.json',dict(source=str(source),source_indices=indices,reused_lower_slots=200*n,new_physics_ticks=0,source_consumed_lower_ticks=0 if samestate else p['protocol']['consumed_source_lower_ticks'],total_budget=p['maximum_active_lower_ticks'] if samestate else p['protocol']['total_campaign_lower_ticks'],warning='Source prefix reused for display only; suffix case logs begin at task1' if samestate else p['protocol']['prefix_provenance']))
            save(directory/'task1_capture_receipt.json',runtime['initial_identity'] if samestate else runtime['initial_identity']['continuation_snapshot'])
            with (directory/'task1_complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            np.savez_compressed(directory/'task1_host_state.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)

        for block in range(p['chunks']):
            if block==1 and p.get('mode')=='h75_channel_switching':
                with (directory/'task1_complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
                if not np.all(np.asarray(jax.device_get(carry.state.tick))==round(p['protocol']['capture_task_s']/env.cc.dt)):
                    raise ValueError('capture is not at declared task1.0 boundary')
                np.savez_compressed(directory/'task1_host_state.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)
                capture=assert_shared_switch_carry(carry,cases)
                cached=json.loads((root/'cached_stay_records.json').read_text())
                capture['stay_prefix']=audit_stay_prefix(all_logs[0],cases,cached,p['protocol'])
                save(directory/'task1_capture_receipt.json',capture)
            if block or branch:
                poses= jax.device_get(runtime['measurements'](carry))
                cross,invalid,failed,fault,ticks=jax.device_get((carry.state.projection.cross_track,carry.state.follower.invalid,carry.state.physical.failed,carry.state.fault,carry.state.tick))
                mask=np.zeros(n,bool)
                for i in range(n):
                    if runtime['planner'].replan_requested(xy=poses[i,:2],goal_xy=goals[i],cross_track_m=float(cross[i]),
                        follower_invalid=bool(invalid[i]),already_replanned=bool(changed[i]),contact=bool(failed[i]),fault=bool(fault[i]),ever_arrived=bool(arrived[i])):
                        if int(ticks[i])%4:raise ValueError('replanning boundary was not an upper boundary')
                        path,meta=runtime['planner'].dubins_goal_path(poses[i],goals[i],radius_m=p['replanning']['radius_m'],**(dict(length_m=p['protocol']['planner_support_m'],spacing_m=p['protocol']['planner_spacing_m']) if p.get('mode') in ('h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar','h75_online_confirmation100','h75_samestate_tolerance72') else {}))
                        path_list[i]=path;changed[i]=True;mask[i]=True
                        np.savez_compressed(directory/f'replanned_path_{i:02d}.npz',**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
                        decisions.append(dict(case_id=cases[i]['case_id'],case_index=i,task_tick=int(ticks[i]),measured_pose=np.asarray(poses[i]).tolist(),cross_track_m=float(cross[i]),follower_invalid=bool(invalid[i]),contact=bool(failed[i]),fault=bool(fault[i]),ever_arrived=bool(arrived[i]),**meta))
                if mask.any():
                    paths=jax.tree.map(lambda *x:j.stack(x),*path_list)
                    carry=runtime['bind'](carry,paths,j.asarray(mask))
                save(directory/'replanning_decisions.json',decisions)
            carry,logs=runtime['chunk'](carry,paths);jax.block_until_ready(carry)
            logs=flatten(jax.device_get(logs));all_logs.append(logs)
            np.savez_compressed(directory/f'chunk_{block+int(branch):02d}.npz',**logs)
            fatal=logs['contact_failure']|logs['nonfinite']|logs['lower_fault']|logs['policy_fault']|logs['continuation_geometry_fault']
            eligible=logs['active_tick']&(np.cumsum(fatal,axis=0)==0)
            arrived|=np.any(eligible&(np.linalg.norm(logs['actual_xy']-goals[None,:,:],axis=-1)<=1.),axis=0)
            # Persist returned state before any numerical analysis can fail.
            with (directory/f'complete_carry_{block+int(branch):02d}.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            np.savez_compressed(directory/f'host_state_{block+int(branch):02d}.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)
            with (directory/f'executed_paths_{block+int(branch):02d}.pkl').open('wb') as f:pickle.dump(jax.device_get(paths),f)

            actual+=int(logs['active_tick'].sum())
            if actual>p['maximum_active_lower_ticks'] or (block+1)*50*4*n>p['maximum_allocated_lower_slots']:
                raise RuntimeError('declared lower-tick budget exceeded')
            update(phase='running',completed_blocks=block+1,active_lower_ticks=actual,allocated_lower_slots=(block+1)*50*4*n,replanned_cases=int(changed.sum()))
            if p.get('mode')=='h75_switching_continuation':
                scheduled=audit_scheduled_pairs(logs,cases)
                save(directory/'scheduled_pair_receipt.json',dict(scheduled,completed_blocks=block+1))
            if (block==1 and p.get('mode')=='h75_channel_switching') or (block==0 and branch and not samestate):
                prefix={k:np.concatenate([b[k] for b in all_logs]) for k in all_logs[0]}
                checked=audit_stay_prefix(prefix,cases,json.loads((root/'cached_stay_records.json').read_text()),p['protocol'],own_switch=True)
                save(directory/'own_switch_prefix_receipt.json',checked)
            if np.any(logs['nonfinite']|logs['lower_fault']|logs['policy_fault']|logs['continuation_geometry_fault']) or bool(j.any(carry.state.fault)):
                raise RuntimeError('engineering fault: partial raw retained, no negative label or automatic retry')
            if bool(j.all(carry.state.physical.failed)):break
        data={k:np.concatenate([b[k] for b in all_logs]) for k in all_logs[0]};(directory/'cases').mkdir();index=[]
        for i,c in enumerate(cases):
            file=directory/'cases'/f'case_{i:02d}.npz';np.savez_compressed(file,**{k:v[:,i] for k,v in data.items()})
            index.append(dict(c,raw_file=str(file),raw_sha256=digest(file),original_goal_xy=goals[i].tolist(),replanned=bool(changed[i])))
        with (directory/'final_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        save(directory/'case_index.json',index);save(directory/'replanning_decisions.json',decisions)
        update(phase='completed',completed_cases=n)
    except BaseException as exc:
        (directory/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
