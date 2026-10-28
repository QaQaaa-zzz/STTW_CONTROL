"""Isolated native-package initial-carry audit; never constructs an environment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode=True


def load_frozen_pickle_module(root,plan,filename,alias):
    """Bind the sealed run's actual pickle class, without constructing physics."""
    source=root/'source'/filename
    expected=[sha for file,sha in plan['source_sha256'].items() if Path(file).name==filename]
    if len(expected)!=1 or hashlib.sha256(source.read_bytes()).hexdigest()!=expected[0]:
        raise ValueError('frozen pickle module identity mismatch '+filename)
    spec=importlib.util.spec_from_file_location(alias,source)
    module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
    return module


def audit(root,output):
    import numpy as np
    p=json.loads((root/'plan.json').read_text());native=Path(p['native_runtime_root']).resolve()
    if 'sttw_control' in sys.modules:raise ValueError('fresh native-only audit process required')
    sys.path.insert(0,str(native))
    late=p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192')
    tolerance=p.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation')
    if late:
        original=Path(p['samestate_source']['source']);original_plan=json.loads((original/'plan.json').read_text())
        load_frozen_pickle_module(original,original_plan,'tolerance_coordination.py','_sttw_tolerance_coordination')
    if tolerance:
        load_frozen_pickle_module(root,p,'tolerance_coordination.py','_sttw_tolerance_coordination')
    source=(root/'source/native_goal_continuation.py') if tolerance else Path(__file__).resolve().parents[1]/'src/sttw_control/native_goal_continuation.py'
    if tolerance:adapter=load_frozen_pickle_module(root,p,'native_goal_continuation.py','_sttw_goal_continuation')
    else:
        spec=importlib.util.spec_from_file_location('_sttw_goal_continuation',source)
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
    import jax
    online=p.get('mode') in ('h75_online_scalar','h75_online_confirmation100','tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation')
    if online or p.get('mode')=='h75_samestate_bank144':
        schedule_source=Path(__file__).resolve().parents[1]/'src/sttw_control/geometric_preference_schedule.py'
        schedule_spec=importlib.util.spec_from_file_location('_sttw_online_scalar',schedule_source)
        schedule=importlib.util.module_from_spec(schedule_spec);sys.modules[schedule_spec.name]=schedule;schedule_spec.loader.exec_module(schedule)
    initial=root/'evaluation/initial_carry.pkl'
    with initial.open('rb') as f:carry=pickle.load(f)
    if online:
        if np.any(carry.latched):raise ValueError('online initial latch must be false')
        if tolerance and np.any(carry.prev_alpha!=np.asarray([c['alpha'] for c in json.loads((root/'inventory.json').read_text())])):raise ValueError('online initial previous alpha must be one')
        carry=carry.native
    if p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192'):
        return audit_bank144_initial(root,p,carry,output,jax,initial)
    if p.get('mode')=='h75_samestate_tolerance72':
        return audit_samestate_initial(root,p,carry,output,jax,initial)
    continuation=p.get('mode')=='h75_switching_continuation'
    resumed=carry if continuation else None
    if continuation:
        failed_source=Path(p['continuation_source']['source'])
        with (failed_source/'evaluation/initial_carry.pkl').open('rb') as f:carry=pickle.load(f)
    source_bank=Path(p['initial_source']['prepared_bank'])
    if hashlib.sha256(source_bank.read_bytes()).hexdigest()!=p['initial_source']['prepared_bank_sha256']:
        raise ValueError('original complete prepared bank identity drift')
    with source_bank.open('rb') as f:bank=pickle.load(f)
    snapshot=jax.tree.map(lambda x:x[p['initial_source']['bank_index']],bank)
    cases=json.loads((root/'inventory.json').read_text())
    if p.get('mode') in ('h75_online_confirmation100','tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation'):
        return audit_confirmation_initial(root,p,carry,bank,output,jax,initial)
    def equal(a,b,label,*,native_load_cast=False,strict_dtype=False):
        aa,ta=jax.tree.flatten(a);bb,tb=jax.tree.flatten(b)
        if ta!=tb:raise ValueError('initial structure mismatch '+label)
        for x,y in zip(aa,bb):
            actual=np.asarray(x);expected=np.asarray(y,dtype=actual.dtype) if native_load_cast else np.asarray(y)
            if strict_dtype and actual.dtype!=expected.dtype:raise ValueError('initial dtype mismatch '+label)
            if not np.array_equal(actual,expected,equal_nan=True):
                raise ValueError('initial dynamic leaf mismatch '+label)
    switching=p.get('mode') in ('h75_channel_switching','h75_switching_continuation')
    response=p.get('mode') in ('h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar')
    count=len(cases)
    if count!=(18 if online else 36 if switching else 54 if response else 80):raise ValueError('declared initial inventory size mismatch')
    state=carry.state
    np.testing.assert_array_equal(state.tick,np.zeros(count))
    np.testing.assert_array_equal(state.physical.physical_tick,np.full(count,int(snapshot.physical_tick)))
    np.testing.assert_array_equal(state.alpha,[c['alpha'] for c in cases])
    np.testing.assert_array_equal(state.filtered,np.zeros((count,2)))
    np.testing.assert_array_equal(state.offsets,np.zeros((count,2)))
    for value in (state.fault,state.physical.failed,state.domain_exit,carry.ever_domain_exit,carry.clean_blocks):
        if np.any(value):raise ValueError('initial stop/domain/clean state differs from declared task0')
    # Native reset may update these four reference/pose bookkeeping fields.
    # Everything else in the prepared physical snapshot must be retained.
    for i in range(count):
        physical=jax.tree.map(lambda x:x[i],state.physical)
        for field in snapshot.__dataclass_fields__:
            if field not in ('raw','reference_pose','yaw_wrapped','yaw_unwrapped'):
                equal(getattr(physical,field),getattr(snapshot,field),f'prepared physical lane{i} '+field,native_load_cast=True)
    groups={};pairs=0
    for i,c in enumerate(cases):
        key=(c['route'],c['alpha'],c['event'])
        current=jax.tree.map(lambda x:x[i],carry)
        if key in groups:equal(current,groups[key],c['case_id']);pairs+=1
        else:groups[key]=current
    expected_groups,expected_pairs=(18,0) if online else (6,30) if switching else (36,18) if response else (20,60)
    if len(groups)!=expected_groups or pairs!=expected_pairs:raise ValueError('matched complete initial groups missing')
    if response:
        receipt=json.loads((root/'evaluation/shape_receipt.json').read_text())
        if not all(receipt.get(k) is True for k in ('all_native_leaves_exact','dtypes_exact','original_paths_exact')):
            raise ValueError('native original reset/path receipt mismatch')
        cached=Path(p['cached_source']);cached_file=cached/'evaluation/initial_carry.pkl'
        if hashlib.sha256(cached_file.read_bytes()).hexdigest()!=p['input_sha256'][str(cached_file)]:
            raise ValueError('cached original native reset SHA mismatch')
        with cached_file.open('rb') as f:native_carry=pickle.load(f)
        old_cases=json.loads((cached/'plan.json').read_text())['cases']
        for i,c in enumerate(cases):
            j=next(j for j,old in enumerate(old_cases) if old['route']==c['route'] and old['speed']==c['speed'] and old['policy']=='H75' and old['alpha']==c['alpha'])
            expected=jax.tree.map(lambda x:x[j],native_carry)
            actual=jax.tree.map(lambda x:x[i],state)
            equal(actual,expected.replace(domain_exit=np.asarray(False)),f'cached original native reset lane{i}',strict_dtype=True)
    switch_review=None;cached_filtered=None;cached_offsets=None
    if switching:
        task1=root/'evaluation/task1_complete_carry.pkl'
        with task1.open('rb') as f:captured=pickle.load(f)
        np.testing.assert_array_equal(captured.state.tick,np.full(count,200))
        capture_groups={};capture_differences=[]
        if continuation:
            with (failed_source/'evaluation/task1_complete_carry.pkl').open('rb') as f:saved=pickle.load(f)
            expected=jax.tree.map(jax.numpy.asarray,saved)
            equal(resumed,expected,'own saved task1 continuation',strict_dtype=True)
            equal(captured,resumed,'saved task1 vs resumed initial',strict_dtype=True)
        for i,c in enumerate(cases):
            key=(c['route'],c['origin']);lane=jax.tree.map(lambda x:x[i],captured)
            if key in capture_groups:
                if not continuation:equal(lane,capture_groups[key],str(key)+' task1',strict_dtype=True)
                else:
                    aa,ta=jax.tree.flatten(lane);bb,tb=jax.tree.flatten(capture_groups[key])
                    if ta!=tb:raise ValueError('continuation group structure differs')
                    differences=[]
                    for leaf,(a,b) in enumerate(zip(aa,bb)):
                        a=np.asarray(a);b=np.asarray(b)
                        if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('continuation group dtype/shape differs')
                        delta=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
                        if delta or not np.array_equal(a,b,equal_nan=True):differences.append(dict(leaf=leaf,maximum_absolute_difference=delta,dtype=str(a.dtype)))
                    capture_differences.append(dict(group=list(key),lane=i,differing_leaves=differences))
            else:capture_groups[key]=lane
        if len(capture_groups)!=4:raise ValueError('four complete task1 source groups required')
        with np.load(root/'evaluation/task1_host_state.npz') as z:
            for field in z.files:
                host_groups={}
                for i,c in enumerate(cases):
                    key=(c['route'],c['origin'])
                    if key in host_groups and not continuation:np.testing.assert_array_equal(z[field][i],host_groups[key])
                    else:host_groups[key]=z[field][i]
        if continuation:
            with np.load(root/'evaluation/task1_host_state.npz') as actual,np.load(failed_source/'evaluation/task1_host_state.npz') as original:
                for field in original.files:np.testing.assert_array_equal(actual[field],original[field])
        capture=json.loads((root/'evaluation/task1_capture_receipt.json').read_text())
        if continuation:
            if not capture.get('all_saved_leaves_preserved'):raise ValueError('own saved continuation receipt missing')
        else:
            if capture.get('groups')!=4 or capture.get('lanes_per_group')!=9 or not capture.get('all_leaves_exact') or capture.get('leaf_count')!=len(jax.tree.leaves(captured)):
                raise ValueError('task1 full-carry host receipt mismatch')
            prefix=capture['stay_prefix']
            if not prefix.get('all_fields_within_tolerance') or not prefix.get('discrete_exact'):raise ValueError('task1 stay-prefix receipt mismatch')
        stay=Path(p['stay_source']);stay_initial=stay/'evaluation/initial_carry.pkl'
        receipt=json.loads((stay/'analysis/receipt.json').read_text())
        if hashlib.sha256(stay_initial.read_bytes()).hexdigest()!=receipt['initial_carry_sha256']:raise ValueError('stay initial carry differs from recorded source analysis')
        with stay_initial.open('rb') as f:cached=pickle.load(f)
        cached_cases=json.loads((stay/'inventory.json').read_text())
        for i,c in enumerate(cached_cases):
            j=next(j for j,old in enumerate(old_cases) if old['route']==c['route'] and old['speed']==c['speed'] and old['policy']=='H75' and old['alpha']==c['alpha'])
            expected=jax.tree.map(lambda x:x[j],native_carry)
            actual=jax.tree.map(lambda x:x[i],cached.state)
            equal(actual,expected.replace(domain_exit=np.asarray(False)),f'cached stay native initial lane{i}',strict_dtype=True)
        cached_filtered=np.asarray(cached.state.filtered).tolist();cached_offsets=np.asarray(cached.state.offsets).tolist()
        switch_review=dict(task1_complete_carry_sha256=hashlib.sha256(task1.read_bytes()).hexdigest(),
            groups=4,lanes_per_group=9,complete_carry_leaves_exact=not continuation,host_state_groups_exact=not continuation,
            all_own_saved_leaves_preserved=continuation,approximate_group_differences=capture_differences,
            numerical_amendment=p['protocol'].get('numerical_amendment'),
            task1_before_planning_and_switch=True,stay_initial_carry_sha256=receipt['initial_carry_sha256'])
    if online:
        stay=Path(p['stay_source']);stay_initial=stay/'evaluation/initial_carry.pkl'
        receipt=json.loads((stay/'analysis/receipt.json').read_text())
        if hashlib.sha256(stay_initial.read_bytes()).hexdigest()!=receipt['initial_carry_sha256']:raise ValueError('online cached initial carry drift')
        with stay_initial.open('rb') as f:cached=pickle.load(f)
        cached_cases=json.loads((stay/'inventory.json').read_text())
        for i,c in enumerate(cached_cases):
            j=next(j for j,old in enumerate(old_cases) if old['route']==c['route'] and old['speed']==c['speed'] and old['policy']=='H75' and old['alpha']==c['alpha'])
            expected=jax.tree.map(lambda x:x[j],native_carry);actual=jax.tree.map(lambda x:x[i],cached.state)
            equal(actual,expected.replace(domain_exit=np.asarray(False)),f'online source cached reset lane{i}',strict_dtype=True)
        cached_filtered=np.asarray(cached.state.filtered).tolist();cached_offsets=np.asarray(cached.state.offsets).tolist()
    loaded={}
    for name,module in tuple(sys.modules.items()):
        if name=='sttw_control' or name.startswith('sttw_control.'):
            file=getattr(module,'__file__',None)
            if file and not Path(file).resolve().is_relative_to(native):raise ValueError('mixed initial native package')
            if file:loaded[name]=str(Path(file).resolve())
    result=dict(initial_carry_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_bank_sha256=p['initial_source']['prepared_bank_sha256'],bank_index=p['initial_source']['bank_index'],
        physical_tick=int(snapshot.physical_tick),native_only_modules=loaded,
        complete_matched_method_pairs=pairs,matched_groups=len(groups),
        initial_filtered=np.asarray(state.filtered).tolist(),initial_offsets=np.asarray(state.offsets).tolist(),
        online_native_initial_preserved=online,online_initial_latch_false=online,
        task_ticks_zero=not continuation,resume_task_tick=200 if continuation else 0,
        initial_filter_scope='Original source task0 for fullH10 raw reconstruction; resumed task1 state separately preserved' if continuation else 'Task0 initial state',original_alpha_checked=True,prepared_physical_fields_checked=True,prepared_physical_lanes_checked=count,
        excluded_native_reset_fields=['raw','reference_pose','yaw_wrapped','yaw_unwrapped'],
        source_dtype_semantics='Prepared-bank leaves converted to actual loaded leaf dtypes, as in native j.asarray loader; matched-method comparisons exact without casts',
        switching_capture=switch_review,cached_initial_filtered=cached_filtered,cached_initial_offsets=cached_offsets,
        pairing_contract='Full carry matched within route/event/state-alpha; physical bank fields checked across every lane. Different state alphas intentionally retained.',
        new_physics_steps=0,optimizer_updates=0,verification='AI array/structure audit, not independent control replay')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def confirmation_initial_layout(plan):
    expected={'h75_online_confirmation100':(100,5),'tolerance_online120':(120,6),'tolerance_online40':(40,2),'tolerance_broaden120':(120,5),'user_preference120':(120,5),'user_prior_online72':(72,3),'task_aware_fallback72':(72,3),'frozen_confirmation200':(200,10),'speed_query_online60':(60,3),'guard_fallback_evaluation':(72,3)}
    if plan.get('mode') not in expected:raise ValueError('unsupported initial panel mode')
    count,arms=expected[plan['mode']]
    if plan['mode']=='frozen_confirmation200':
        variant=plan['protocol'].get('variant')
        if variant not in (None,'geometry_duration_extension_v1'):raise ValueError('unknown frozen confirmation variant')
        if variant=='geometry_duration_extension_v1':count,arms=240,10
    if plan['mode']=='guard_fallback_evaluation':
        stage=plan['protocol'].get('stage')
        if stage not in ('extension','regression'):raise ValueError('unknown guard fallback stage')
        count,arms=(72,3) if stage=='extension' else (60,3)
    if plan['protocol']['expected_cases']!=count or len(plan['protocol']['arms'])!=arms:
        raise ValueError('initial panel dimensions differ from declared mode')
    return count,arms


def audit_confirmation_initial(root,p,carry,bank,output,jax,initial):
    import numpy as np
    cases=json.loads((root/'inventory.json').read_text());state=carry.state
    tolerance=p.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation')
    count,arms=confirmation_initial_layout(p)
    if len(cases)!=count:raise ValueError('complete declared initial cases required')
    np.testing.assert_array_equal(state.tick,np.zeros(count));np.testing.assert_array_equal(state.alpha,[c['alpha'] for c in cases])
    np.testing.assert_array_equal(state.filtered,np.zeros((count,2)));np.testing.assert_array_equal(state.offsets,np.zeros((count,2)))
    for value in (state.fault,state.physical.failed,state.domain_exit,carry.ever_domain_exit,carry.clean_blocks):
        if np.any(value):raise ValueError('new bank initial stop/fault/clean latch')
    snapshots={b:jax.tree.map(lambda x:jax.numpy.asarray(x[b]),bank) for b in p['protocol']['bank_indices']}
    fingerprints={};groups={}
    for b,snapshot in snapshots.items():
        sha=hashlib.sha256()
        for path,value in jax.tree_util.tree_flatten_with_path(snapshot)[0]:
            a=np.asarray(value);sha.update(jax.tree_util.keystr(path).encode());sha.update(str(a.dtype).encode());sha.update(a.tobytes())
        fingerprints[str(b)]=sha.hexdigest()
        if sha.hexdigest()!=p['protocol']['bank_state_fingerprints'][str(b)]:raise ValueError('new bank fingerprint mismatch')
    for i,c in enumerate(cases):
        snapshot=snapshots[c['bank_index']]
        if c['initial_fingerprint']!=fingerprints[str(c['bank_index'])]:raise ValueError('case snapshot fingerprint differs')
        physical=jax.tree.map(lambda x:x[i],state.physical)
        for field in snapshot.__dataclass_fields__:
            if field in ('raw','reference_pose','yaw_wrapped','yaw_unwrapped'):continue
            actual,ta=jax.tree.flatten(getattr(physical,field));expected,tb=jax.tree.flatten(getattr(snapshot,field))
            if ta!=tb:raise ValueError('newbank physical structure drift '+field)
            for a,b in zip(actual,expected):
                a=np.asarray(a);b=np.asarray(b)
                if a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True):raise ValueError('newbank complete physical snapshot drift '+field)
        key=(c['bank_index'],c['route'],c['event']);groups.setdefault(key,[]).append(i)
        with np.load(root/'evaluation'/(c['path_key']+'_path.npz')) as z:
            np.testing.assert_allclose(z['xy'][0],snapshot.data.qpos[:2],atol=3e-6,rtol=0)
            np.testing.assert_allclose(z['heading'][0],state.physical.reference_pose[i,2],atol=3e-6,rtol=0)
    neutral=carry.replace(state=carry.state.replace(alpha=jax.numpy.zeros_like(carry.state.alpha))) if p.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation') else carry
    leaves=jax.tree.leaves(neutral)
    expected_groups=24 if p.get('mode') in ('tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72') or (p.get('mode') in ('frozen_confirmation200','guard_fallback_evaluation') and p['protocol'].get('variant')=='geometry_duration_extension_v1') else 20
    if len(groups)!=expected_groups or any(len(ids)!=arms for ids in groups.values()):raise ValueError(f'{expected_groups} paired bank/route/event groups required')
    for ids in groups.values():
        for leaf in leaves:
            for i in ids[1:]:np.testing.assert_array_equal(leaf[i],leaf[ids[0]])
    receipt=json.loads((root/'evaluation/shape_receipt.json').read_text())
    if not receipt.get('all_arm_reset_leaves_exact' if tolerance else 'all_five_arm_reset_leaves_exact') or receipt.get('new_bank_fingerprints')!=fingerprints:raise ValueError('paired reset receipt mismatch')
    loaded={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if (name=='sttw_control' or name.startswith('sttw_control.')) and getattr(module,'__file__',None)}
    native=Path(p['native_runtime_root']).resolve()
    if any(not Path(file).is_relative_to(native) for file in loaded.values()):raise ValueError('mixed native initial100 modules')
    result=dict(initial_carry_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_bank_sha256=p['initial_source']['prepared_bank_sha256'],bank_indices=p['protocol']['bank_indices'],bank_fingerprints=fingerprints,
        initial_filtered=np.asarray(state.filtered).tolist(),initial_offsets=np.asarray(state.offsets).tolist(),
        initial_nominal_reference=np.asarray(state.physical.raw).tolist(),
        matched_groups=expected_groups,lanes_per_group=arms,complete_paired_native_leaves_exact=True,prepared_physical_lanes_checked=count,
        original_paths_anchored_to_own_pose=True,online_initial_latch_false=True,native_only_modules=loaded,
        no_old_trajectory_reuse=True,new_physics_steps=0,optimizer_updates=0,
        data_role='Two prepared physical initial states, not independent training seeds.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def audit_samestate_initial(root,p,carry,output,jax,initial):
    import numpy as np
    source=Path(p['samestate_source']['source']);file=source/'evaluation/task1_complete_carry.pkl'
    if hashlib.sha256(file.read_bytes()).hexdigest()!=p['input_sha256'][str(file)]:raise ValueError('same-state source carry SHA drift')
    with file.open('rb') as f:saved=pickle.load(f)
    cases=json.loads((root/'inventory.json').read_text());actual,ta=jax.tree.flatten(carry);original,tb=jax.tree.flatten(saved)
    if len(cases)!=72 or ta!=tb:raise ValueError('complete72 source-copy structure required')
    for a,b in zip(actual,original):
        a=np.asarray(a);b=np.asarray(b)
        if a.dtype!=b.dtype:raise ValueError('source-copy dtype drift')
        for lane,c in enumerate(cases):np.testing.assert_array_equal(a[lane],b[c['source_index']])
    fingerprints={};groups={}
    for c in p['protocol']['source_rows']:
        selected=jax.tree.map(lambda x:np.asarray(x)[c['source_index']],saved);sha=hashlib.sha256()
        for path,value in jax.tree_util.tree_flatten_with_path(selected)[0]:
            a=np.asarray(value);sha.update(jax.tree_util.keystr(path).encode());sha.update(str(a.dtype).encode());sha.update(a.tobytes())
        if sha.hexdigest()!=c['fingerprint']:raise ValueError('complete source fingerprint mismatch')
        fingerprints[str(c['source_index'])]=sha.hexdigest()
    state=carry.state;np.testing.assert_array_equal(state.tick,np.full(72,200));np.testing.assert_array_equal(state.alpha,np.ones(72))
    for flag in (state.fault,state.physical.failed,state.domain_exit):
        if np.any(flag):raise ValueError('same-state source is stopped/invalid')
    with np.load(root/'evaluation/source_features.npz') as z:features=z['native_observation' if p.get('mode')=='h75_user_late_state192' else 'native_observation_user1']
    if features.shape!=(72,351) or not np.isfinite(features).all():raise ValueError('finite72x351 source features required')
    np.testing.assert_allclose(features[:,336],[c.get('source_user_alpha',1.) for c in cases],atol=1e-6,rtol=0)
    for lane,c in enumerate(cases):groups.setdefault(c['source_index'],[]).append(lane)
    if len(groups)!=4 or any(len(ids)!=18 for ids in groups.values()):raise ValueError('four source groups18 copies required')
    for ids in groups.values():
        for lane in ids[1:]:np.testing.assert_array_equal(features[lane],features[ids[0]])
    with np.load(root/'evaluation/task1_host_state.npz') as host,np.load(source/'evaluation/task1_host_state.npz') as old:
        indices=[c['source_index'] for c in cases]
        for field in old.files:np.testing.assert_array_equal(host[field],old[field][indices])
    receipt=json.loads((root/'evaluation/shape_receipt.json').read_text())
    if not receipt.get('all_source_copy_leaves_exact') or receipt.get('source_fingerprints')!=fingerprints:raise ValueError('source-copy receipt mismatch')
    result=dict(initial_carry_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),source_carry_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_groups=4,copies_per_source=18,source_fingerprints=fingerprints,
        all_complete_source_copy_leaves_exact=True,all_host_state_copies_exact=True,source_features_exact_within_group=True,
        initial_filtered=np.asarray(state.filtered).tolist(),initial_offsets=np.asarray(state.offsets).tolist(),
        new_physics_steps=0,optimizer_updates=0,data_role='Four familiar physical ancestor states;72 conditional suffixes are not independent source states.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def assert_late_planner_state_preserved(before,after,jax):
    """Task6 pure geometry binding may change projection/follower only."""
    import numpy as np
    aa,ta=jax.tree_util.tree_flatten_with_path(before);bb,tb=jax.tree_util.tree_flatten_with_path(after)
    if ta!=tb:raise ValueError('late planner changed source tree structure')
    for (path,a),(other,b) in zip(aa,bb):
        key=jax.tree_util.keystr(path)
        if key!=jax.tree_util.keystr(other):raise ValueError('late planner changed leaf identity')
        if key.startswith('.state.projection.') or key.startswith('.state.follower.'):continue
        if np.asarray(a).dtype!=np.asarray(b).dtype:raise ValueError('late planner changed retained dtype '+key)
        np.testing.assert_array_equal(a,b,err_msg='late planner changed retained native state '+key)


def audit_bank144_initial(root,p,carry,output,jax,initial):
    import numpy as np
    late=p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192');count=p['protocol'].get('expected_cases',144);group_count=len(p['protocol']['source_groups']);copies=count//group_count;source_tick=round(p['protocol'].get('source_task_s',1.)/.005)
    seal=json.loads((root/'sources/seal.json').read_text())
    if seal['plan_sha256']!=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest():raise ValueError('source pool plan identity mismatch')
    for file,sha in seal['files'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:raise ValueError('source pool seal drift')
    with (root/'sources/complete_carry.pkl').open('rb') as f:saved=pickle.load(f)
    records=json.loads((root/'sources/records.json').read_text());cases=json.loads((root/'inventory.json').read_text())
    if len(records)!=group_count or len(cases)!=count:raise ValueError('declared source pool/copies required')
    actual,ta=jax.tree.flatten(carry);original,tb=jax.tree.flatten(saved)
    if ta!=tb:raise ValueError('source pool copy structure drift')
    for a,b in zip(actual,original):
        a=np.asarray(a);b=np.asarray(b)
        if a.dtype!=b.dtype:raise ValueError('source pool dtype drift')
        for lane,c in enumerate(cases):np.testing.assert_array_equal(a[lane],b[c['source_index']])
    fingerprints={};loaded={}
    for record,declared in zip(records,p['protocol']['source_groups']):
        if any(record[k]!=v for k,v in declared.items()):raise ValueError('source pool metadata drift')
        lane=record['source_index'];selected=jax.tree.map(lambda x:np.asarray(x)[lane],saved);sha=hashlib.sha256()
        for path,value in jax.tree_util.tree_flatten_with_path(selected)[0]:
            a=np.asarray(value);sha.update(jax.tree_util.keystr(path).encode());sha.update(str(a.dtype).encode());sha.update(a.tobytes())
        if sha.hexdigest()!=record['source_fingerprint']:raise ValueError('assembled source fingerprint mismatch')
        fingerprints[str(lane)]=sha.hexdigest()
        file=record['source_file']
        if file not in loaded:
            with Path(file).open('rb') as f:loaded[file]=pickle.load(f)
        original_source=loaded[file].native if hasattr(loaded[file],'native') else loaded[file]
        ancestor=jax.tree.map(lambda x:np.asarray(x)[record['source_row']],original_source)
        aa,at=jax.tree.flatten(selected);bb,bt=jax.tree.flatten(ancestor)
        if at!=bt:raise ValueError('source ancestor structure mismatch')
        for a,b in zip(aa,bb):
            if np.asarray(a).dtype!=np.asarray(b).dtype:raise ValueError('source ancestor dtype mismatch')
            np.testing.assert_array_equal(a,b)
    state=carry.state;np.testing.assert_array_equal(state.tick,np.full(count,source_tick));np.testing.assert_array_equal(state.alpha,[c['alpha'] for c in cases])
    np.testing.assert_array_equal(state.physical.physical_tick,np.full(count,700+source_tick))
    if np.any(state.fault|state.physical.failed):raise ValueError('source pool stopped state')
    with np.load(root/'evaluation/source_features.npz') as z:features=z['native_observation' if p.get('mode')=='h75_user_late_state192' else 'native_observation_user1']
    if features.shape!=(count,351) or not np.isfinite(features).all():raise ValueError('finite144x351 source features required')
    np.testing.assert_allclose(features[:,336],[c.get('source_user_alpha',1.) for c in cases],atol=1e-6,rtol=0);groups={}
    for lane,c in enumerate(cases):groups.setdefault(c['source_index'],[]).append(lane)
    if len(groups)!=group_count or any(len(ids)!=copies for ids in groups.values()):raise ValueError('declared source copy groups required')
    for ids in groups.values():
        for lane in ids[1:]:np.testing.assert_array_equal(features[lane],features[ids[0]])
    with np.load(root/'evaluation/task1_host_state.npz') as actual,np.load(root/'sources/host_state.npz') as source:
        for field in source.files:np.testing.assert_array_equal(actual[field],source[field][[c['source_index'] for c in cases]])
        if not late and np.any(source['ever_arrived']|source['replanned']):raise ValueError('unarrived original-path source pool required')
    if p['protocol'].get('source_initialization')=='prepared_bank_all_prefixes' and not late:
        bankfile=Path(p['initial_source']['prepared_bank'])
        if hashlib.sha256(bankfile.read_bytes()).hexdigest()!=p['initial_source']['prepared_bank_sha256']:raise ValueError('prepared confirmation bank drift')
        with bankfile.open('rb') as f:prepared=pickle.load(f)
        for record in records:
            if record['split']!=('late_state_development' if late else 'new_confirmation') or record['bank_index'] not in (2,5):raise ValueError('confirmation source identity mismatch')
            file=record['prefix_initial_carry_file']
            if file not in seal['files']:raise ValueError('prefix initialization not sealed')
            if file not in loaded:
                with Path(file).open('rb') as f:loaded[file]=pickle.load(f)
            prefix_initial=jax.tree.map(lambda x:np.asarray(x)[record['prefix_initial_source_row']],loaded[file])
            original_bank=jax.tree.map(lambda x:np.asarray(x)[record['bank_index']],prepared)
            sha=hashlib.sha256()
            for path,value in jax.tree_util.tree_flatten_with_path(original_bank)[0]:
                a=np.asarray(value);sha.update(jax.tree_util.keystr(path).encode());sha.update(str(a.dtype).encode());sha.update(a.tobytes())
            if sha.hexdigest()!=record['prepared_state_fingerprint'] or sha.hexdigest()!=p['protocol']['bank_state_fingerprints'][str(record['bank_index'])]:raise ValueError('prepared ancestor fingerprint mismatch')
            # Exact original task reset changes only raw/reference pose/yaw fields.
            for field in ('data','controller','actuator','physical_tick','governor','failed'):
                actual_leaves=jax.tree.leaves(getattr(prefix_initial.state.physical,field));expected_leaves=jax.tree.leaves(getattr(original_bank,field))
                if len(actual_leaves)!=len(expected_leaves):raise ValueError('prepared ancestor leaf structure mismatch')
                for a,b in zip(actual_leaves,expected_leaves):
                    if np.asarray(a).dtype!=np.asarray(b).dtype:raise ValueError('prepared ancestor dtype mismatch')
                    np.testing.assert_array_equal(a,b)
            if int(prefix_initial.state.tick)!=0 or int(prefix_initial.state.physical.physical_tick)!=700:raise ValueError('prefix initialization clock mismatch')
    if late:
        prefile=root/'prefix/initial_carry.pkl'
        if str(prefile) not in seal['files']:raise ValueError('late imported preplanner carry is not sealed')
        with prefile.open('rb') as f:preplanner=pickle.load(f)
        for record in records:
            file=record['original_source_file']
            if file not in loaded:
                with Path(file).open('rb') as f:loaded[file]=pickle.load(f)
            ancestor=loaded[file].native if hasattr(loaded[file],'native') else loaded[file]
            before=jax.tree.map(lambda x:np.asarray(x)[record['source_index']],preplanner)
            original=jax.tree.map(lambda x:np.asarray(x)[record['original_source_row']],ancestor)
            aa,ta=jax.tree.flatten(before);bb,tb=jax.tree.flatten(original)
            if ta!=tb:raise ValueError('late imported original tree differs')
            for a,b in zip(aa,bb):
                if np.asarray(a).dtype!=np.asarray(b).dtype:raise ValueError('late imported original dtype differs')
                np.testing.assert_array_equal(a,b)
            after=jax.tree.map(lambda x:np.asarray(x)[record['source_index']],saved)
            assert_late_planner_state_preserved(before,after,jax)
        with (root/'sources/executed_paths.pkl').open('rb') as f:source_paths=pickle.load(f)
        for record in records:
            path=jax.tree.map(lambda x:np.asarray(x)[record['source_index']],source_paths)
            with np.load(root/'sources'/f"executed_source_{record['source_index']:02d}_path.npz") as z:
                for key in ('s','xy','heading','curvature','turn_end','goal'):np.testing.assert_array_equal(z[key],getattr(path,key))
    prefix_ticks=seal['active_prefix_lower_ticks']
    if prefix_ticks!=p['protocol']['maximum_prefix_lower_ticks'] or seal['allocated_prefix_lower_slots']!=prefix_ticks:raise ValueError('exact declared prefix ticks required')
    result=dict(initial_carry_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_groups=group_count,copies_per_source=copies,source_task_tick=source_tick,source_physical_tick=700+source_tick,late_original_import_all_leaves_exact=late,task6_planner_preserved_non_geometry_state=late,all_complete_source_copy_leaves_exact=True,all_ancestor_provenance_exact=True,source_fingerprints=fingerprints,
        initial_filtered=np.asarray(state.filtered).tolist(),initial_offsets=np.asarray(state.offsets).tolist(),
        source_features_exact_within_group=True,all_host_state_copies_exact=True,active_new_prefix_lower_ticks=prefix_ticks,
        new_physics_steps=0,optimizer_updates=0,heldout_suffix_labels_accessed=False,
        data_role='Source identity only; '+p['data_role'])
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit(args.root.resolve(),args.output.resolve())


def validate_user_preference_cache(source,*,config,models,protocol,initial_source=None,replanning=None):
    """Authenticate96 saved trajectories and the fixed-half alpha independence contract."""
    import ast
    import numpy as np
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if old['mode']!='tolerance_broaden120' or sha(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed broaden120 cache required')
    for key,value in [('config',config),('models',models)]+([('initial_source',initial_source)] if initial_source is not None else [])+([('replanning',replanning)] if replanning is not None else []):
        if old[key]!=value:raise ValueError('cached identity differs '+key)
    for key in ('bank_indices','bank_state_fingerprints','routes','forces_N','onsets_s','zero_onset_s','duration_s','rule'):
        if old['protocol'][key]!=protocol[key]:raise ValueError('cached condition differs '+key)
    status=json.loads((source/'evaluation/status.json').read_text());receipt=json.loads((source/'analysis/receipt.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=120 or receipt['all_cases']!=120 or receipt['plan_sha256']!=sha(source/'plan.json'):raise ValueError('complete authenticated broaden analysis required')
    native=Path(old['native_runtime_root'])/'sttw_control/path_command_env.py';tree=ast.parse(native.read_text());uses=[]
    for method in [x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]:
        for node in ast.walk(method):
            if isinstance(node,ast.Attribute) and node.attr=='alpha' and isinstance(node.value,ast.Name):uses.append((method.name,node.value.id))
    if sorted(uses)!=[('observation','s'),('tick','s')]:raise ValueError('native state alpha has unreviewed physical dependencies')
    tick=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='tick')
    alpha_calls=[x for x in ast.walk(tick) if isinstance(x,ast.Call) and any(k.arg=='alpha' and isinstance(k.value,ast.Attribute) and k.value.attr=='alpha' for k in x.keywords)]
    if len(alpha_calls)!=1 or not isinstance(alpha_calls[0].func,ast.Name) or alpha_calls[0].func.id!='costs':raise ValueError('state alpha must only feed reward in tick')
    identity={str(native):sha(native)}
    for name in ('plan.json','seal.json','inventory.json','evaluation/case_index.json','evaluation/status.json','evaluation/manifest.json','analysis/results.json','analysis/receipt.json','q/receipt.json'):
        identity[str(source/name)]=sha(source/name)
    index=json.loads((source/'evaluation/case_index.json').read_text());rows=json.loads((source/'analysis/results.json').read_text())
    if len(rows)!=120 or len(index)!=120 or [x['case_id'] for x in rows]!=[x['case_id'] for x in index]:raise ValueError('complete ordered cache inventory required')
    selected=[(i,r) for i,r in enumerate(rows) if r['mode'] in ('temporal_rank_last','task_guard_only','fixed1','fixed_half')]
    if len(selected)!=96:raise ValueError('96 cached physical cases required')
    for i,row in selected:
        if any(row.get(k)!=v for k,v in index[i].items()):raise ValueError('cached raw/index drift')
        f=Path(row['raw_file'])
        if sha(f)!=row['raw_sha256']:raise ValueError('cached raw drift')
        identity[str(f)]=row['raw_sha256']
        with np.load(f) as z:
            n=int(np.asarray(z['active_tick'],bool).sum())
            if n!=row['active_lower_ticks']:raise ValueError('cached endpoint mismatch')
            np.testing.assert_array_equal(z['alpha_user'][:n],1.)
            if row['mode']=='fixed_half':
                np.testing.assert_array_equal(z['alpha_exec'][:n],.5)
                np.testing.assert_allclose(z['query_observation'][:n,336],.5,atol=1e-6,rtol=0)
                np.testing.assert_allclose(z['composed_latent'][:n],z['coord_latents'][:n,1],atol=1e-6,rtol=0)
        for f in [source/'evaluation'/(row['path_key']+'_path.npz')]+([source/'evaluation'/f'replanned_path_{i:02d}.npz'] if row.get('replan_task_tick') is not None else []):identity[str(f)]=sha(f)
    return dict(source=str(source),identity=identity,cached_cases=96,total_records=216,reuse_physics_only=True,raw_user_alpha=1.,effective_cached_fixed_half_user_alpha=.5,alpha_half_reuse_proof=dict(native_environment=str(native),native_environment_sha256=sha(native),state_alpha_uses=[list(x) for x in uses],physical_alpha_dependency=False,fixed_query_alpha=.5,all24_fixed_half_queries_checked=True,explanation='state.alpha feeds observation and costs only; fixed actor queries overwrite feature336; frozen predictor features exclude336. Original raw alpha1/reward retained; effectiveuser.5 rewards independently rebuilt.'))


def validate_prior_online_cache(source,*,config,models,protocol,initial_source=None,replanning=None):
    """Authenticate the complete prior user experiment, including its 96 ancestors."""
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if old['mode']!='user_preference120' or sha(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed user preference cache required')
    for key,value in [('config',config),('models',models)]+([('initial_source',initial_source)] if initial_source is not None else [])+([('replanning',replanning)] if replanning is not None else []):
        if old[key]!=value:raise ValueError('cached identity differs '+key)
    for key in ('bank_indices','bank_state_fingerprints','routes','forces_N','onsets_s','zero_onset_s','duration_s','rule'):
        if old['protocol'][key]!=protocol[key]:raise ValueError('cached condition differs '+key)
    ancestor=validate_user_preference_cache(old['comparison_source']['source'],config=config,models=models,protocol=protocol,initial_source=initial_source,replanning=replanning)
    if ancestor!=old['comparison_source']:raise ValueError('ancestor cache identity drift')
    receipt=json.loads((source/'analysis/receipt.json').read_text());status=json.loads((source/'evaluation/status.json').read_text())
    rows=json.loads((source/'analysis/results.json').read_text());index=json.loads((source/'evaluation/case_index.json').read_text())
    if receipt['all_cases']!=216 or receipt['plan_sha256']!=sha(source/'plan.json') or status['phase']!='completed' or status['completed_cases']!=120 or len(rows)!=216 or len(index)!=120:raise ValueError('complete216 user comparison required')
    identity=dict(ancestor['identity'])
    for name in ('plan.json','seal.json','inventory.json','evaluation/case_index.json','evaluation/status.json','evaluation/manifest.json','evaluation/replanning_decisions.json','analysis/results.json','analysis/receipt.json','q/receipt.json'):
        identity[str(source/name)]=sha(source/name)
    for i,row in enumerate(rows):
        if i<120 and any(row.get(k)!=v for k,v in index[i].items()):raise ValueError('cached raw/index drift')
        if row['effective_user_alpha'] not in (0.,.5,1.) or row['method'] not in ('last','guard','fixed'):raise ValueError('cached effective identity drift')
        if sha(row['raw_file'])!=row['raw_sha256']:raise ValueError('cached raw drift')
        identity[row['raw_file']]=row['raw_sha256']
        if i<120:
            for f in [source/'evaluation'/(row['path_key']+'_path.npz')]+([source/'evaluation'/f'replanned_path_{i:02d}.npz'] if row.get('replan_task_tick') is not None else []):identity[str(f)]=sha(f)
    return dict(source=str(source),identity=identity,cached_cases=216,total_records=288,reuse_physics_only=True,preserve_effective_user_identity=True)


def validate_task_aware_cache(source,*,config,models,protocol,initial_source=None,replanning=None):
    """Authenticate all288 prior-online records and their nested216/96 ancestry."""
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if old['mode']!='user_prior_online72' or sha(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed prior online cache required')
    for key,value in [('config',config),('models',models)]+([('initial_source',initial_source)] if initial_source is not None else [])+([('replanning',replanning)] if replanning is not None else []):
        if old[key]!=value:raise ValueError('cached identity differs '+key)
    for key in ('bank_indices','bank_state_fingerprints','routes','forces_N','onsets_s','zero_onset_s','duration_s','rule'):
        if old['protocol'][key]!=protocol[key]:raise ValueError('cached condition differs '+key)
    ancestor=validate_prior_online_cache(old['comparison_source']['source'],config=config,models=models,protocol=protocol,initial_source=initial_source,replanning=replanning)
    if ancestor!=old['comparison_source']:raise ValueError('ancestor cache identity drift')
    receipt=json.loads((source/'analysis/receipt.json').read_text());status=json.loads((source/'evaluation/status.json').read_text())
    rows=json.loads((source/'analysis/results.json').read_text());index=json.loads((source/'evaluation/case_index.json').read_text())
    if receipt['all_cases']!=288 or receipt['plan_sha256']!=sha(source/'plan.json') or status['phase']!='completed' or status['completed_cases']!=72 or len(rows)!=288 or len(index)!=72:raise ValueError('complete288 prior comparison required')
    identity=dict(ancestor['identity'])
    for name in ('plan.json','seal.json','inventory.json','evaluation/case_index.json','evaluation/status.json','evaluation/manifest.json','evaluation/replanning_decisions.json','analysis/results.json','analysis/receipt.json','q/receipt.json'):
        identity[str(source/name)]=sha(source/name)
    for i,row in enumerate(rows):
        if i<72 and any(row.get(k)!=v for k,v in index[i].items()):raise ValueError('cached raw/index drift')
        if row['effective_user_alpha'] not in (0.,.5,1.) or row['method'] not in ('last','guard','fixed','prior_last'):raise ValueError('cached effective identity drift')
        if sha(row['raw_file'])!=row['raw_sha256']:raise ValueError('cached raw drift')
        identity[row['raw_file']]=row['raw_sha256']
        if i<72:
            for f in [source/'evaluation'/(row['path_key']+'_path.npz')]+([source/'evaluation'/f'replanned_path_{i:02d}.npz'] if row.get('replan_task_tick') is not None else []):identity[str(f)]=sha(f)
    groups={}
    for row in rows:groups.setdefault((row['effective_user_alpha'],row['bank_index'],row['route'],row['event']),[]).append(row['method'])
    if len(groups)!=72 or any(sorted(x)!=['fixed','guard','last','prior_last'] for x in groups.values()):raise ValueError('complete four-method cached groups required')
    return dict(source=str(source),identity=identity,cached_cases=288,total_records=360,reuse_physics_only=True,preserve_effective_user_identity=True)


def _validate_frozen_panel_cache(source,*,config,models,protocol,initial_source=None,replanning=None,source_count=200,total_records=260,source_variant=None):
    """Reuse the complete immutable200 full-input controls; no new baseline physics."""
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if old.get('mode')!='frozen_confirmation200' or old['protocol'].get('variant')!=source_variant or sha(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed frozen200 cache required')
    for key,value in [('config',config),('models',models)]+([('initial_source',initial_source)] if initial_source is not None else [])+([('replanning',replanning)] if replanning is not None else []):
        if old[key]!=value:raise ValueError('speed-query cache identity differs '+key)
    for key in ('bank_indices','bank_state_fingerprints','routes','forces_N','onsets_s','zero_onset_s','duration_s','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','coordination','rule','user_preference'):
        if old['protocol'][key]!=protocol[key]:raise ValueError('speed-query cached condition differs '+key)
    receipt=json.loads((source/'analysis/receipt.json').read_text());status=json.loads((source/'evaluation/status.json').read_text());rows=json.loads((source/'analysis/results.json').read_text());index=json.loads((source/'evaluation/case_index.json').read_text())
    if receipt['all_cases']!=source_count or receipt['plan_sha256']!=sha(source/'plan.json') or status.get('phase')!='completed' or status.get('completed_cases')!=source_count or len(rows)!=source_count or len(index)!=source_count:raise ValueError('complete full-input200 analysis required')
    identity={}
    for name in ('plan.json','seal.json','inventory.json','evaluation/case_index.json','evaluation/status.json','evaluation/manifest.json','evaluation/replanning_decisions.json','evaluation/shape_receipt.json','analysis/results.json','analysis/receipt.json','q/receipt.json'):
        identity[str(source/name)]=sha(source/name)
    qreceipt=json.loads((source/'q/receipt.json').read_text())
    if qreceipt.get('feature_mask_keep_indices') is not None or qreceipt['arm_order']!=['prior_rank_last']:raise ValueError('cached comparator must use original unmasked prior q')
    for name,expected in qreceipt['files_sha256'].items():
        file=source/'q'/name
        if sha(file)!=expected:raise ValueError('cached full q changed')
        identity[str(file)]=expected
    groups={};ids=set()
    for i,row in enumerate(rows):
        if row['case_id'] in ids or any(row.get(k)!=v for k,v in index[i].items()):raise ValueError('full200 raw/index identity differs')
        ids.add(row['case_id'])
        if sha(row['raw_file'])!=row['raw_sha256']:raise ValueError('full200 cached raw changed')
        identity[row['raw_file']]=row['raw_sha256']
        for file in [source/'evaluation'/(row['path_key']+'_path.npz')]+([source/'evaluation'/f'replanned_path_{i:02d}.npz'] if row.get('replan_task_tick') is not None else []):identity[str(file)]=sha(file)
        groups.setdefault((row['bank_index'],row['route'],row['event']),[]).append((row['effective_user_alpha'],row['method']))
    expected={(u,m) for u in (0.,.5,1.) for m in ('task_aware','guard','fixed')}|{(1.,'speedcap')}
    if len(groups)!=source_count//10 or any(len(v)!=10 or set(v)!=expected for v in groups.values()):raise ValueError('full200 ten-arm condition coverage differs')
    return dict(source=str(source),identity=identity,cached_cases=source_count,total_records=total_records,reuse_physics_only=True,preserve_effective_user_identity=True,source_data_role=old['data_role'],reused_frozen_q_audit_receipt_sha256=sha(source/'analysis/receipt.json'))


def validate_speed_query_cache(source,*,config,models,protocol,initial_source=None,replanning=None):
    return _validate_frozen_panel_cache(source,config=config,models=models,protocol=protocol,initial_source=initial_source,replanning=replanning)


def validate_guard_fallback_cache(source,*,config,models,protocol,initial_source=None,replanning=None):
    stage=protocol.get('stage')
    if stage not in ('extension','regression'):raise ValueError('unknown guard fallback cache stage')
    extension=stage=='extension'
    checked=_validate_frozen_panel_cache(source,config=config,models=models,protocol=protocol,initial_source=initial_source,replanning=replanning,source_count=240 if extension else 200,total_records=312 if extension else 260,source_variant='geometry_duration_extension_v1' if extension else None)
    if extension and json.loads((Path(source)/'plan.json').read_text())['protocol']['route_contexts']!=protocol['route_contexts']:raise ValueError('guard fallback cached route contexts differ')
    checked.update(stage=stage,old_control_audit='Original sealed analysis receipt and original source core inherited; new guard fallback core not applied to old trajectories')
    return checked


def guard_fallback_stage1_gate(root):
    """One declared promotion gate: any paired goal/contact rescue, no tuning."""
    root=Path(root).resolve();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    plan=json.loads((root/'plan.json').read_text());status=json.loads((root/'evaluation/status.json').read_text());receipt=json.loads((root/'analysis/receipt.json').read_text())
    if plan.get('mode')!='guard_fallback_evaluation' or plan['protocol'].get('stage')!='extension' or plan['cases']!=72 or sha(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed guard fallback extension stage required')
    if status.get('phase')!='completed' or status.get('completed_cases')!=72 or status['active_lower_ticks']>144000 or receipt['all_cases']!=312 or receipt['plan_sha256']!=sha(root/'plan.json'):raise ValueError('complete stage1 physical/analysis evidence required')
    rows=json.loads((root/'analysis/results.json').read_text());index=json.loads((root/'evaluation/case_index.json').read_text());declared=json.loads((root/'inventory.json').read_text())
    if len(declared)!=72 or {r['case_id'] for r in declared}!={r['case_id'] for r in index}:raise ValueError('stage1 actual index differs from declared inventory')
    if len(rows)!=312 or len(index)!=72 or len({r['case_id'] for r in rows})!=312:raise ValueError('complete unique stage1 inventory required')
    key=lambda r:(r['effective_user_alpha'],r['bank_index'],r['route'],r['event'])
    new={key(r):r for r in rows if r['method']=='guard_degraded'};old={key(r):r for r in rows if r['method']=='task_aware'}
    if len(new)!=72 or set(new)!=set(old) or {r['case_id'] for r in new.values()}!={r['case_id'] for r in index}:raise ValueError('stage1 must pair all72 with oldtaskaware')
    details=[];identity={str(root/name):sha(root/name) for name in ('plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/case_index.json','analysis/results.json','analysis/receipt.json')}
    for k,n in new.items():
        o=old[k];a=n['horizons']['10'];b=o['horizons']['10']
        if a['goal_arrived'] is None or b['goal_arrived'] is None or a['physical_outcome'] not in ('survived','contact') or b['physical_outcome'] not in ('survived','contact'):raise ValueError('unknown stage1 outcomes cannot authorize promotion')
        for row in (n,o):
            if sha(row['raw_file'])!=row['raw_sha256']:raise ValueError('stage1 paired raw identity changed')
            identity[row['raw_file']]=row['raw_sha256']
        details.append(dict(user=k[0],bank=k[1],route=k[2],event=k[3],goal_gain=bool(a['goal_arrived'] and not b['goal_arrived']),goal_loss=bool(b['goal_arrived'] and not a['goal_arrived']),contact_rescue=b['physical_outcome']=='contact' and a['physical_outcome']=='survived',new_contact=a['physical_outcome']=='contact' and b['physical_outcome']=='survived'))
    gains=sum(r['goal_gain'] for r in details);rescues=sum(r['contact_rescue'] for r in details)
    return dict(source=str(root),allowed=bool(gains or rescues),total_goal_gains=gains,total_contact_rescues=rescues,total_goal_losses=sum(r['goal_loss'] for r in details),total_new_contacts=sum(r['new_contact'] for r in details),compared_to='oldtaskaware only, not guard',identity=identity,paired_cases=72,details=details,rule='Any goal or contact rescue authorizes fixedstage2 even if losses elsewhere; zero rescues stops candidate')
