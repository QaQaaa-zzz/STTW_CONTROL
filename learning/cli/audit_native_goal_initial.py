"""Isolated native-package initial-carry audit; never constructs an environment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode=True


def audit(root,output):
    import numpy as np
    p=json.loads((root/'plan.json').read_text());native=Path(p['native_runtime_root']).resolve()
    if 'sttw_control' in sys.modules:raise ValueError('fresh native-only audit process required')
    sys.path.insert(0,str(native))
    source=Path(__file__).resolve().parents[1]/'src/sttw_control/native_goal_continuation.py'
    spec=importlib.util.spec_from_file_location('_sttw_goal_continuation',source)
    adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
    import jax
    online=p.get('mode') in ('h75_online_scalar','h75_online_confirmation100')
    if online:
        schedule_source=Path(__file__).resolve().parents[1]/'src/sttw_control/geometric_preference_schedule.py'
        schedule_spec=importlib.util.spec_from_file_location('_sttw_online_scalar',schedule_source)
        schedule=importlib.util.module_from_spec(schedule_spec);sys.modules[schedule_spec.name]=schedule;schedule_spec.loader.exec_module(schedule)
    initial=root/'evaluation/initial_carry.pkl'
    with initial.open('rb') as f:carry=pickle.load(f)
    if online:
        if np.any(carry.latched):raise ValueError('online initial latch must be false')
        carry=carry.native
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
    if p.get('mode')=='h75_online_confirmation100':
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


def audit_confirmation_initial(root,p,carry,bank,output,jax,initial):
    import numpy as np
    cases=json.loads((root/'inventory.json').read_text());state=carry.state
    if len(cases)!=100:raise ValueError('complete100 initial cases required')
    np.testing.assert_array_equal(state.tick,np.zeros(100));np.testing.assert_array_equal(state.alpha,np.ones(100))
    np.testing.assert_array_equal(state.filtered,np.zeros((100,2)));np.testing.assert_array_equal(state.offsets,np.zeros((100,2)))
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
    leaves=jax.tree.leaves(carry)
    if len(groups)!=20 or any(len(ids)!=5 for ids in groups.values()):raise ValueError('20 paired bank/route/event groups required')
    for ids in groups.values():
        for leaf in leaves:
            for i in ids[1:]:np.testing.assert_array_equal(leaf[i],leaf[ids[0]])
    receipt=json.loads((root/'evaluation/shape_receipt.json').read_text())
    if not receipt.get('all_five_arm_reset_leaves_exact') or receipt.get('new_bank_fingerprints')!=fingerprints:raise ValueError('fivearm reset receipt mismatch')
    loaded={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if (name=='sttw_control' or name.startswith('sttw_control.')) and getattr(module,'__file__',None)}
    native=Path(p['native_runtime_root']).resolve()
    if any(not Path(file).is_relative_to(native) for file in loaded.values()):raise ValueError('mixed native initial100 modules')
    result=dict(initial_carry_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_bank_sha256=p['initial_source']['prepared_bank_sha256'],bank_indices=p['protocol']['bank_indices'],bank_fingerprints=fingerprints,
        initial_filtered=np.asarray(state.filtered).tolist(),initial_offsets=np.asarray(state.offsets).tolist(),
        initial_nominal_reference=np.asarray(state.physical.raw).tolist(),
        matched_groups=20,lanes_per_group=5,complete_paired_native_leaves_exact=True,prepared_physical_lanes_checked=100,
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
    with np.load(root/'evaluation/source_features.npz') as z:features=z['native_observation_user1']
    if features.shape!=(72,351) or not np.isfinite(features).all():raise ValueError('finite72x351 source features required')
    np.testing.assert_allclose(features[:,336],1.,atol=1e-6,rtol=0)
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit(args.root.resolve(),args.output.resolve())
