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
    online=p.get('mode')=='h75_online_scalar'
    if online:
        schedule_source=Path(__file__).resolve().parents[1]/'src/sttw_control/geometric_preference_schedule.py'
        schedule_spec=importlib.util.spec_from_file_location('_sttw_online_scalar',schedule_source)
        schedule=importlib.util.module_from_spec(schedule_spec);sys.modules[schedule_spec.name]=schedule;schedule_spec.loader.exec_module(schedule)
    initial=root/'evaluation/initial_carry.pkl'
    with initial.open('rb') as f:carry=pickle.load(f)
    if online:
        if np.any(carry.latched):raise ValueError('online initial latch must be false')
        carry=carry.native
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit(args.root.resolve(),args.output.resolve())
