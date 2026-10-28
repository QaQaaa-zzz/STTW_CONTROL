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
    initial=root/'evaluation/initial_carry.pkl'
    with initial.open('rb') as f:carry=pickle.load(f)
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
    response=p.get('mode')=='h75_channel_response'
    count=len(cases)
    if count!=(54 if response else 80):raise ValueError('declared initial inventory size mismatch')
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
    expected_groups,expected_pairs=(36,18) if response else (20,60)
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
        task_ticks_zero=True,original_alpha_checked=True,prepared_physical_fields_checked=True,prepared_physical_lanes_checked=count,
        excluded_native_reset_fields=['raw','reference_pose','yaw_wrapped','yaw_unwrapped'],
        source_dtype_semantics='Prepared-bank leaves converted to actual loaded leaf dtypes, as in native j.asarray loader; matched-method comparisons exact without casts',
        pairing_contract='Full carry matched within route/event/state-alpha; physical bank fields checked across every lane. Different state alphas intentionally retained.',
        new_physics_steps=0,optimizer_updates=0,verification='AI array/structure audit, not independent control replay')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit(args.root.resolve(),args.output.resolve())
