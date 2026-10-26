"""Recover only the serialization-failed original16; never execute physics."""
import argparse,ast,json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.reference_bias_runtime import score_raw,encode_objective
from sttw_control.reference_bias_optimization import bias_targets


def recover(root):
    root=root.resolve();eng=root/'engineering';physical=eng/'physical';out=eng/'recovered_numeric';out.mkdir(exist_ok=False)
    original=json.loads((eng/'status.json').read_text());cost=json.loads((physical/'status.json').read_text())
    if original['phase']!='error' or cost['phase']!='physical_completed' or cost['completed_physical_cases']!=16:raise ValueError('original completed physical16 with retained error required')
    if 'JSON serializable' not in original['error']:raise ValueError('only the identified serializer error can be recovered')
    design=json.loads((eng/'design.json').read_text());plan=json.loads((root/'plan.json').read_text());cfg=json.loads((root/'common_environment_config.json').read_text())
    if digest(eng/'design.json')!=json.loads((eng/'seal.json').read_text())['design_sha256']:raise ValueError('design changed')
    if digest(root/'plan.json')!=design['source_plan_sha256']:raise ValueError('plan changed')
    changed=[]
    for file,sha in design['source_sha256'].items():
        if digest(file)!=sha:
            if Path(file).name!='reference_bias_runtime.py':raise ValueError('unreviewed source drift')
            old=next((eng/'source').glob('*_reference_bias_runtime.py'))
            if digest(old)!=sha:raise ValueError('original runtime snapshot missing')
            def node(file,name):
                tree=ast.parse(Path(file).read_text())
                if name=='score_raw':return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BiasRuntime')
                return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),include_attributes=False)
            for name in ['score_raw','__init__']:
                if node(old,name)!=node(file,name):raise ValueError('numeric scorer or physical advance factory changed')
            changed.append(dict(file=file,original_sha256=sha,current_sha256=digest(file),unchanged_AST=['score_raw','BiasRuntime.__init__'],change='JSON objective row encoding only; no integration replay'))
    chunks=[];chunk_files=[]
    for file in sorted(physical.glob('chunk_*.npz')):
        with np.load(file) as z:chunks.append({k:z[k] for k in z.files})
        chunk_files.append(dict(file=str(file),sha256=digest(file)))
    combined={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
    if combined['active_tick'].shape!=(2000,16):raise ValueError('complete physical batch shape required')
    rows=[];active=0
    for lane,(case,vector) in enumerate(zip(design['cases'],design['vectors'])):
        file=physical/'cases'/f'case_{lane:02d}.npz'
        with np.load(file) as z:raw={k:z[k] for k in z.files}
        if set(raw)!=set(combined):raise ValueError('raw schema changed')
        for field in raw:np.testing.assert_array_equal(raw[field],combined[field][:,lane])
        ctx=plan['contexts'][case['context']]
        if digest(ctx['path'])!=ctx['path_sha256']:raise ValueError('original path changed')
        with np.load(ctx['path']) as z:path={k:z[k] for k in z.files}
        row=score_raw(raw,case,cfg,path,targets=bias_targets(vector),positions=ctx['bias_positions_m'],feedback_spec=plan['feedback_spec'])
        row.update(raw_file=str(file),raw_sha256=digest(file),vector=vector);rows.append(encode_objective(row));active+=row['physical_ticks']
    if active!=cost['active_lower_ticks'] or active!=32000 or cost['allocated_lower_slots']!=32000:raise ValueError('original real cost mismatch')
    atomic_json(out/'results.json',rows)
    atomic_json(out/'recovery_receipt.json',dict(original_status=original,source_changes=changed,all_case_fields_equal_original_chunks=True,chunk_files=chunk_files,
        new_physics_steps=0,optimizer_updates=0,original_active_lower_ticks=active,scope='Original status/error/traceback remain unchanged; recovered only raw-backed numerical summaries.'))
    atomic_json(out/'status.json',dict(cost,phase='completed',recovered_cases=16,new_physics_steps=0,optimizer_updates=0))
    print('original16 recovered with all raw/chunk fields verified; zero new physics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);recover(p.parse_args().root)
