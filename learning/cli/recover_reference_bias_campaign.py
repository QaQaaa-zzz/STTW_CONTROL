"""Authenticate/recover first complete physical batch, no simulation or retry."""
import argparse
import ast
import json
import shutil
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json, digest
from sttw_control.reference_bias_runtime import score_raw, encode_objective
from sttw_control.reference_bias_optimization import bias_targets


def recover(root):
    root=root.resolve();out=root/'production';archive=out/'audit_error_recovery'
    archive.mkdir(exist_ok=False)
    state=json.loads((out/'status.json').read_text())
    if state['phase']!='error' or 'reference_bias' not in (out/'traceback.txt').read_text():
        raise ValueError('identified interpolation-audit error required')
    if state['completed_searches'] or state['evaluation_cases'] or state['recorded_physical_batches']!=1:
        raise ValueError('only first-batch physical recovery allowed')
    for name in ['status.json','dispatch_status.json','traceback.txt','execute_stdout.log','launch_receipt.json']:
        shutil.copy2(out/name,archive/name)
    contract=json.loads((out/'contract.json').read_text())
    if digest(out/'contract.json')!=json.loads((out/'seal.json').read_text())['contract_sha256']:
        raise ValueError('original contract changed')
    changes={}
    allowed={'reference_bias_runtime.py','reference_bias_campaign.py'}
    for file,sha in contract['source_sha256'].items():
        if digest(file)!=sha:
            if Path(file).name not in allowed:raise ValueError('unreviewed source change '+file)
            old=next((out/'source').glob('*_'+Path(file).name))
            if digest(old)!=sha:raise ValueError('original source snapshot mismatch')
            if Path(file).name=='reference_bias_runtime.py':
                def init(file):
                    tree=ast.parse(Path(file).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BiasRuntime')
                    return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
                if init(old)!=init(file):raise ValueError('physical runtime factory changed')
            changes[file]=dict(old_sha256=sha,new_sha256=digest(file))
    added=Path('learning/src/sttw_control/reference_bias_float32_audit.py').resolve()
    changes[str(added)]=dict(old_sha256=None,new_sha256=digest(added))
    (archive/'source').mkdir()
    for file in changes:shutil.copy2(file,archive/'source'/Path(file).name)
    batch=out/'searches/search_00/generation_00'
    previous=json.loads((batch/'status.json').read_text())
    if previous['phase']!='physical_completed' or previous['completed_physical_cases']!=16:
        raise ValueError('complete original physical16 required')
    shutil.copy2(batch/'status.json',archive/'batch_original_status.json')
    files=sorted(batch.glob('chunk_*.npz'));chunks=[]
    for file in files:
        with np.load(file) as z:chunks.append({k:z[k] for k in z.files})
    combined={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    if combined['active_tick'].shape!=(2000,16):raise ValueError('complete original batch required')
    inv=json.loads((batch/'inventory.json').read_text());plan=json.loads((root/'plan.json').read_text())
    cfg=json.loads((root/'common_environment_config.json').read_text());rows=[]
    for lane,(case,vector) in enumerate(zip(inv['cases'],inv['vectors'])):
        file=batch/'cases'/f'case_{lane:02d}.npz'
        with np.load(file) as z:raw={k:z[k] for k in z.files}
        if set(raw)!=set(combined):raise ValueError('raw schema mismatch')
        for field in raw:np.testing.assert_array_equal(raw[field],combined[field][:,lane])
        c=plan['contexts'][case['context']]
        if digest(c['path'])!=c['path_sha256']:raise ValueError('original path changed')
        with np.load(c['path']) as z:path={k:z[k] for k in z.files}
        row=score_raw(raw,case,cfg,path,targets=bias_targets(vector),positions=c['bias_positions_m'],feedback_spec=plan['feedback_spec'])
        row.update(raw_file=str(file),raw_sha256=digest(file),vector=vector);rows.append(encode_objective(row))
    if sum(r['physical_ticks'] for r in rows)!=32000 or previous['allocated_lower_slots']!=32000:
        raise ValueError('original batch cost mismatch')
    atomic_json(batch/'results.json',rows)
    atomic_json(batch/'audit_recovery.json',dict(results_sha256=digest(batch/'results.json'),
        original_status=previous,new_physics_steps=0,all_case_fields_equal_original_chunks=True,
        chunk_sha256={str(f):digest(f) for f in files},
        scope='Only interpolation audit tolerance1e-7 to1e-6; original control source unchanged; other audits unchanged.'))
    atomic_json(batch/'status.json',dict(previous,phase='completed',completed_cases=16,audit_recovered=True))
    atomic_json(out/'recovery_binding.json',dict(original_contract_sha256=digest(out/'contract.json'),original_status=state,
        source_changes=changes,scope='Explicit audit-only continuation; deterministic optimizer replays recovered first callback without integrating it twice. Original error/exit1 preserved. No extra candidates or budget.'))
    print('first16 authenticated and recovered; original contract/error preserved; zero new physics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);recover(p.parse_args().root)
