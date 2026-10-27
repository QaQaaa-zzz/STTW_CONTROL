"""CPU-only input and Actor-tree audit on an isolated archived runtime.

No environment construction, source state reset or numerical integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode=True


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args();root=a.root.resolve();plan=json.loads((root/'plan.json').read_text())
    seal=json.loads((root/'seal.json').read_text())
    if digest(root/'plan.json')!=seal['plan_sha256'] or digest(root/'inventory.json')!=seal['inventory_sha256']:
        raise ValueError('preparation identity changed')
    if any(n=='sttw_control' or n.startswith('sttw_control.') for n in sys.modules):
        raise ValueError('fresh process required, no modern control imports')
    native=Path(plan['runtime_root']);sys.path.insert(0,str(native))
    import jax
    import jax.numpy as j
    import numpy as np
    from sttw_control.path_command_policy import make_path_actor
    cases=json.loads((root/'inventory.json').read_text());records=[];actors={};shapes={}
    for route,context in plan['routes'].items():
        with Path(context['initial']).open('rb') as f:bank=pickle.load(f).state
        cfg=json.loads(Path(context['config']).read_text())
        models=json.loads(Path(context['models']).read_text())['models']
        for role in ('parent','task_only','physical_credit'):
            model=models[role]
            if digest(model['actor'])!=model['actor_sha256']:
                raise ValueError('Actor changed')
            with Path(model['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
            config=model.get('actor_config',cfg);actor=make_path_actor(config)
            shape=jax.eval_shape(actor.apply,params,j.zeros(351)).shape
            if shape!=(2,):raise ValueError('Actor observation/output interface differs')
            shapes[route+'/'+role]=list(shape)
        source_states=[]
        for case in cases:
            if case['route_group']!=route:continue
            state=jax.tree.map(lambda x:np.asarray(x[case['source_batch_slot']]),bank)
            if int(state.tick)!=0 or int(state.physical.physical_tick)!=700:
                raise ValueError('complete original task0/physical700 required')
            if bool(state.fault|state.physical.failed|state.domain_exit):
                raise ValueError('invalid initial state')
            if float(state.alpha)!=case['state_alpha']:
                raise ValueError('state alpha mismatch; no override permitted')
            source_states.append(state.replace(alpha=np.asarray(0.,dtype=np.float32)))
            records.append(dict(case_id=case['case_id'],source_batch_slot=case['source_batch_slot'],
                task_tick=0,physical_tick=700,state_alpha=float(state.alpha),
                physical_xy=np.asarray(state.physical.data.qpos[:2]).tolist()))
        # At task0 all roles and alpha slots have identical full carry except
        # alpha. This checks, rather than assumes, physical reuse of zero.
        reference=source_states[0];leaves=jax.tree.leaves(reference)
        for state in source_states[1:]:
            if jax.tree.structure(state)!=jax.tree.structure(reference):raise ValueError('initial carry schema differs')
            for actual,expected in zip(jax.tree.leaves(state),leaves):
                np.testing.assert_array_equal(actual,expected)
        actors[route]={role:models[role]['actor_sha256'] for role in ('parent','task_only','physical_credit')}
    consumed={}
    for name,module in tuple(sys.modules.items()):
        if name=='sttw_control' or name.startswith('sttw_control.'):
            f=getattr(module,'__file__',None)
            if not f:continue
            f=Path(f).resolve()
            if not f.is_relative_to(native):raise ValueError('mixed runtime import: '+str(f))
            expected=plan['runtime_expected_sources'].get(str(f))
            if expected!=digest(f):raise ValueError('unsealed consumed runtime')
            consumed[str(f)]=expected
    result=dict(status='completed',cases_checked=len(records),records=records,actors=actors,
        actor_shape_by_route_role=shapes,initial_full_carry_equality_excluding_alpha=True,
        consumed_archived_sources=consumed,analysis_script_sha256=digest(__file__),
        plan_sha256=digest(root/'plan.json'),physics_steps=0,optimizer_updates=0,
        scope='CPU state deserialization and dummy351 Actor eval_shape only; no actual observation/step/environment, GPU shape or planning rollout verified.')
    (root/'input_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('input audit completed:',len(records),'exact task0 carries; 6 Actor shapes; physics_steps=0')


if __name__=='__main__':main()
