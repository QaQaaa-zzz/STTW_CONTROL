"""Reuse original Local300 command sampler on CPU; no physics or optimizer."""
import argparse
import ast
import inspect
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from sttw_control.controller import ControllerConfig, _system
import sttw_control.controller as controller_module
# Reuse the exact original sampler functions without importing unrelated Torch
# policy classes through lower_command's environment imports.
source=Path(inspect.getfile(controller_module)).with_name('lower_command.py')
module=ast.parse(source.read_text())
names={'local_roll_reference','local_steer_limit','sample_local_v2'}
functions=ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
exec(compile(functions,str(source),'exec'),globals())

p=argparse.ArgumentParser()
p.add_argument('--source-config',type=Path,required=True)
p.add_argument('--controller-config',type=Path,required=True)
p.add_argument('--initial-state',type=Path,required=True)
p.add_argument('--count',type=int,default=32768)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
s=json.loads(a.source_config.read_text())
c=ControllerConfig(**json.loads(a.controller_config.read_text()))
initial=np.load(a.initial_state)['previous_reference']
speeds=np.atleast_2d(initial)[:,0]
# Every initial state receives an equal number of original-sampler schedules.
per_state=a.count//len(speeds)
assert per_state*len(speeds)==a.count, 'count must be divisible by initial-state count'
sample=jax.jit(jax.vmap(lambda i,v:sample_local_v2(i,jp.int32(0),v,s,c)))
rows,slew,labels=sample(jp.arange(a.count),jp.repeat(jp.asarray(speeds),per_state))
rows,slew,labels=map(np.asarray,(rows,slew,labels))
counts=np.bincount(labels[:,0].astype(int),minlength=4)
if initial.ndim==2:
 rows=rows.reshape(len(speeds),per_state,*rows.shape[1:]);slew=slew.reshape(len(speeds),per_state,*slew.shape[1:]);labels=labels.reshape(len(speeds),per_state,*labels.shape[1:])
a.output.mkdir(parents=True,exist_ok=False)
np.savez_compressed(a.output/'commands.npz',rows=np.asarray(rows),slew=np.asarray(slew),labels=np.asarray(labels))
(a.output/'identity.json').write_text(json.dumps(dict(source_config=str(a.source_config.resolve()),
    source_config_sha256=hashlib.sha256(a.source_config.read_bytes()).hexdigest(),
    command_design=s['command_design'],controller=json.loads(a.controller_config.read_text()),
    count=a.count,initial_speeds=speeds.tolist(),schedules_per_state=per_state,family_counts=counts.tolist(),
    role='finite training command bank sampled by original sample_local_v2; GPU reset samples bank rows',
    physics_steps=0,optimizer_updates=0),indent=2))
print('Exported command families:',counts.tolist())
