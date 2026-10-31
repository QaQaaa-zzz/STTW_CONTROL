from pathlib import Path
import argparse,json,pickle,hashlib,importlib.util,shutil,time,os
import numpy as np
parser=argparse.ArgumentParser(description='Capture an actual online decision boundary using the existing native batch runtime.')
parser.add_argument('root',type=Path);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--task-s',type=float,required=True)
args=parser.parse_args();source=args.source.resolve();out=args.root.resolve()
tick=round(args.task_s*200);begin=tick//20*20;steps=(tick-begin)//4
if abs(args.task_s*200-tick)>1e-9 or tick%4 or not 1<=steps<=4:raise ValueError('choose an upper boundary strictly between saved10Hz boundaries')
block=begin//20-1;out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
plan=json.loads((source/'plan.json').read_text());assert sha(source/'plan.json')==json.loads((source/'seal.json').read_text())['plan_sha256']
for f,h in plan['input_sha256'].items():assert sha(f)==h
for f,h in plan['source_sha256'].items():assert sha(source/'source'/Path(f).name)==h
assert not any(x['task_tick']==begin for x in json.loads((source/'evaluation/replanning_decisions.json').read_text()))
plan['source_sha256']={f:sha(f) for f in plan['source_sha256']}
plan['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
budget=plan['cases']*4*steps
plan['maximum_active_lower_ticks']=budget;plan['maximum_allocated_lower_slots']=budget;plan['chunks']=1;plan['chunk_upper_steps']=[steps]
(out/'source').mkdir()
for file in plan['source_sha256']:shutil.copyfile(file,out/'source'/Path(file).name)
shutil.copyfile(__file__,out/'capture_script.py')
plan['capture_contract']=dict(source=str(source),source_plan_sha256=sha(source/'plan.json'),task_start_s=begin/200,task_end_s=tick/200,lower_slots=budget,optimizer_updates=0,source_code='Current implementation; original sealed inputs/controller unchanged. One-block physical/log comparison allows small numerical changes.')
shutil.copyfile(source/'inventory.json',out/'inventory.json');save(out/'plan.json',plan);save(out/'seal.json',dict(plan_sha256=sha(out/'plan.json')))
save(out/'status.json',dict(phase='initializing',pid=os.getpid(),allocated_lower_slots=0,active_lower_ticks=0,optimizer_updates=0))
spec=importlib.util.spec_from_file_location('native_goal',Path.cwd()/'learning/src/sttw_control/native_goal_run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
runtime=m.load(out)
import jax
import jax.numpy as j
with (source/f'evaluation/complete_carry_{block:02d}.pkl').open('rb') as f:carry=jax.tree.map(j.asarray,pickle.load(f))
with (source/f'evaluation/executed_paths_{block:02d}.pkl').open('rb') as f:paths=jax.tree.map(j.asarray,pickle.load(f))
with np.load(source/f'evaluation/host_state_{block:02d}.npz') as z:host=dict(z)
assert np.all(np.asarray(carry.state.tick)==begin) and not host['ever_arrived'].any()
save(out/'status.json',dict(phase='inflight',pid=os.getpid(),allocated_lower_slots=budget,active_lower_ticks=None,optimizer_updates=0))
started=time.monotonic();end,logs=runtime['chunk'](carry,paths,upper_steps=steps);jax.block_until_ready(end);raw=m.flatten(jax.device_get(logs))
np.savez_compressed(out/'capture_raw.npz',**raw)
checks=[]
for i,c in enumerate(json.loads((source/'evaluation/case_index.json').read_text())):
 assert sha(c['raw_file'])==c['raw_sha256']
 with np.load(c['raw_file']) as z:
  errors={k:float(np.max(np.abs(raw[k][:,i]-z[k][begin:tick]))) for k in ('actual_xy','actual_forward_speed','phi','applied_force_impulse')}
  assert max(errors.values())<.001 and np.array_equal(raw['alpha_exec'][:,i],z['alpha_exec'][begin:tick])
  assert float(end.prev_alpha[i])==float(z['coord_prev_alpha'][tick])
  next_decision=dict(previous_alpha=float(z['coord_prev_alpha'][tick]),published_alpha=float(z['alpha_exec'][tick]))
 checks.append(dict(case_id=c['case_id'],max_absolute_errors=errors,**next_decision))
assert np.all(np.asarray(end.state.tick)==tick)
with (out/'complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(end),f)
with (out/'executed_paths.pkl').open('wb') as f:pickle.dump(jax.device_get(paths),f)
np.savez_compressed(out/'host_state.npz',**host)
files={n:sha(out/n) for n in ('complete_carry.pkl','executed_paths.pkl','host_state.npz','capture_raw.npz')}
save(out/'receipt.json',dict(**plan['capture_contract'],files=files,checks=checks,actual_lower_ticks=int(raw['active_tick'].sum()),elapsed_s=time.monotonic()-started))
save(out/'status.json',dict(phase='completed',pid=os.getpid(),allocated_lower_slots=budget,active_lower_ticks=int(raw['active_tick'].sum()),optimizer_updates=0))
print('capture completed',checks)
