"""One four-lane, forty-tick GPU pulse-wiring check; no optimization."""
import argparse,json,pickle,hashlib,os,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from sttw_control.lower_command import LowerCommandEnv
from sttw_control.runtime import configure_compilation_cache

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.config.read_text());out=a.output;out.mkdir(exist_ok=False,parents=True)
def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
save('status.json',dict(phase='initializing',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
configure_compilation_cache(cfg['shared_jax_cache']);env=LowerCommandEnv(cfg)
source=json.loads(Path(cfg['prepared_manifest']).read_text());assert env.physics.bundle.identity==source['model']
assert hashlib.sha256(Path(cfg['prepared_bank']).read_bytes()).hexdigest()==source['prepared_bank_sha256']
with np.load(source['state_domains']) as z:idx=int(np.flatnonzero(z['domain_index']==0)[0]);yaw=float(z['initial_yaw_rate'][idx])
assert source['domains'][0]['domain_id']=='nominal'
with Path(cfg['prepared_bank']).open('rb') as f:bank=pickle.load(f)
snap=jax.tree.map(lambda x:j.asarray(x[idx]),bank);one=env.reset(snap,j.int32(0),j.int32(0),initial_yaw_rate=yaw)
start=float(snap.physical_tick)*cfg['dt'];pulse=j.array([[start+.02,start+.17,0.,force,0.] for force in [0.,24.,-24.,0.]])
states=jax.tree.map(lambda x:j.stack([x]*4),one).replace(force_pulse=pulse)
save('plan.json',dict(config=str(a.config.resolve()),config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),snapshot_index=idx,physical_start_s=start,forces_N=[0,24,-24,0],relative_onset_s=.02,duration_s=.15,steps=40,lanes=4,maximum_lower_ticks=160,optimizer_updates=0,action='zero residual, existing ECBC/ESO',scope='Force wiring only; onset shortened for bounded engineering, not the training onset distribution.'))
np.savez_compressed(out/'initial.npz',pulse=np.asarray(pulse),actor_obs=np.asarray(jax.vmap(env.observation)(states)[0]))
def body(state,_):
    state,reward,done,logs=jax.vmap(env.step)(state,j.zeros((4,2)))
    return state,logs
fn=jax.jit(lambda state:jax.lax.scan(body,state,None,length=40))
save('status.json',dict(phase='inflight',pid=os.getpid(),active_lower_ticks=None,allocated_lower_slots=160,optimizer_updates=0))
t=time.monotonic();end,logs=fn(states);jax.block_until_ready(end);elapsed=time.monotonic()-t
raw=jax.tree.map(np.asarray,logs)
from sttw_control.direct_command_training import flatten_logs
flat=flatten_logs(raw);np.savez_compressed(out/'raw.npz',**flat)
impulse=np.asarray(logs['applied_force_impulse']);total=impulse.sum(axis=0)
assert np.isfinite(np.asarray(end.frames)).all();np.testing.assert_allclose(total[:,1],[0,3.6,-3.6,0],atol=.01)
np.testing.assert_allclose(impulse[:3],0,atol=1e-6);np.testing.assert_allclose(impulse[35:],0,atol=1e-6)
np.testing.assert_allclose(np.asarray(logs['actual_xy'])[:,0],np.asarray(logs['actual_xy'])[:,3],atol=1e-5)
save('receipt.json',dict(actual_lower_ticks=160,allocated_lower_slots=160,optimizer_updates=0,elapsed_s=elapsed,total_impulse_Ns=total.tolist(),post_pulse_clear=True,duplicate_nominal_agreement=True,finite_final_history=True,failures=np.asarray(end.physical.failed).tolist(),scope='GPU wiring verification; CPU twelve-tick test supplies absent-option/zero comparison.'))
save('status.json',dict(phase='completed',pid=os.getpid(),active_lower_ticks=160,allocated_lower_slots=160,optimizer_updates=0))
print(json.dumps(dict(elapsed_s=elapsed,impulse=total.tolist())))
