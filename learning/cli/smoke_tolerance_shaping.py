"""Four reward modes, frozen actors, shared real snapshots; no PPO."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys,json,pickle,argparse,time,copy,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import jax
import jax.numpy as j
import numpy as np
from sttw_control.path_command_training import load_environment
from sttw_control.path_command_policy import PathCommandActor
from sttw_control.geometric_path import build_path
from sttw_control.preference_best import atomic_json,digest
from sttw_control.runtime import configure_compilation_cache
from sttw_control.tolerance_shaping import validate_tolerance_config
from calibrate_directional_tolerance import BASE,PACKAGE

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise FileExistsError(out)
 started=time.monotonic();ticks=0
 def status(phase,**kw):atomic_json(out/'status.json',dict(phase=phase,pid=os.getpid(),checked_unix=time.time(),elapsed_s=time.monotonic()-started,active_lower_ticks=ticks,declared_lower_ticks=1600,training_steps=0,**kw))
 status('running',stage='load')
 try:
  cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache');cfg=json.loads((BASE/'learning/configs/STTW_Path_Coverage_V2.json').read_text());source=json.loads((BASE/'docs/evidence/select300_integration_20261010/manifest.json').read_text());pkg=json.loads(PACKAGE.read_text());env,prepared=load_environment(cfg,source,'terminal_contract_v2');path=build_path('left90_R2',np.asarray(env.physics.helpers.pose(prepared.data)),cfg)
  states=[];params=[]
  for alpha in [0,1]:
   ident=pkg['model_selection'][f'alpha{alpha}'];assert digest(ident['actor'])==ident['actor_sha256']
   with open(ident['actor'],'rb') as f:params.append(pickle.load(f))
   with open(f'runs/cross_state_v1_attempt2/left90_R2_v2.3_a{alpha}_t1_snapshot.pkl','rb') as f:states.append(pickle.load(f))
  states=jax.tree.map(lambda *x:j.stack(x),*states);params=jax.tree.map(lambda *x:j.stack(x),*params);single=jax.tree.map(lambda x:x[0],states)
  atomic_json(out/'manifest.json',dict(code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),modes=['baseline','zero','static','dynamic'],endpoints=[0,1],steps_per_mode=50,lower_ticks=1600,checkpoint_identities=pkg['model_selection'],physical_identity=env.physics.bundle.identity,lower=source['candidate'],cache=cache,scope='Numerical reward-accounting smoke only; no performance comparison or PPO'))
  results={};base=None;actor=PathCommandActor()
  for mode in ['baseline','zero','static','dynamic']:
   current=copy.copy(env);current.config=copy.deepcopy(cfg)
   if mode!='baseline':
    c=current.config['path_reward'];weight=0. if mode=='zero' else 10.;c['tolerance']=dict(method='static' if mode=='static' else 'dynamic',weight=weight,gamma=.815560907125473,effort_budget=.030839085298575167,duration=.1,target_force=6.)
    if weight:c['component_caps']['tolerance']=weight;c['failure_cost_rate_bound']+=weight
   validate_tolerance_config(current.config);atomic_json(out/(mode+'_config.json'),current.config);current.set_log_template(single,path,2.3)
   def one(st,pa):
    obs,_,fault=current.observation(st,path,2.3,20.);st=st.replace(fault=st.fault|fault)
    return current.policy_step(st,actor.apply(pa,obs),path,2.3,20.)
   advance=jax.vmap(one);execute=jax.jit(lambda st:jax.lax.scan(lambda ss,_:advance(ss,params),st,None,length=50))
   status('running',stage=mode);end,logs=execute(states);jax.block_until_ready(end);ticks+=int(np.asarray(logs['active_tick']).sum());host=jax.device_get(logs)
   assert not np.any(host['policy_fault']) and not np.any(host['lower_fault']) and not np.any(host['physical_failure']) and not np.any(host['domain_exit'])
   expected=-cfg['path_reward']['scale']*.005*sum(host['effective_components'].values());np.testing.assert_allclose(host['scored_tick_reward'],expected,rtol=1e-5,atol=1e-7)
   q=np.asarray(end.physical.data.qpos);v=np.asarray(end.physical.data.qvel);assert np.isfinite(q).all() and np.isfinite(v).all()
   if base is None:base=(q,v,host['scored_tick_reward'])
   result=dict(ticks=int(host['active_tick'].sum()),components=list(host['effective_components']),reward_sum=float(host['scored_tick_reward'].sum()),max_qpos_difference=float(abs(q-base[0]).max()),max_qvel_difference=float(abs(v-base[1]).max()),max_reward_difference=float(abs(host['scored_tick_reward']-base[2]).max()))
   if 'tolerance' in host['effective_components']:
    cost=host['effective_components']['tolerance'];result.update(tolerance_cost_rate_min=float(cost.min()),tolerance_cost_rate_max=float(cost.max()),tolerance_cost_integral=float(cost.sum()*.005));assert np.isfinite(cost).all() and np.all((cost>=0)&(cost<=10))
   results[mode]=result;atomic_json(out/'summary.json',results);status('running',stage=mode+'_saved')
  status('completed',summary=str(out/'summary.json'))
 except BaseException as e:status('error',error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':main()
