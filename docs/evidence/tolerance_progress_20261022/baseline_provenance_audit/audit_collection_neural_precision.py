"""Bounded GPU-only neural layout/precision diagnostic; zero physics or optimizer."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as j
from sttw_control.preference_best import atomic_json,digest
from sttw_control.survival_online_learner import OnlineSurvivalLearner
from sttw_control.survival_collection import evaluate_collection_networks
from sttw_control.path_command_policy import make_path_actor,make_path_critic
from sttw_control.survival_task_credit import reconstruct_task_critic

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
output=root/'neural_precision_audit.json'
if output.exists():raise FileExistsError('preserve prior diagnostic')
plan=json.loads((root/'plan.json').read_text());cfg=json.loads((root/'effective_config.json').read_text());torch.set_num_threads(plan['torch_threads'])
assert digest(plan['model']['checkpoint'])==plan['model']['checkpoint_sha256']
checkpoint=torch.load(plan['model']['checkpoint'],map_location='cpu',weights_only=False)
learner=OnlineSurvivalLearner(cfg,checkpoint['policy'],beta=0.,seed=plan['learner_seed'])
raw=Path(plan['source_plan']).parent/'case_000.npz'
with np.load(raw) as z:
    obs=z['decision_observation'][z['decision_valid']][:16];ticks=z['decision_task_tick'][z['decision_valid']][:16]
    actions=z['decision_raw_action'][z['decision_valid']][:16]
view=reconstruct_task_critic(obs,ticks,np.ones(16,bool),dt_s=.005,horizon_s=10.)
params=learner.sampling_parameters();rows=[]
for layout,n in [('shared_parameter_GEMM',16),('per_environment_parameters',96)]:
    x=np.tile(obs,(n//16,1));v=np.tile(view,(n//16,1));acts=torch.tensor(np.tile(actions,(n//16,1)))
    with torch.no_grad():
        mean=learner.actor(torch.tensor(x));task=learner.task_critic(torch.tensor(v)).squeeze(-1);physical=learner.physical_critic(torch.tensor(v)).squeeze(-1)
        logp=torch.distributions.Normal(mean,learner.log_std.exp()).log_prob(acts).sum(-1).numpy()
    for precision in ['default','highest']:
        started=time.monotonic()
        with jax.default_matmul_precision(precision):
            if layout=='shared_parameter_GEMM':
                fn=jax.jit(jax.vmap(lambda x,v:evaluate_collection_networks(make_path_actor(cfg),make_path_critic(cfg),params,x,v)))
                values=fn(j.asarray(x),j.asarray(v))
            else:
                batch=jax.tree.map(lambda x:j.stack([x]*n),params)
                fn=jax.jit(jax.vmap(lambda p,x,v:evaluate_collection_networks(make_path_actor(cfg),make_path_critic(cfg),p,x,v)))
                values=fn(batch,j.asarray(x),j.asarray(v))
            values=jax.device_get(values)
        candidate=torch.distributions.Normal(torch.tensor(values[0]),learner.log_std.detach().exp()).log_prob(acts).sum(-1).numpy()
        rows.append(dict(layout=layout,batch=n,precision=precision,head_max_errors=[float(np.max(np.abs(a-b.detach().numpy()))) for a,b in zip(values,[mean,task,physical])],
            joint_log_probability_max_error=float(np.max(np.abs(candidate-logp))),elapsed_s_including_compile=time.monotonic()-started))
atomic_json(output,dict(rows=rows,devices=[str(d) for d in jax.devices()],default_matmul_precision=str(jax.config.jax_default_matmul_precision),
    raw_sha256=digest(raw),checkpoint_sha256=digest(plan['model']['checkpoint']),source_sha256=digest(__file__),new_physics_ticks=0,optimizer_steps=0,
    scope='Bounded network layout diagnosis, not physical replay or task efficacy.'))
print(json.dumps(rows))
