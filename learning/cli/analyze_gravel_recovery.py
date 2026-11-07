"""Audit prefix identity and report task entry/recovery matrices from saved P3."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from sttw_control.isaac_evaluation_batch import split_case
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();raw=a.run/'batch/evaluation';protocol=json.loads((a.run/'protocol.json').read_text());cases=json.loads((raw.parent/'cases.json').read_text());contract=json.loads((raw/'contract.json').read_text());status=json.loads((raw/'status.json').read_text());archive=np.load(raw/'trajectory.npz');data={k:archive[k] for k in archive.files};archive.close();forks=json.loads((raw/'gravel_recovery_events.json').read_text());events={e['group']:e for e in forks};n=data['reward'].shape[1];trace=data['lower_trace'].reshape(-1,n,11);audits=[];records=[];index=[]
base=json.loads(Path(protocol['base_sweep_config']).read_text());tolerance=protocol['prefix_acceptance']
for gi,case in enumerate(cases):
 lanes=np.arange(gi*5,gi*5+5);event=events.get(gi);stop=len(data['reward']) if event is None else event['policy_step'];lower_stop=stop*4;commands=data['executed_commands'][:stop,:,lanes,:];position=trace[:lower_stop,lanes,:2];speed=trace[:lower_stop,lanes,2];roll=data['measurement'][:lower_stop,lanes,0];obs=data['observation'][:lower_stop,lanes,:]
 # Advanced indexing keeps the explicit lane axis in the saved command trace.
 command_delta=float(np.max(abs(commands-commands[:,:,:1,:])[:,:,:,1:3])) if stop else 0.
 delta=lambda v:float(np.max(abs(v-v[:,:1]))) if lower_stop else 0.
 audit=dict(group=gi,entry_alpha=case['entry_alpha'],parameter_seed=case['seed'],forked=event is not None,prefix_policy_steps=stop,position_max_m=delta(position),speed_max_m_s=delta(speed),roll_max_rad=delta(roll),observation_max=delta(obs),executed_command_max_rad_s=command_delta)
 audit['runtime_prefix_at_decision']=None if event is None else event['prefix_at_decision']
 audit['prefix_accepted']=bool(event is not None and all(audit[k]<=v for k,v in tolerance.items()));audits.append(audit)
 part,c=split_case(data,contract,dict(case,batch_output=str(raw.resolve())),gi);c['settings']['alphas']=[case['entry_alpha']]*5;c['settings']['gravel_recovery']['groups']=[dict(lanes=list(range(5)),entry_alpha=case['entry_alpha'],target_alphas=protocol['recovery_alphas'])];run=Path(case['output']);run.mkdir(parents=True);np.savez_compressed(run/'trajectory.npz',**part);(run/'contract.json').write_text(json.dumps(c,indent=2)+'\n');(run/'status.json').write_text(json.dumps(dict(status,physics_steps=len(part['reward'])*5*100,failures=part['failure'].sum(0).tolist(),valid_steps=part['valid'].sum(0).tolist()),indent=2)+'\n')
 with (run.parent/'analysis.log').open('w') as log:subprocess.run([sys.executable,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(run),'--preference-label-prefix','recovery alpha','--title',f"P3 S6 | common entry{case['entry_alpha']} | parameter{case['seed']} |0.10s recovery ramp after clearance"],stdout=log,stderr=subprocess.STDOUT,check=True)
 index.append(dict(case,run=str(run.resolve())))
 for i,b in enumerate(json.loads((run/'condition_metrics.json').read_text())):
  local=part['lower_trace'][:,:,i].reshape(-1,11);valid=local[:,6].astype(bool);peak=float(abs(local[valid,3]).max());r=dict(b,entry_alpha=case['entry_alpha'],recovery_alpha=protocol['recovery_alphas'][i],parameter_seed=case['seed'],peak_path_error_m=peak,prefix_accepted=audit['prefix_accepted'],fork_time_s=None if event is None else event['time_s'])
  for k,t in base['tasks'].items():r['task_'+k]=bool(b['goal_crossed'] and b['final_hold'] and not b['failed'] and peak<=t['W_m'] and b['speed_rmse_m_s']<=t['speed_rmse_m_s'])
  if event is not None:
   indices=np.arange(len(local));time=indices*.005+local[:,7]*contract['physics_dt'];post=valid&(time>=event['time_s'])&(time<=event['time_s']+2)
   r['post_fork_2s']=dict(observed_s=float(post.sum()*.005),path_rmse_m=float(np.sqrt(np.mean(local[post,3]**2))) if post.any() else None,speed_rmse_m_s=float(np.sqrt(np.mean((local[post,2]-case['user_speed_m_s'])**2))) if post.any() else None,peak_path_m=float(abs(local[post,3]).max()) if post.any() else None)
  records.append(r)
out=a.run/'evaluation';(out/'index.json').write_text(json.dumps(index,indent=2)+'\n');(out/'prefix_audit.json').write_text(json.dumps(audits,indent=2)+'\n');(out/'recovery_task_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
lines=['# P3 S6 entry/recovery DEV','','75 same-start candidates,15 cohorts. No snapshot, no training. Inspect prefix_audit.json before causal claims; unaccepted or missing forks retained. Reward varies with preference schedule and is not used to rank physical task quality.']
for case in index:lines.extend(['',f"## entry{case['entry_alpha']} / parameter{case['seed']}",f"![XY]({case['run']}/xy_all_alpha.png)",f"[Per-step/cumulative rewards]({case['run']}/overview.png)"])
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n');(out/'status.json').write_text(json.dumps(dict(stage='complete',conditions=len(records),forked_groups=len(forks),prefix_accepted_groups=sum(x['prefix_accepted'] for x in audits),physics_lane_steps=status['physics_steps'],optimizer_updates=0),indent=2)+'\n');print(json.dumps(dict(conditions=len(records),prefix_accepted_groups=sum(x['prefix_accepted'] for x in audits))))
