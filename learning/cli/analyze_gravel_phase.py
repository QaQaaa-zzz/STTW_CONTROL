"""Split the150-lane P2 record, reuse ordinary plots and native contact analysis."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from sttw_control.isaac_evaluation_batch import split_case
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();raw=a.run/'batch/evaluation';c=json.loads((raw/'contract.json').read_text());status=json.loads((raw/'status.json').read_text());cases=json.loads((raw.parent/'cases.json').read_text());archive=np.load(raw/'trajectory.npz');arrays={key:archive[key] for key in archive.files};archive.close();index=[]
for i,case in enumerate(cases):
 run=Path(case['output']);run.mkdir(parents=True)
 part,contract=split_case(arrays,c,dict(case,batch_output=str(raw.resolve())),i)
 np.savez_compressed(run/'trajectory.npz',**part);(run/'contract.json').write_text(json.dumps(contract,indent=2)+'\n');(run/'status.json').write_text(json.dumps(dict(status,physics_steps=len(part['reward'])*5*100,failures=part['failure'].sum(0).tolist(),valid_steps=part['valid'].sum(0).tolist(),elapsed_scope='shared150-lane batch, not additive'),indent=2)+'\n')
 with (run.parent/'analysis.log').open('w') as log:subprocess.run([sys.executable,'learning/cli/plot_isaac_gpu_evaluation.py','--run',str(run),'--title',f"P2 separate extended route | {case['terrain']} | {case['user_speed_m_s']}m/s | parameter{case['seed']}"],stdout=log,stderr=subprocess.STDOUT,check=True)
 index.append(dict(terrain=case['terrain'],user_speed_m_s=case['user_speed_m_s'],seed=case['seed'],run=str(run.resolve()),role='P2 separate-route fixed preference DEV; no Zero in150-lane budget'))
 out=a.run/'evaluation';(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
out=a.run/'evaluation';lines=['# P2 separate-route three-cluster spacing DEV','','150 trajectories, no PPO; five alphas, five frozen physical parameter groups, two speeds. Do not mix rankings with original-route P1. All actual failed endpoints retained.']
for case in index:lines.extend(['',f"## {case['terrain']} / {case['user_speed_m_s']}m/s / parameter{case['seed']}",f"![XY]({case['run']}/xy_all_alpha.png)",f"[Per-step and cumulative reward]({case['run']}/overview.png)"])
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n');(out/'status.json').write_text(json.dumps(dict(stage='physics_and_plots_complete',conditions=150,physics_lane_steps=status['physics_steps'],optimizer_updates=0),indent=2)+'\n')
subprocess.run([sys.executable,'learning/cli/analyze_gravel_p0.py','--evaluation',str(out),'--config',str(a.run/'analysis_config.json')],check=True)
print(json.dumps(dict(conditions=150,cases=len(index),physics_lane_steps=status['physics_steps'])))

from sttw_control.gravel_cluster_phase import cluster_contact_onsets
spec=json.loads((a.run/'protocol.json').read_text());onsets=cluster_contact_onsets(raw/'physics_contacts',c,cases,spec['stones_per_cluster']);conditions=json.loads((a.run/'conditions.json').read_text());route=np.load(a.run/'bundle/route.npz');cluster_rows=[]
for lane,condition in enumerate(conditions):
 tr=arrays['lower_trace'][:,:,lane].reshape(-1,11);time=np.arange(len(tr))*.005+tr[:,7]*c['physics_dt'];states=[]
 for cluster in range(3):
  t=onsets[lane,0,cluster]
  if not np.isfinite(t):states.append(None);continue
  i=int(np.searchsorted(time,t,side='left')-1)
  if i<0 or not tr[i,6]:states.append(None);continue
  measurement=arrays['measurement'][i,lane];yaw=np.interp(arrays['progress'][i,lane],route['s'],route['heading'])-arrays['pose'][i,lane,2]
  states.append(dict(front_contact_on_s=float(t),preceding_sample_s=float(time[i]),roll_rad=float(measurement[0]),roll_rate_rad_s=float(measurement[1]),steering_rate_rad_s=float(measurement[3]),heading_error_rad=float(np.arctan2(np.sin(yaw),np.cos(yaw)))))
 cluster_rows.append(dict(condition,front_onsets_s=[float(t) if np.isfinite(t) else None for t in onsets[lane,0]],rear_onsets_s=[float(t) if np.isfinite(t) else None for t in onsets[lane,1]],pre_cluster_state=states,scope='Native loaded patch point on actual raised mesh; analytic max-cap owner assigns cluster; missing contacts remain null,5ms preceding motion state'))
(out/'cluster_first_contacts.json').write_text(json.dumps(cluster_rows,indent=2)+'\n')
(out/'status.json').write_text(json.dumps(dict(stage='complete',conditions=150,physics_lane_steps=status['physics_steps'],optimizer_updates=0),indent=2)+'\n')
