"""Report every declared frozen native skill/continuation case, including failures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
s=json.loads((a.root/'status.json').read_text());d=json.loads((a.root/'declaration.json').read_text())
config=json.loads(Path(d['source_config']).read_text());hand=config['ground_handoff']
mechanism=hand.get('handoff_mechanism','contact_delay')
fig,axes=plt.subplots(len(d['routes']),1,figsize=(12,6),squeeze=False)
for ax,route in zip(axes[:,0],d['routes']):
 for row in s['results']:
  if row['route']!=route:continue
  z=np.load(a.root/row.get('case_dir',f"{row['skill']}_route{route}_{row['continuation']}")/'trajectory.npz');t=z['trace']
  ax.plot(t[:,1],t[:,2],label=row.get('case_dir',row['skill']+' / '+row['continuation']),linestyle='-' if row['continuation']=='h75' else '--')
  ax.scatter(t[-1,1],t[-1,2],marker='x',s=20)
 ax.plot(z['reference_xy'][:,0],z['reference_xy'][:,1],'k:',label='original route')
 ax.axvspan(d['terrain']['front_x_m'],d['terrain']['front_x_m']+d['terrain']['width_m'],alpha=.2,color='black')
 ax.set(xlabel='x (m)',ylabel='y (m)',title=f'Route {route}',xlim=(0,20),ylim=(-1.5,1.5));ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
axes[-1,0].legend(fontsize=6);fig.tight_layout();fig.savefig(a.root/'xy.png',dpi=160)
tracking,panels=plt.subplots(3,len(d['routes']),figsize=(15,9),squeeze=False,layout='constrained')
for j,route in enumerate(d['routes']):
 for row in s['results']:
  if row['route']!=route:continue
  z=np.load(a.root/row.get('case_dir',f"{row['skill']}_route{route}_{row['continuation']}")/'trajectory.npz');t=z['trace'];q=z['qpos'];label=row.get('case_dir',row['skill'])
  panels[0,j].plot(t[:,0],t[:,7]*np.cos(t[:,6])+t[:,8]*np.sin(t[:,6]),label=label)
  panels[1,j].plot(t[:,0],np.rad2deg(t[:,4]),label=label)
  panels[2,j].plot(t[:,0],q[:,-2],label=label+' hip');panels[2,j].plot(t[:,0],q[:,-1],ls='--',label=label+' knee')
  for event in ['rb_entry_s','first_far_contact_s','handoff_s']:
   if row.get(event) is not None:
    for ax in panels[:,j]:ax.axvline(row[event],color='gray',ls=':',lw=.7)
 for ax in panels[:,j]:ax.grid(alpha=.2);ax.set_xlabel('Time (s)')
 panels[0,j].set(title=f'Route {route}',ylabel='Body forward speed (m/s)');panels[1,j].set_ylabel('Roll (deg)');panels[2,j].set_ylabel('Hip / knee (rad)')
 panels[0,j].legend(fontsize=6);panels[2,j].legend(fontsize=6)
tracking.savefig(a.root/'stages_tracking.png',dpi=140);plt.close(tracking)
lines=['# Native Playground RB–STTW handoff trial','','![All actual/reference XY trajectories](xy.png)','','Role: bounded integration DEV; one common nominal ground start per route, not an independent success-rate estimate.','','| Skill | Route | Variant | Continuation | RB entry s | Airborne | Far contact | Handoff s | End | Final XY m |','|---|---:|---|---|---:|---|---|---:|---|---|']
for r in s['results']:
 lines.append(f"|{r['skill']}|{r['route']}|{r.get('variant',{})}|{r['continuation']}|{r.get('rb_entry_s')}|{r['airborne']}|{r['far_contact']}|{r['handoff_s']}|{r['termination']}|{r['final_xy']}|")
lines+=['','![Stage timing, actual speed/roll and joint angles](stages_tracking.png)','',f"Completed {s['completed']}/{s['total']} episodes. The declared approach controller (see declaration.json) precedes native RB jumping; handoff mechanism is {mechanism}, landing control={hand.get('landing_control','none')}. State-triggered runs use descent/contact/posture/joint readiness; the legacy fixed delay {hand.get('handoff_delay_s',0.)} s applies only to contact_delay runs. STTW hip/knee brakes are active; RB releases them and measured handoff re-locks the actual pose. No pose reset. Final goal/hold follows declaration and source config.",'',f"Terrain uses original Isaac rectangular samples and matching triangle diagonals, without resampling. Ground physics dt={d['physics_dt']} s, RB physics={d.get('rb_physics')}, contact domain={d.get('ground_physics')}, lower dt={d['lower_dt']} s, upper dt={d['upper_dt']} s. This is a transfer feasibility trial, not a simulator-only causal comparison. H75 starts with measured ESO state and empty policy history."]
if d.get('ground_only',False):
 lines[0]='# Native RB chassis ground-only STTW test'
 lines += ['', f"Ground-only diagnostic: RB is never activated. Initial x={d['start_x_m']} m, requested speed={hand['approach']['speed_m_s']} m/s; frozen H75 and zero-middle share the same physical start and contact domain. This independent settled start is not a physical reset in the full chain and does not establish RB handoff success."]
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
