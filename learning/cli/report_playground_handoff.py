"""Report every declared frozen native skill/continuation case, including failures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
s=json.loads((a.root/'status.json').read_text());d=json.loads((a.root/'declaration.json').read_text())
fig,axes=plt.subplots(1,len(d['routes']),figsize=(15,5),squeeze=False)
for ax,route in zip(axes[0],d['routes']):
 for row in s['results']:
  if row['route']!=route:continue
  z=np.load(a.root/f"{row['skill']}_route{route}_{row['continuation']}"/'trajectory.npz');t=z['trace']
  ax.plot(t[:,1],t[:,2],label=row['skill']+' / '+row['continuation'],linestyle='-' if row['continuation']=='h75' else '--')
  ax.scatter(t[-1,1],t[-1,2],marker='x',s=20)
 ax.plot(z['reference_xy'][:,0],z['reference_xy'][:,1],'k:',label='original route')
 ax.axvspan(d['terrain']['front_x_m'],d['terrain']['front_x_m']+d['terrain']['width_m'],alpha=.2,color='black')
 ax.set(xlabel='x (m)',ylabel='y (m)',title=f'Route {route}',xlim=(0,20));ax.axis('equal');ax.grid(alpha=.2)
axes[0,-1].legend(fontsize=6);fig.tight_layout();fig.savefig(a.root/'xy.png',dpi=160)
lines=['# Native Playground RB–STTW handoff trial','','![All actual/reference XY trajectories](xy.png)','','Role: bounded integration DEV; one common nominal ground start per route, not an independent success-rate estimate.','','| Skill | Route | Continuation | Airborne | Far contact | Handoff s | End | Final XY m |','|---|---:|---|---|---|---:|---|---|']
for r in s['results']:
 lines.append(f"|{r['skill']}|{r['route']}|{r['continuation']}|{r['airborne']}|{r['far_contact']}|{r['handoff_s']}|{r['termination']}|{r['final_xy']}|")
lines+=['',f"Completed {s['completed']}/{s['total']} episodes. The declared approach controller (see declaration.json) precedes native RB jumping; after measured far-bank wheel contact RB continues for {d.get('handoff_delay_s', 0.)} s before H75 or its zero-middle comparator is engaged. Native vehicle, servos and finite-torque leg hold remain physical; no pose reset or artificial stable-support requirement.",'','The heightfield retains the original seeded rough-bank surface through 25mm resampling. Native vehicle/collision behavior and 5ms physics differ from Isaac, so this is a transfer feasibility trial, not a simulator-only causal comparison. H75 starts with measured ESO state and empty policy history.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
