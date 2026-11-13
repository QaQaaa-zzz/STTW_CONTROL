"""Summarize the declared two-terrain native RB chassis ground test."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
fig,axes=plt.subplots(2,3,figsize=(15,8));speedfig,saxes=plt.subplots(2,3,figsize=(15,8))
rows=[]
for i,terrain in enumerate(('flat','rough')):
 root=a.root/terrain;state=json.loads((root/'status.json').read_text())
 for r in state['results']:
  z=np.load(root/f"{r['skill']}_route{r['route']}_{r['continuation']}"/'trajectory.npz');t=z['trace'];j=r['route']-60
  ax=axes[i,j];sx=saxes[i,j];mode=r['continuation'];color='tab:blue' if mode=='h75' else 'tab:orange'
  ax.plot(t[:,1],t[:,2],label=mode,color=color);ax.scatter(t[-1,1],t[-1,2],marker='x',color=color)
  if mode=='h75':ax.plot(z['reference_xy'][:,0],z['reference_xy'][:,1],'k:',label='original route')
  actual=t[:,7]*np.cos(t[:,6])+t[:,8]*np.sin(t[:,6])
  sx.plot(t[:,0],actual,color=color,label=mode+' body speed')
  sx.plot(t[:,0],t[:,17]*.07,color=color,linestyle=':',label=mode+' wheel speed')
  sx.plot(t[:,0],t[:,13]*.07,color=color,linestyle='--',label=mode+' delivered ref')
  ax.set(title=f'{terrain}, route {r["route"]}',xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
  sx.set(title=f'{terrain}, route {r["route"]}',xlabel='time (s)',ylabel='speed (m/s)');sx.grid(alpha=.2)
  if mode=='zero_middle':sx.axhline(2,color='black',linewidth=.8,label='request 2 m/s');sx.legend(fontsize=6)
  rows.append(dict(terrain=terrain,route=r['route'],method=mode,end=r['termination'],duration_s=r['duration_s'],goal=r['goal'],max_path_error_m=float(t[:,11].max()),speed_mean_m_s=float(actual.mean()),speed_rmse_m_s=float(np.sqrt(((actual-2)**2).mean())),max_abs_roll_rad=float(abs(t[:,4]).max()),max_abs_pitch_rad=float(abs(t[:,5]).max()),failure_axis='roll' if abs(t[-1,4])>.7 else 'pitch' if abs(t[-1,5])>1.309 else 'other',jump_activated=bool((t[:,16]==0).any())))
fig.tight_layout();fig.savefig(a.root/'xy.png',dpi=160);speedfig.tight_layout();speedfig.savefig(a.root/'speed.png',dpi=160)
(a.root/'metrics.json').write_text(json.dumps(rows,indent=2))
lines=['# RB native chassis: STTW ground-only verification','','![All actual and reference XY](xy.png)','','![Body, wheel, delivered and requested speeds](speed.png)','','Nominal DEV integration diagnostic, 12 episodes: two terrains × three routes × H75/zero-middle. Requested speed 2 m/s, H75 alpha 0.5. Same nominal initialization; no guarantee of bitwise identical GPU dynamics. No jumping or training. Zero-middle retains ECBC+ESO and the reference filters; neither method includes Local300. Different actual observation lengths end at each real failure; averages are descriptive, not matched-window rankings.','','|Terrain|Route|Method|End|Duration s|Mean speed m/s|Speed RMSE m/s|Max path error m|Peak roll rad|Peak pitch rad|Failure axis|','|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"|{r['terrain']}|{r['route']}|{r['method']}|{r['end']}|{r['duration_s']:.3f}|{r['speed_mean_m_s']:.3f}|{r['speed_rmse_m_s']:.3f}|{r['max_path_error_m']:.3f}|{r['max_abs_roll_rad']:.3f}|{r['max_abs_pitch_rad']:.3f}|{r['failure_axis']}|")
lines+=['','The old floor/step explicit contact pairs were removed. The trench is at x=40 m, outside this test. Flat has no stones/ridges; rough preserves the declared millimeter stones and ridges. All actions and physical trajectories are in the terrain subdirectories. No reward comparison: this inference-only runner does not implement a task reward.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
