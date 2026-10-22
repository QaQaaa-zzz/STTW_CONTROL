#!/usr/bin/env python3
"""Analyze the fixed finite pulse grid. No simulation or learned calibration."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run);data=json.loads((root/'results.json').read_text())['cases'];manifest=json.loads((root/'manifest.json').read_text());path=np.load(root/'path.npz')
 assert len(data)==42
 fig,axes=plt.subplots(2,3,figsize=(16,9));colors=plt.cm.coolwarm(np.linspace(0,1,7));order=[-6.,-3.,-1.,0.,1.,3.,6.]
 direction=[];nominal=[]
 for alpha in [0,1]:
  for col,onset in enumerate([1.,2.,4.]):
   ax=axes[alpha,col];rows=[r for r in data if r['alpha']==alpha and r['onset_s']==onset];allxy=[]
   for color,f in zip(colors,order):
    row=next(r for r in rows if r['force_N']==f);d=np.load(root/(row['name']+'.npz'));xy=d['actual_xy'];allxy.append(xy)
    ax.plot(xy[:,0],xy[:,1],color=color,lw=1.2,label=f'{f:+g} N '+('pass' if row['success'] else 'fail'))
    idx=min(round(onset/.005),len(xy)-1);ax.scatter(*xy[idx],color=color,s=12)
   xy=np.concatenate(allxy);ref=path['xy'];mask=(path['s']<=float(np.max([np.load(root/(r['name']+'.npz'))['path_progress'].max() for r in rows]))+.2)
   ax.plot(ref[mask,0],ref[mask,1],'k--',lw=1,label='original path');ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'alpha={alpha}; pulse onset={onset:g}s');ax.set_xlabel('world X [m]');ax.set_ylabel('world Y [m]');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2)
   base=next(r for r in rows if r['force_N']==0);nominal.append({'alpha':alpha,'onset':onset,'success':base['success']})
   for sign in [-1,1]:
    vals=sorted([r for r in rows if r['force_N']*sign>0],key=lambda r:abs(r['force_N']))
    prefix=[]
    for row in vals:
     if not row['success']:break
     prefix.append(abs(row['force_N']))
    direction.append({'alpha':alpha,'onset':onset,'sign':sign,'nominal_success':base['success'],'sampled_successes':[abs(r['force_N']) for r in vals if r['success']],'first_sampled_failure':next((abs(r['force_N']) for r in vals if not r['success']),None),'successful_grid_prefix':prefix,'continuous_margin':'UNKNOWN'})
 fig.suptitle('Frozen Coverage u200 / local300 | 0.1 s world-fixed lateral force | DEV pilot',fontsize=12);fig.tight_layout();fig.savefig(root/'xy_all_cases.png',dpi=170);plt.close(fig)
 summary={'cases':42,'success_count':sum(r['success'] for r in data),'nominal':nominal,'directions':direction,'physical_failures':sum(r['physical_failure'] for r in data),'working_failures':sum(r['working_failure'] for r in data),'domain_failures':sum(r['domain_failure'] for r in data),'hold_failures':sum(not r['final_hold'] for r in data),'scope':'one route, one demand, deterministic development states; not formula validation or independent performance estimate'}
 (root/'summary.json').write_text(json.dumps(summary,indent=2))
 rows=['# Directional pulse pilot v1','', 'Frozen Coverage u200 endpoints and local300; 42 declared trajectories completed. No training.', '', f"Success: {summary['success_count']}/42 (six zero-force repetitions included; not independent statistical trials).", '', '| alpha | pulse time (s) | direction | sampled success amplitudes (N) | first sampled failure (N) |','|---|---|---|---|---|']
 for r in direction:rows.append(f"| {r['alpha']} | {r['onset']} | {r['sign']:+d} | {r['sampled_successes']} | {r['first_sampled_failure']} |")
 rows+=['','![All actual and original reference XY trajectories](xy_all_cases.png)','','Positive sign means force toward left at pulse start, not force coming from left. Force point is subtree COM equivalent wrench on chassis. Pulse direction stays fixed in world coordinates.','', 'No continuous tolerance boundary is certified. All-pass at 6 N is test-range censoring, not proof of 6 N continuous tolerance. A sampled failure bracket is not an exact boundary. Nominal failures must remain separate.','', 'Mathematical primitives are repaired, but force mapping, effort budget and full closed-loop sensitivity remain uncalibrated. These data establish an initial physical test panel; they do not establish superiority of dynamic UCEDT.']
 (root/'REPORT.md').write_text('\n'.join(rows)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('directions','nominal')},indent=2))
if __name__=='__main__':main()
