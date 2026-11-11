"""Plot exact best-control envelopes against frozen recorded states."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_nominal_recovery import envelope
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output
c=json.loads(Path('runs/isaac_signals_parity/dynamic.json').read_text());rows=list(csv.DictReader((o/'recorded_states_frozen_model.csv').open()))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,start in zip(axes,[1.9,1.94]):
    for label,color in [('full_0.5','#008b77'),('full_1','#c84742')]:
        r=next(r for r in rows if r['case']=='single10_v2.6_seed101' and r['label']==label and abs(float(r['t'])-start)<1e-6)
        x=[np.radians(float(r['phi_deg'])),np.radians(float(r['omega_deg_s'])),float(r['delta_rad'])];end=float(r['event_s']);t=np.linspace(0,end,501);(phi,w,d),side=envelope(float(r['v']),x,t,c)
        ax.plot(t*1000,np.degrees(side*phi),color=color,lw=2.5,label=label.replace('full_','alpha = '));ax.scatter(end*1000,np.degrees(side*phi[-1]),color=color,marker='x' if label=='full_1' else 'o',s=60)
        ax.text(.06 if label=='full_0.5' else .60,.83 if label=='full_0.5' else .20,f"{r['outcome'].replace('unavoidable_nominal_limit','limit reached')}\n{end*1000:.2f} ms",transform=ax.transAxes,color=color,fontsize=10)
    ax.axhline(-np.degrees(.7),color='#444444',ls='--',label='roll termination limit');ax.set(title=f'Frozen recorded state at {start:.3f} s',xlabel='Time after state capture (ms)',ylabel='Signed roll (degrees)',ylim=(-46,20));ax.grid(alpha=.18);ax.legend(loc='center right',fontsize=9)
fig.suptitle('10 cm ridge / physical group 101: best admissible nominal recovery',fontsize=16)
fig.savefig(o/'optimal_recovery.png',dpi=180)
