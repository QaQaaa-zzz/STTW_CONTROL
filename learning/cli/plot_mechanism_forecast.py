"""Show all candidate margins; dashed lines are interpolated predictions, not trials."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();d=json.loads((a.run/'mechanism_forecast.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
for family,shape in [('ridge',(4,4)),('gravel',(2,3))]:
    contexts=[c for c in d['contexts'] if c['family']==family]
    fig,axes=plt.subplots(*shape,figsize=(shape[1]*3.3,shape[0]*2.65),squeeze=False,layout='constrained')
    for ax,c in zip(axes.flat,contexts):
        x=[p['alpha'] for p in c['predictions']]
        for field,scale,color,label in [('recovery_margin_rad',.7,'#0072B2','Roll recovery'),('domain_margin_m',1.,'#D55E00','Forward preview'),('heading_margin_rad',np.pi/2,'#009E73','Heading')]:
            y=np.array([[m[field]/scale for m in p['members']] for p in c['predictions']])
            ax.plot(x,y.mean(1),'--',color=color,lw=1.6,label=label)
            ax.fill_between(x,y.min(1),y.max(1),color=color,alpha=.10)
        ax.axhline(0,color='.25',lw=.8)
        chosen=c['selected_alpha_tasks']['C'];ax.axvline(chosen['alpha'],color='#CC79A7',ls=':',lw=1.4)
        ax.set(title=f"{c['height_m']*1000:g} mm | {c['speed_m_s']:g} m/s | C: {chosen['alpha']:g}"+(' (fallback)' if chosen['feasible_vote']==0 else ''),xlabel='Preference alpha',ylabel='Margin (symlog)',xticks=[0,.25,.5,.75,1])
        ax.set_yscale('symlog',linthresh=1)
        ax.title.set_fontsize(9)
        ax.grid(alpha=.18)
    axes.flat[0].legend(fontsize=8,loc='best')
    fig.suptitle(f'{family.title()}: model-linked margin predictions\nDashed: two DEV scenario mean; band: range of groups 101/102\nDotted: task C choice; linear y within +/-1',fontsize=11)
    fig.savefig(a.run/f'{family}_margin_predictions.png');plt.close(fig)
