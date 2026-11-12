import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description='Plot cached paired contact-OAT findings without rerunning physics.')
parser.add_argument('--root', type=Path, required=True)
root=parser.parse_args().root
out=root/'analysis'
s=json.loads((out/'summary.json').read_text());rows=s['cells'];learn=[r for r in rows if r['label']!='Zero']
def aggregate(group):
 n=sum(r['n'] for r in group)
 return dict(n=n,full=sum(r['full_window'] for r in group),physical=sum(r['physical_termination'] for r in group),hold=sum(r['final_hold'] for r in group),unload=sum(r['front_unload_max_mean_s']*r['n'] for r in group)/n)
lengths=[.4,.6,.8,1,1.2,1.6,2]
slope=[dict(length=l,**aggregate([r for r in learn if r['center_s_m'] in (3.9,4.5) and (r['factor']=='length' and r['length_m']==l or l==.8 and r['factor']=='position')] )) for l in lengths]
wheel=[dict(scale=v,**aggregate([r for r in learn if (r['factor']=='wheelbase' and r['value']==v) or (v==1 and r['factor']=='position' and r['center_s_m'] in (3.9,4.5))])) for v in [.9,1,1.1,1.2,1.3]]
oblique=[dict(obliquity=v,**aggregate([r for r in learn if r['factor']=='obliquity' and r['value']==v or v==.25 and r['id']=='position_s4.2'])) for v in [-.5,-.25,0,.25,.5]]
(out/'findings.json').write_text(json.dumps(dict(slope=slope,wheelbase=wheel,obliquity=oblique),indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
ax=axs[0];ax.plot(lengths,[r['full'] for r in slope],'o-',label='Full 10 s',color='#0072B2');ax.plot(lengths,[r['physical'] for r in slope],'s--',label='Physical termination',color='#D55E00');ax.plot(lengths,[r['hold'] for r in slope],'^:',label='Final hold',color='#009E73');ax.set(title='A  Gentler ridge',xlabel='Ridge length (m)',ylabel='Episodes / 80',ylim=(0,84));ax.legend(fontsize=9)
ax=axs[1];ax.plot(lengths,[r['unload'] for r in slope],'o-',color='#0072B2');ax.set(title='B  Front-wheel unloading',xlabel='Ridge length (m)',ylabel='Mean longest unloading interval (s)',ylim=(0,.43))
ax=axs[2];ax.plot([r['scale'] for r in wheel],[r['full'] for r in wheel],'o-',color='#0072B2',label='Full 10 s');ax.plot([r['scale'] for r in wheel],[r['physical'] for r in wheel],'s--',color='#D55E00',label='Physical termination');ax.set(title='C  Frozen-controller geometry transfer',xlabel='Wheelbase scale',ylabel='Episodes / 80',ylim=(0,84));ax.legend(fontsize=9)
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('10 cm ridge | 2.6 m/s request | five alpha values x two placements x eight matched physics blocks',fontsize=12)
fig.savefig(out/'key_factor_findings.png');plt.close(fig)
pos=sorted({r['center_s_m'] for r in learn if r['factor']=='position'});matrix=np.array([[next(r['full_window'] for r in learn if r['factor']=='position' and r['center_s_m']==v and r['lane']==k) for k in range(5)] for v in pos]);headings=[next(r['reference_heading_deg'] for r in learn if r['factor']=='position' and r['center_s_m']==v) for v in pos]
fig,ax=plt.subplots(figsize=(7,7),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=8,cmap='Blues',aspect='auto');ax.set_xticks(range(5),['0','0.25','0.5','0.75','1']);ax.set_yticks(range(len(pos)),[f'{v:.2f} m / {h:.1f} deg' for v,h in zip(pos,headings)]);ax.set(xlabel='Fixed alpha',ylabel='Ridge position / reference heading at center',title='Suitable alpha changes with bend phase\nFull 10 s count / 8 matched physics blocks')
for i in range(len(pos)):
 for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>=5 else 'black')
fig.colorbar(im,ax=ax,label='Full 10 s episodes / 8');fig.savefig(out/'position_alpha_windows.png');plt.close(fig)
print(json.dumps(dict(slope=slope,wheelbase=wheel,obliquity=oblique),indent=2))
