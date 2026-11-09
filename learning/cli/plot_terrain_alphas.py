"""Plot saved matched terrain cases; preserve endpoints and original reference."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args();out=a.evaluation/'alpha_comparison';out.mkdir(exist_ok=True);records=[]
for case in json.loads((a.evaluation/'index.json').read_text()):
 run=Path(case['run']);c=json.loads((run/'contract.json').read_text());d=np.load(run/'trajectory.npz');route=np.load(Path(c['bundle'])/'route.npz');fig,axs=plt.subplots(1,3,figsize=(16,5));bounds=[]
 for lane,alpha in enumerate(c['settings']['alphas']):
  t=d['lower_trace'][:,:,lane].reshape(-1,11);ids=np.flatnonzero(t[:,6]);t=t[ids];ts=ids*.005+t[:,7]*c['physics_dt'];label=f'alpha={alpha:g}' if lane<5 else 'Zero middle + Local300';style=['-','--','-.',':','-','--'][lane];color=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#555555'][lane]
  axs[0].plot(t[:,0],t[:,1],label=label,ls=style,c=color);axs[0].scatter(t[-1,0],t[-1,1],marker='x',c=color);axs[1].plot(ts,t[:,2],ls=style,c=color);axs[2].plot(ts,t[:,3],ls=style,c=color);bounds.append(t[:,:2])
 xy=np.concatenate(bounds);ref=route['xy'];lo=xy.min(0)-1;hi=xy.max(0)+1;axs[0].plot(ref[:,0],ref[:,1],':',c='black',alpha=.6,label='Original reference');axs[0].set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='x (m)',ylabel='y (m)');axs[0].set_aspect('equal',adjustable='box');axs[0].legend(fontsize=8)
 axs[1].axhline(case['user_speed_m_s'],ls=':',c='black',label='Requested speed');axs[1].set(xlabel='time (s)',ylabel='Actual forward speed (m/s)');axs[1].legend();axs[2].set(xlabel='time (s)',ylabel='Signed path error (m)')
 for ax in axs:ax.grid(alpha=.25)
 fig.suptitle(f'Original H75 u77 + Local300 | {case["terrain"]} | {case["user_speed_m_s"]} m/s | parameter seed {case["seed"]}\nFive alphas in the same case; x marks actual endpoints; not the continued update76 model')
 fig.tight_layout();name=f'{case["terrain"]}_{case["user_speed_m_s"]}_seed{case["seed"]}.png';fig.savefig(out/name,dpi=150);plt.close(fig);records.append(dict(image=name,source=str(run),checkpoint='original H75 u77',case=case))
(out/'manifest.json').write_text(json.dumps(records,indent=2));(out/'README.md').write_text('# 原H75同工况五alpha对照\n\n原H75 u77，非第76轮续训模型。所有曲线截止实际终点；Zero是零中层+Local300/ECBC/ESO。参考线仅按观察范围裁剪显示，原始参考不改。无平滑，无新仿真。\n\n'+'\n'.join(f'- [{r["image"]}]({r["image"]})' for r in records))
