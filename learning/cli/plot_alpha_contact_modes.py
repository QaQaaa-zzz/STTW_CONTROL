"""Full case coverage: original XY, roll, Euler rate, command, and contact moment."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
labels=['full_0','full_0.25','full_0.5','full_0.75','full_1','Zero'];colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#777777']
lines=['# 全工况五档接触恢复图','','每个工况先展示实际/原参考XY，再展示同一时间轴的侧倾、Euler侧倾角速度、前轮转向速度命令与后轮接触侧倾力矩。×表示提前终止。虚线±3rad/s是命令限制。接触力矩包含法向和摩擦，投影到负车架x轴。','']
for path in sorted(a.source.glob('*_seed*.npz')):
 z=np.load(path);fields=list(z['contact_fields']);fig=plt.figure(figsize=(14,10),layout='constrained');gs=fig.add_gridspec(4,2,width_ratios=[1,1.7]);xy=fig.add_subplot(gs[:2,0]);zoom=fig.add_subplot(gs[2:,0]);axes=[fig.add_subplot(gs[i,1]) for i in range(4)]
 contract=json.loads((a.source.parent/path.stem/'contract.json').read_text())
 with np.load(Path(contract['bundle'])/'route.npz') as route:
  for axis in [xy,zoom]:axis.plot(*route['xy'].T,'k:',label='Original path')
 for label,color in zip(labels,colors):
  lane=list(z['labels']).index(label);valid=z['valid'][:,lane];t=z['time'][valid,lane];last=len(t)-1
  xx=z['xy'][valid,lane];xy.plot(*xx.T,color=color,label=label,lw=1);zoom.plot(*xx.T,color=color,lw=1)
  values=[np.degrees(z['phi'][valid,lane]),np.degrees(z['euler_roll_rate'][valid,lane]),z['command'][valid,lane],z['contact_mean'][valid,lane,fields.index('rear_normal_roll_Nm')]+z['contact_mean'][valid,lane,fields.index('rear_friction_roll_Nm')]]
  if t[-1]<9.99:
   xy.scatter(*xx[-1],marker='x',color=color);zoom.scatter(*xx[-1],marker='x',color=color)
  for ax,y in zip(axes,values):
   ax.plot(t,y,color=color,lw=1,label=label)
   if t[-1]<9.99:ax.scatter(t[-1],y[-1],marker='x',color=color)
  # First raised front and rear contact marked on their corresponding roll trace.
  for wheel,marker in [('front','o'),('rear','s')]:
   hits=np.flatnonzero(z['contact_mean'][valid,lane,fields.index(wheel+'_raised')]>0)
   if len(hits):j=hits[0];axes[0].scatter(t[j],values[0][j],marker=marker,s=18,color=color)
 for ax,title in zip(axes,['Roll (deg)','Euler roll rate (deg/s)','Steering-rate command (rad/s)','Rear contact roll moment (Nm)']):
  ax.set_ylabel(title);ax.set_xlim(.8,3.3);ax.grid(alpha=.2)
 axes[2].axhline(3,color='.5',ls='--');axes[2].axhline(-3,color='.5',ls='--');axes[-1].set_xlabel('Recorded time (s)')
 for axis,mask,title in [(xy,z['valid'],'Full observed XY'),(zoom,z['valid']&(z['time']<=3.3),'Encounter XY (to 3.3s)')]:
  points=z['xy'][mask];lo=points.min(0)-.5;hi=points.max(0)+.5;axis.set(xlabel='World x (m)',ylabel='World y (m)',title=title,xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));axis.set_aspect('equal');axis.grid(alpha=.2)
 xy.legend(fontsize=7)
 fig.suptitle(path.stem+' | original frozen H75 + Local300\nCircle/square: first raised front/rear contact; full XY, event-time detail')
 name=path.stem+'_comparison.png';fig.savefig(a.output/name,dpi=140);plt.close(fig);lines+=['## '+path.stem,'','![]('+name+')','']
(a.output/'FIGURES.md').write_text('\n'.join(lines)+'\n');print('case figures',len(list(a.output.glob('*_comparison.png'))))
