#!/usr/bin/env python3
"""Supplementary plots from saved integration NPZ; no simulation."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_command_reporting import reward_audit
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--upper-run',required=True);args=p.parse_args();r=Path(args.run);o=Path(args.output);u=Path(args.upper_run)
for case in ['straight_hold','fast_turn']:
 for a in [0,1]:
  ds=[dict(np.load(r/case/f'alpha{a}.npz')),dict(np.load(u/'evaluationbest'/case/f'alpha{a}.npz')),dict(np.load(u/'evaluationbest'/case/'B0.npz'))];names=['upper best + fixed300','same upper best + R196','zero-upper R196, same-alpha rescoring'];parts=[reward_audit(d,a,u,16,include_reconstruction=True) for d in ds]
  fig,axes=plt.subplots(3,2,figsize=(17,14));keys=list(parts[0]);colors=plt.cm.tab20(np.linspace(0,1,len(keys)));data={}
  for i,(d,p) in enumerate(zip(ds,parts)):
   n=len(d['time']);steps=np.arange(1,n+1)
   for key,col in zip(keys,colors):
    value=p[key];axes[i,0].plot(steps,value,color=col,label=key);axes[i,1].plot(steps,np.cumsum(value),color=col,label=key);data[f'method{i}__'+key]=value
   for j in [0,1]:axes[i,j].set_title(names[i]+(' signed step components' if j==0 else ' cumulative components'));axes[i,j].set_xlabel('Control steps');axes[i,j].set_ylabel('Reward');axes[i,j].legend(ncol=3,fontsize=6);axes[i,j].grid(alpha=.25)
   data[f'method{i}__time']=d['time']
  fig.suptitle(f'{case} alpha{a} | original V5.2 reward independently reconstructed | no reward change');fig.tight_layout(rect=(0,0,1,.97));fig.savefig(o/f'{case}_alpha{a}_reward_parts.png',dpi=120);plt.close(fig)
  np.savez_compressed(o/'data'/case/f'alpha{a}_reward_parts.npz',**data)
  fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
  for d,name,ls in zip(ds,names,['-',':','--']):
   axes[0].plot(d['time'],d['actual_forward_speed'],ls,label=name+' true body speed');axes[0].plot(d['time'],d['wheel_speed_proxy'],ls,lw=.6,alpha=.7,label=name+' rear axle x0.1m');axes[1].plot(d['time'],d['wheel_speed_proxy']-d['actual_forward_speed'],ls,label=name)
  axes[0].plot(ds[0]['time'],ds[0]['limited_command'][:,0],'k--',label='raw');axes[0].plot(ds[0]['time'],ds[0]['governed'][:,0],color='.5',label='300 governed');axes[0].set_ylabel('m/s');axes[1].set_ylabel('wheel proxy - true (m/s)');axes[1].set_xlabel('Time (s)')
  for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.25)
  fig.suptitle(f'{case} alpha{a} | real body forward speed vs measured wheel proxy | fixed preparation');fig.tight_layout();fig.savefig(o/f'{case}_alpha{a}_speed_proxy.png',dpi=120);plt.close(fig)
print('Four paired reward-component and four speed-proxy figures saved')
fig,axes=plt.subplots(4,3,figsize=(17,18))
for i,(case,a,h) in enumerate([('straight_hold',0,10),('straight_hold',1,10),('fast_turn',0,16),('fast_turn',1,16)]):
    d=np.load(r/case/f'alpha{a}.npz');b=np.load(u/'evaluationbest'/case/f'alpha{a}.npz');base=np.load(u/'evaluationbest'/case/'B0.npz');t=d['time'];ax=axes[i,0]
    ax.plot(*d['reference_xy'].T,'k--',label='fixed original reference');ax.plot(*d['actual_xy'].T,label='upper best + lower300');ax.plot(*b['actual_xy'].T,':',label='same upper best + R196');ax.plot(*base['actual_xy'].T,color='.5',ls=':',label='zero-upper R196')
    ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_title(f'{case} alpha{a} {h}s | actual XY')
    for j,ch in enumerate(['speed','steer']):
        ax=axes[i,j+1];actual=d['actual_forward_speed'] if j==0 else d['actual_delta'];old=b['actual_forward_speed'] if j==0 else b['actual_delta']
        ax.plot(t,d['limited_command'][:,j],'k--',label='raw');ax.plot(t,d['governed'][:,j],label='governed sent to lower300');ax.plot(t,actual,label='lower300 actual');ax.plot(b['time'],old,':',label='old upper+R196 actual');ax.set_xlabel('Time (s)');ax.set_ylabel('m/s' if j==0 else 'rad');ax.set_title(ch)
        for row in ([1.,4.] if case=='fast_turn' else []):ax.axvline(row,color='.6',alpha=.5)
        if h==16:ax.axvline(10,color='r',ls=':',label='10s primary boundary')
    for ax in axes[i]:ax.legend(fontsize=7);ax.grid(alpha=.25)
fig.suptitle('Four completed diagnostic-only combinations | same fixed300 Actor | upper0 best added200 / upper1 best added250\nseed77001, identical complete prepared state | 200Hz lower / 50Hz upper | raw reference unchanged | training0')
fig.tight_layout(rect=(0,0,1,.965));fig.savefig(o/'FOUR_CONTROL_TRAJECTORIES.png',dpi=130);plt.close(fig)
