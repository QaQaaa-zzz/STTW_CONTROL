import argparse,json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run;out=a.output;out.mkdir(parents=True,exist_ok=False);records=[]
for alpha in [0,1]:
 ds={f:np.load(r/f'alpha{alpha}_t1_F{f:+g}.npz') for f in [0,.5,-.5,4,-4,6,-6]}
 z=ds[0]['phi'];s=ds[.5]['phi']-ds[-.5]['phi'];h=(ds[.5]['phi']+ds[-.5]['phi']-2*z)/.25
 for f in [4,-4,6,-6]:
  actual=ds[f]['phi'];linear=z+f*s;quad=linear+.5*f*f*h;k=np.argmax(abs(actual));records.append(dict(alpha=alpha,force_N=f,actual_substep_peak=float(ds[f]['peak_roll'].max()),actual_tick_peak=float(abs(actual).max()),linear_peak=float(abs(linear).max()),quadratic_peak=float(abs(quad).max()),linear_max_error=float(abs(actual-linear).max()),quadratic_max_error=float(abs(actual-quad).max()),actual_peak_time=float(ds[f]['time'][k])))
 if alpha==0:
  fig,axs=plt.subplots(1,2,figsize=(12,4));t=ds[0]['time'];mask=(t>=.9)&(t<=3.)
  for f in [0,-.5,-4,-6]:axs[0].plot(t[mask],ds[f]['phi'][mask],label=f'{f:g} N actual')
  axs[0].plot(t[mask],(z-6*s)[mask],'k--',label='-6 N linear prediction');axs[0].axhline(-.302,color='r',ls=':');axs[0].set(xlabel='time [s]',ylabel='signed roll [rad]',title='alpha0: false-safe prediction');axs[0].legend(fontsize=7)
  axs[1].plot(t[mask],(ds[-6]['phi']-(z-6*s))[mask]);axs[1].set(xlabel='time [s]',ylabel='actual minus predicted roll [rad]',title='linear extrapolation residual')
  for ax in axs:ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(out/'false_safe.png',dpi=160)
(out/'diagnostic.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
