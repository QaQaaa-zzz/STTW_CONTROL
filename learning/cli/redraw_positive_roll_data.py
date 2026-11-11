"""Replot packaged positive examples without access to the experiment machine.
Usage: python redraw.py  (requires numpy and matplotlib)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
meta=json.loads((root/'metadata.json').read_text());cases=meta['cases'];datasets=[dict(np.load(root/'data'/c['file'])) for c in cases]
output=root/'redrawn';output.mkdir(exist_ok=True)
colours=['#0072B2','#D55E00','#009E73','#777777'];styles=['-','--','-.',':']
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2,'figure.dpi':130})
def decorate(ax,c,d):
    for e in c['events']:
        if e['label']!=c['positive']:continue
        ax.axvspan(float(e['start_s']),float(e['end_s']),color='gray',alpha=.18)
        if e['recovery_entry_s']:ax.axvline(float(e['recovery_entry_s']),color='#009E73',ls=':',lw=.7)
    for lane in range(4):
        t=d['time'][d['valid'][:,lane],lane][-1]
        if t<9.999:ax.axvline(t,color=colours[lane],ls='--',lw=.7)
    ax.axvspan(9,10,color='#56B4E9',alpha=.1);ax.set_xlim(0,10)
def lines(ax,d,key,factor=1):
    for lane,label in enumerate(d['labels']):
        mask=d['valid'][:,lane];ax.plot(d['time'][mask,lane],d[key][mask,lane]*factor,label=str(label),color=colours[lane],ls=styles[lane],lw=1)
def save(fig,name):
    fig.savefig(output/name);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,9),layout='constrained')
for ax,c,d in zip(axes,cases,datasets):
    ax.plot(*d['reference_xy'].T,'k--',label='Original reference')
    points=[]
    for lane,label in enumerate(d['labels']):
        xy=d['xy'][d['valid'][:,lane],lane];points.append(xy);ax.plot(*xy.T,label=str(label),color=colours[lane],ls=styles[lane]);ax.scatter(*xy[-1],color=colours[lane],marker='o' if d['valid'][:,lane].sum()==2000 else 'x')
    xy=np.concatenate(points);ax.set_xlim(xy[:,0].min()-.6,xy[:,0].max()+.6);ax.set_ylim(xy[:,1].min()-.6,xy[:,1].max()+.6);ax.scatter(*d['goal_point'],marker='*',color='black');ax.set_aspect('equal',adjustable='box');ax.set(title=c['title'],xlabel='x (m)',ylabel='y (m)');ax.legend(fontsize=7)
save(fig,'POSITIVE_XY.png')
fig,axes=plt.subplots(5,2,figsize=(15,17),sharex=True,layout='constrained')
for j,(c,d) in enumerate(zip(cases,datasets)):
    for i,(key,unit,scale) in enumerate([('phi','Roll (deg)',180/np.pi),('euler_roll_rate','Euler roll rate (rad/s)',1),('speed','Actual speed (m/s)',1),('delta','Steering (deg)',180/np.pi),('delta_dot','Steering rate (rad/s)',1)]):
        ax=axes[i,j];lines(ax,d,key,scale);decorate(ax,c,d);ax.set_ylabel(unit)
        bounds={0:np.degrees(.7),3:np.degrees(.8),4:3}
        if i in bounds:
            for limit in [-bounds[i],bounds[i]]:ax.axhline(limit,color='black',ls=':',lw=.6)
        if i==0:ax.set_title(c['title']);ax.legend(fontsize=7)
    axes[-1,j].set_xlabel('Time (s)')
fig.suptitle('Recovery states | gray: positive-lane raised contacts; blue: final hold | +/-3rad/s is COMMAND range')
save(fig,'POSITIVE_STATES.png')
fig,axes=plt.subplots(5,2,figsize=(15,17),sharex=True,layout='constrained')
for j,(c,d) in enumerate(zip(cases,datasets)):
    t=d['time'][:,0];valid=d['valid'][:,0];m=d['contact_mean'][:,0];low=d['contact_min'][:,0];high=d['contact_max'][:,0]
    for k,index in enumerate([0,6]):
        axes[0,j].plot(t[valid],m[valid,index],color=colours[k],label=['Front','Rear'][k]);axes[0,j].fill_between(t,low[:,index],high[:,index],color=colours[k],alpha=.12)
    for k,index in enumerate([4,10]):axes[1,j].plot(t[valid],(m[:,index]+m[:,index+1])[valid],color=colours[k],label=['Front','Rear'][k])
    axes[2,j].plot(t[valid],m[valid,15],label='Requested motor',ls=':');axes[2,j].plot(t[valid],m[valid,14],label='Applied motor')
    for b in [-1,1]:axes[2,j].axhline(b,color='black',ls='--',lw=.6)
    lines(axes[3,j],d,'path_error');lines(axes[4,j],d,'goal_distance')
    for b in [-c['path_task_bound_m'],c['path_task_bound_m']]:axes[3,j].axhline(b,color='black',ls=':',lw=.6)
    for lane,row in enumerate(c['outcomes']):
        if row['goal_crossing_s']:
            time=float(row['goal_crossing_s']);idx=np.argmin(abs(d['time'][:,lane]-time));axes[4,j].scatter(time,d['goal_distance'][idx,lane],marker='*',color=colours[lane])
    for i,title in enumerate(['Normal force (N)','Roll moment about COM (Nm)','Steering motor torque (Nm)','Path error (m)','Goal-anchor distance (m)']):
        ax=axes[i,j];decorate(ax,c,d);ax.set_ylabel(title);ax.legend(fontsize=7)
    axes[0,j].set_title(c['title']);axes[-1,j].set_xlabel('Time (s)')
fig.suptitle('Contact and task | force/torque panels: positive lane | goal-section crossing, not parking')
save(fig,'POSITIVE_CONTACT_TASK.png')
fig,axes=plt.subplots(3,2,figsize=(15,11),sharex=True,layout='constrained')
for j,(c,d) in enumerate(zip(cases,datasets)):
    valid=d['valid'][:,0];t=d['time'][valid,0];lines(axes[0,j],d,'z')
    for key,label in [('unstable','sqrt(a4)*abs(z)'),('restoring','Actual nominal steering correction')]:axes[1,j].plot(t,d[key][valid,0],label=label)
    for key,label in [('nominal_demand','Required rate: dot(z)=0,d=0'),('delta_dot','Actual rate'),('command','Pre-servo-clipping command')]:axes[2,j].plot(t,d[key][valid,0],label=label)
    for b in [-3,3]:axes[2,j].axhline(b,color='black',ls=':',lw=.6)
    for i,title in enumerate(['z (rad/s)','Nominal contribution (rad/s^2)','Steering rate (rad/s)']):
        ax=axes[i,j];decorate(ax,c,d);ax.set_ylabel(title);ax.legend(fontsize=7)
    axes[0,j].set_title(c['title']);axes[-1,j].set_xlabel('Time (s)')
fig.suptitle('Nominal recovery demand | excluded contact residual; command bounds are not safety guarantees')
save(fig,'POSITIVE_RECOVERY_DEMAND.png')
fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
for ax,c,d in zip(axes,cases,datasets):
    valid=d['valid'][:,0];t=d['time'][:,0];ax.plot(t[valid],np.degrees(d['phi'][valid,0]),label='Recorded')
    for key,label in [('pred_nominal100','Nominal100ms replay'),('pred_contact100','Other-seed contact correction')]:ax.plot(t,np.degrees(d[key][:,0]),label=label,ls='--')
    decorate(ax,c,d);ax.set(title=c['title'],ylabel='Roll (deg)');ax.legend()
axes[-1].set_xlabel('Time (s)');fig.suptitle('Overlapping100ms measured-input replays; NOT a full-episode alpha-only prediction')
save(fig,'POSITIVE_MODEL_CHECK.png')
print('Wrote five figures to',output)
