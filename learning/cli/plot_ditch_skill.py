"""Actual-state XY/XZ, contact and reward diagnostics; no reconstructed successful motion."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.run/'config.json').read_text());terrain=cfg['ditch_terrain'];front=terrain['front_x_m'];back=front+terrain['width_m'];width_cm=100*terrain['width_m']
verified=cfg.get('verified_ditch_reward',False)
cases=json.loads((a.run/'summary.json').read_text())['cases'];colors=['#0072B2','#D55E00','#009E73'];styles=['-','--','-.']
mesh_path=a.run/'terrain/terrain.npz'
if cfg.get('ditch_randomization'):
 if len({c['terrain_id'] for c in cases})!=1:raise ValueError('Shared ground profile requires matched terrain IDs')
 mesh_path=a.run/'terrain'/str(cases[0]['terrain_id'])/'terrain.npz'
mesh=np.load(mesh_path);verts=mesh['vertices'];profile=verts[np.abs(verts[:,1])<1e-6];profile=profile[np.argsort(profile[:,0])]
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained'); xy,xz,v,reward,cumulative,height=axes.ravel()
for i,c in enumerate(cases):
 name=c['candidate']['name'];d=np.load(a.run/name/'trajectory.npz');r=d['root'][:,0];t=d['time'];color=colors[i];style=styles[i];label=f"{name} (end {t[-1]:.2f}s, code {c['terminal_code']})"
 xy.plot(r[:,0],r[:,1],style,color=color,label=label);xy.scatter(*r[-1,:2],color=color,marker='x',s=60)
 xz.plot(r[:,0],r[:,2],style,color=color,label=name);xz.scatter(r[-1,0],r[-1,2],color=color,marker='x',s=60)
 v.plot(t,r[:,7],style,color=color,label=name);reward.plot(t,d['reward'][:,0],style,color=color,label=name)
 cumulative.plot(t,np.cumsum(d['reward'][:,0]),style,color=color,label=name)
 if verified:
  diagnostic=np.load(a.run/name/'verified_diagnostics.npz');height.step(diagnostic['time'],diagnostic['stage'][:,0],where='post',color=color,linestyle=style,label=name)
 else:height.plot(t,d['reward_height'][:,0],style,color=color,label=name)
for ax in [xy,xz]:
 ax.axvspan(front,back,color='0.8',alpha=.6,label=f'open ditch {width_cm:g}cm');ax.set_xlabel('World x (m)');ax.set_xlim(1.2,max(4.8,max(c['max_root_x'] for c in cases)+.3));ax.set_aspect('equal',adjustable='box')
xy.plot([1.2,4.8],[0,0],':',color='black',label='original route y=0');ys=np.concatenate([line.get_ydata() for line in xy.lines]);xy.set_ylim(min(-.4,float(ys.min())-.1),max(.6,float(ys.max())+.1));xy.set_ylabel('World y (m)');xy.set_title('Actual XY, all declared candidates');xy.legend(fontsize=7,loc='upper left')

for bank in [profile[profile[:,0]<=front],profile[profile[:,0]>=back]]:xz.plot(bank[:,0],bank[:,2],color='saddlebrown')
xz.set_ylim(-.25,.7);xz.set_ylabel('Root z (m)');xz.set_title('Actual root XZ; x marks true episode end')
for ax,title,y in [(v,'Actual forward velocity','vx (m/s)'),(reward,'Per-step emitted reward','reward / step'),(cumulative,'Cumulative emitted reward','reward'),(height,'Verified contact stage' if verified else 'Inherited height shaping alone','stage 0-4' if verified else 'reward / step')]:
 ax.set_title(title);ax.set_xlabel('Time (s)');ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle(f'{width_cm:g}cm rough-bank ditch | fixed evaluation seed 20261109 | three frozen checkpoints\nSingle initial-condition diagnostic, not a success-rate comparison; no ground-controller handoff',fontsize=12)
fig.savefig(a.run/'xy_and_rewards.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
for i,c in enumerate(cases):
 name=c['candidate']['name'];d=np.load(a.run/name/'physics_contact.npz');names=d['body_names'].tolist()
 for wheel,style in [('frontwheel','-'),('rearwheel','--')]:
  force=d['contact'][:,0,names.index(wheel),2];ax[i].plot(d['time'],force,style,label=wheel)
 ax[i].set_title(name);ax[i].set_ylabel('Vertical force (N)');ax[i].legend();ax[i].grid(alpha=.2)
ax[-1].set_xlabel('Time (s), solved physics steps 0.0025s');fig.savefig(a.run/'wheel_contacts.png',dpi=150);plt.close(fig)
# Side-view animation of measured body centers and wheel positions, not a renderer video.
fig,axs=plt.subplots(3,1,figsize=(9,7),layout='constrained');datasets=[];artists=[]
for i,c in enumerate(cases):
 name=c['candidate']['name'];d=np.load(a.run/name/'physics_contact.npz');datasets.append(d)
 axs[i].axvspan(front,back,color='0.85');axs[i].plot(profile[profile[:,0]<=front,0],profile[profile[:,0]<=front,2],color='saddlebrown');axs[i].plot(profile[profile[:,0]>=back,0],profile[profile[:,0]>=back,2],color='saddlebrown');axs[i].set(xlim=(1.2,max(4.8,max(c['max_root_x'] for c in cases)+.3)),ylim=(-.3,.85),ylabel='z (m)',title=name);axs[i].set_aspect('equal')
 points,=axs[i].plot([],[],'o',color=colors[i],ms=5);wheels,=axs[i].plot([],[],'o',mfc='none',mec=colors[i],ms=13);text=axs[i].text(.02,.86,'',transform=axs[i].transAxes);artists.append((points,wheels,text))
axs[-1].set_xlabel('x (m)');fig.suptitle('Measured motion schematic (body centers + wheels); stops at actual termination')
frames=np.arange(0,max(d['time'][-1] for d in datasets)+.04,.04)
def animate(t):
 for d,(points,wheels,text) in zip(datasets,artists):
  j=min(np.searchsorted(d['time'],t),len(d['time'])-1);b=d['body_state'][j,0];names=d['body_names'].tolist();ids=[names.index(w) for w in ('frontwheel','rearwheel')]
  points.set_data(b[:,0],b[:,2]);wheels.set_data(b[ids,0],b[ids,2]);text.set_text(f"t={d['time'][j]:.3f}s"+(' TERMINATED' if t>=d['time'][-1] else ''))
 return [x for row in artists for x in row]
FuncAnimation(fig,animate,frames=frames,interval=40,blit=False).save(a.run/'actual_motion.gif',writer=PillowWriter(fps=25));plt.close(fig)
