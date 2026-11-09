"""Audit matched initial conditions and plot both models across five alpha values."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'plots';out.mkdir(exist_ok=True)
alphas=['0','0.25','0.5','0.75','1'];models=['before','after'];data={};base=None;state=None;properties=None
for model in models:
 for alpha in alphas:
  src=run/f'{model}_{alpha}';case=json.loads((src/'initial_cases.json').read_text());assert np.allclose(case.pop('alpha'),float(alpha));current=np.load(src/'initial_state.npz');props=json.loads((src/'randomization.json').read_text())
  if base is None:base=case;state={k:current[k] for k in current.files};properties=props
  else:
   assert case==base, f'Initial case mismatch: {src}'
   assert props==properties, f'Randomized physics mismatch: {src}'
   for key in state:np.testing.assert_allclose(current[key],state[key],rtol=0,atol=1e-6,err_msg=f'{src}: {key}')
  data[model,alpha]=np.load(src/'trajectory.npz')
spec=json.loads((run/'config.json').read_text());routes=np.load(Path(spec['training_bank'])/'routes.npz');colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00'];styles=['-','--','-.',':','-'];records=[];metrics=[]
for lane,rid in enumerate(base['route_ids']):
 fig,axs=plt.subplots(5,2,figsize=(14,18));bounds=[];speed=float(routes['speed'][rid]);ref=routes['xy'][rid]
 for col,model in enumerate(models):
  for alpha,color,style in zip(alphas,colors,styles):
   d=data[model,alpha];tr=d['lower_trace'][:,:,lane].reshape(-1,11);valid=tr[:,6]>0;ids=np.flatnonzero(valid);tr=tr[ids];time=ids*.005+tr[:,7]*.0002;steps=np.unique(ids//4);reward=d['reward'][steps,lane];label=f'alpha={alpha}'
   axs[0,col].plot(tr[:,0],tr[:,1],c=color,ls=style,label=label);axs[0,col].scatter(tr[-1,0],tr[-1,1],c=color,marker='x');bounds.append(tr[:,:2])
   for row,xx,yy in [(1,time,tr[:,2]),(2,time,tr[:,3]),(3,steps,reward),(4,steps,np.cumsum(reward))]:axs[row,col].plot(xx,yy,c=color,ls=style,label=label)
   metrics.append(dict(lane=lane,model=model,alpha=float(alpha),route=int(rid),terrain=base['terrain_ids'][lane],early_terminated=bool(d['failure'][:,lane].any()),contact_observed=bool(d['terrain_contacted'][lane]),path_rmse=float(np.sqrt(np.mean(tr[:,3]**2))),speed_rmse=float(np.sqrt(np.mean((tr[:,2]-speed)**2))),observed_s=float(time[-1])))
  axs[0,col].plot(ref[:,0],ref[:,1],':',color='black',label='Original reference');axs[0,col].set_aspect('equal',adjustable='box');axs[0,col].set(xlabel='x (m)',ylabel='y (m)',title='Original H75 u77' if model=='before' else 'Continued H75 update76');axs[0,col].legend(fontsize=8)
  axs[1,col].axhline(speed,c='black',ls=':',label='Requested speed');axs[1,col].legend(fontsize=8)
  for row,label in [(1,'Forward speed (m/s)'),(2,'Signed path error (m)'),(3,'Per-step reward'),(4,'Cumulative reward')]:axs[row,col].set(xlabel='time (s)' if row<3 else 'policy step',ylabel=label)
 xy=np.concatenate(bounds);lo=xy.min(0)-1;hi=xy.max(0)+1
 for ax in axs[0]:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
 for row in range(1,5):
  limits=[ax.get_ylim() for ax in axs[row]]
  for ax in axs[row]:ax.set_ylim(min(v[0] for v in limits),max(v[1] for v in limits))
 for ax in axs.flat:ax.grid(alpha=.2)
 fig.suptitle(f'Matched case {lane} | route {rid} | terrain {base["terrain_ids"][lane]} | request {speed:.3f} m/s\nSame initial state, route, terrain, physical randomization and RNG seed; endpoints retained\nReward weights depend on alpha; cross-alpha reward is not a common performance score')
 fig.tight_layout(rect=[0,0,1,.96]);name=f'case_{lane:02d}.png';fig.savefig(out/name,dpi=120);plt.close(fig);records.append(name)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'identity_check.json').write_text(json.dumps(dict(passed=True,checked='initial cases except alpha, randomized physical properties, root, dofs, pulse, servo gains and steering zero offsets',seed=spec['seed'],cases=len(base['route_ids']),models=models,alphas=alphas),indent=2));(out/'README.md').write_text('# 原H75与76轮续训模型：同工况五alpha\n\n每张左列原H75、右列76轮，五个alpha同图。依次为XY轨迹、速度、路径误差、每步奖励、累计奖励。真实终点截止；不把不同alpha加权奖励当共同物理性能排名。逐条初始条件和物理随机化核验通过；噪声使用相同seed和采样时序。探索性诊断，非独立泛化确认。\n\n'+'\n'.join(f'- [{x}]({x})' for x in records)+'\n\n[指标](metrics.json) · [配对核验](identity_check.json)\n')
