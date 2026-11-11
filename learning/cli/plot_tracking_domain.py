"""Original-route tracking-domain geometry, with per-lane frozen boundary slices."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
metrics=json.loads((a.run/'analysis_summary.json').read_text())['rows'];rows=[];index=['# 跟踪域：实际轨迹、状态平面与前视余量','','每个工况首先给出实际/原参考XY。状态平面的虚线是各车道临界时刻冻结的局部边界，曲线路径及变化前视距离没有被替换成统一直线。右图逐帧用实际速度、原路径和位姿重建前视点前向距离减0.05m。×为真实提前终止，圆点为完整记录末点。','','θ=车辆航向−原路径切向；e=车辆相对原路径插值点的左向位移。令A为前视目标相对车辆的切向距离，C为目标相对原路径插值点的左向位移，则前向距离为 A cosθ+(C−e) sinθ。冻结A、C时，边界满足该式=0.05m；另要求|θ|<90°。','']
colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#777777'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
for job in json.loads((a.run/'jobs.json').read_text()):
    folder=a.run/job['name'];c=json.loads((folder/'contract.json').read_text());z=np.load(folder/'trajectory.npz');route=np.load(Path(c['bundle'])/'route.npz');f=json.loads((Path(c['bundle'])/'config.json').read_text())['follower'];n=len(c['settings']['alphas']);tr=z['lower_trace'].reshape(-1,n,11)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');route_mask=route['s']<=float(z['progress'].max())+2.;axes[0].plot(*route['xy'][route_mask].T,'k--',lw=1,label='Original route (travelled extent)');phase=[];plotted=0
    for lane,label in enumerate(c['settings']['case_labels']):
        ids=np.flatnonzero(tr[:,lane,6]);v=tr[ids,lane,2];pose=z['pose'][ids,lane];progress=z['progress'][ids,lane];heading=np.interp(progress,route['s'],route['heading']);theta=(pose[:,2]-heading+np.pi)%(2*np.pi)-np.pi
        center=np.stack([np.interp(progress,route['s'],route['xy'][:,k]) for k in [0,1]],-1);tangent=np.stack([np.cos(heading),np.sin(heading)],-1);normal=np.stack([-np.sin(heading),np.cos(heading)],-1)
        look=np.clip(f['lookahead_time_s']*np.maximum(v,f['lookahead_min_speed_m_s']),f['lookahead_min_m'],f['lookahead_max_m']);target=np.stack([np.interp(progress+look,route['s'],route['xy'][:,k]) for k in [0,1]],-1)
        e=np.sum((pose[:,:2]-center)*normal,axis=1);A=np.sum((target-pose[:,:2])*tangent,axis=1);C=np.sum((target-center)*normal,axis=1);forward=A*np.cos(theta)+(C-e)*np.sin(theta)
        direct=np.sum((target-pose[:,:2])*np.stack([np.cos(pose[:,2]),np.sin(pose[:,2])],-1),axis=1);m=next(r for r in metrics if r['case']==job['name'] and r['label']==label);margin=forward-f['preview_body_x_min_m'];critical=int(np.argmin(margin));time=(ids+1)*c['lower_dt']
        rows.append(dict(case=job['name'],label=label,cause=m['cause'],min_forward_margin_m=float(margin.min()),terminal_forward_margin_m=float(margin[-1]),terminal_heading_deg=float(np.degrees(theta[-1])),terminal_roll_deg=float(np.degrees(tr[ids[-1],lane,4])),geometry_identity_error_m=float(np.max(abs(forward-direct))),source=str(folder/'trajectory.npz')))
        if label not in ['full_0','full_0.25','full_0.5','full_0.75','full_1','Zero']:continue
        color=colors[plotted];plotted+=1;terminal='x' if m['failed'] else 'o';stride=4
        axes[0].plot(*pose[::stride,:2].T,color=color,lw=1.2,label=label);axes[0].scatter(*pose[-1,:2],color=color,marker=terminal,s=35)
        axes[1].plot(np.degrees(theta[::stride]),e[::stride],color=color,lw=1.2);axes[1].scatter(np.degrees(theta[-1]),e[-1],color=color,marker=terminal,s=35)
        phase.append((e,theta,A[critical],C[critical],color));axes[2].plot(time[::stride],margin[::stride],color=color,lw=1.2);axes[2].scatter(time[-1],margin[-1],color=color,marker=terminal,s=35)
    all_e=np.concatenate([x[0] for x in phase]);lim=(all_e.min()-.15,all_e.max()+.15);angle=np.linspace(-89.9,89.9,1000);rad=np.radians(angle)
    for e,theta,A,C,color in phase:
        boundary=(A*np.cos(rad)+C*np.sin(rad)-f['preview_body_x_min_m'])/np.sin(rad);boundary[abs(rad)<.01]=np.nan;axes[1].plot(angle,boundary,':',color=color,alpha=.65,lw=1)
    axes[0].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual and original route');axes[0].set_aspect('equal',adjustable='box');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Heading error theta (deg)',ylabel='Left path displacement e (m)',title='State plane; dotted = frozen boundary',ylim=lim,xlim=(-95,95));axes[1].axvline(-90,color='.5',ls='--',lw=.8);axes[1].axvline(90,color='.5',ls='--',lw=.8)
    axes[2].set(xlabel='Time (s)',ylabel='Forward preview margin (m)',title='Reconstructed moving-domain margin');axes[2].axhline(0,color='.2',lw=.8)
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle(job['name']+' | frozen H75 / Local300 | original full-path start',fontsize=12);out=a.output/(job['name']+'.png');fig.savefig(out);plt.close(fig);index+=['## '+job['name'],'',f'![]({out.name})','']
with (a.output/'domain_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
index+=['## 全216条检查','',f"几何恒等式最大误差：{max(r['geometry_identity_error_m'] for r in rows):.3g} m。终止原因直接读取原分析，不根据重建曲线重新定义失败。采样是200Hz记录，控制判据和记账时刻可能相差一帧。",'', '[全车道数值与原始数据路径](domain_summary.csv)']
(a.output/'README.md').write_text('\n'.join(index)+'\n');print('records',len(rows),'domain',sum('domain' in r['cause'] for r in rows),'identity_max',max(r['geometry_identity_error_m'] for r in rows))
