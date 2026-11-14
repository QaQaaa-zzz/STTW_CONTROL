"""Report every calibration trajectory and the paired support-error distribution."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--errors',type=Path,required=True);a=p.parse_args();manifest=json.loads((a.collection/'manifest.json').read_text());records=json.loads((a.errors/'records.json').read_text());summary=json.loads((a.errors/'summary.json').read_text());margins=json.loads((a.errors/'margins.json').read_text());h=max(x['horizon_s'] for x in summary);bank=np.load(a.collection/'routes.npz');lines=['# E1整车平面支持界校准','','只用于固定轴对齐道路/禁止矩形的整车平面边界，不是一般三维表面的Hausdorff距离，也不是安全概率保证。误差取每条完整预测时域内所有碰撞几何x/y上下界的最大差，保持几何与动态方法、同根/曲线/地图配对。所有初态已用于DEV；同根的八条曲线相关，非独立TEST。','',f'本轮最大时域{h}s，从reset后20ms开始，不能外推为任意滚动初态或RB回交后模型。裕量规则在评估前冻结：每方法取DEV组P95误差的最大值，再加0.02m地图/离散检查余量。未根据E1成败调节。','']
for entry in manifest['entries']:
 if entry['split'] not in ['DEV_ROOT','DEV_MAP']:continue
 root=Path(entry['output']);status=json.loads((root/'status.json').read_text());fig,axes=plt.subplots(4,2,figsize=(16,12),layout='constrained')
 for row in status['results']:
  data=np.load(root/row['case_dir']/'trajectory.npz');xy=data['qpos'][:,:2];ax=axes.flat[row['route']];ax.plot(*xy.T,lw=.8,alpha=.7);ax.plot(*xy[-1],'.' if row['termination']=='timeout' else 'x',color='black',ms=3)
 for route,ax in enumerate(axes.flat):
  ref=bank['xy'][route];mask=(ref[:,0]>=1.2)&(ref[:,0]<=10.2);ax.plot(*ref[mask].T,'k--',lw=.8);ax.set(title=f'Curve{route}:8 roots',xlabel='World x(m)',ylabel='World y(m)',xlim=(1.2,10.2),ylim=(-1,1),aspect='equal');ax.grid(alpha=.2)
 fig.suptitle(entry['name']+' | actual solid, reference dashed; calibration window is not task success');path=a.errors/(entry['name']+'_xy.png');fig.savefig(path,dpi=150);plt.close(fig);lines+=['![全部实际/参考XY]('+path.name+')','']
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,entry in zip(axes,manifest['entries']):
 for method,color in [('geometry','#0072B2'),('dynamic','#D55E00')]:
  z=sorted(x['max_planar_support_error_m'] for x in records if x['method']==method and x['group']==entry['name'] and x['horizon_s']==h);ax.step(z,np.arange(1,len(z)+1)/len(z),where='post',label=method,color=color)
 ax.set(title=entry['name'],xlabel=f'Max {h}s planar support error(m)',ylabel='Empirical trajectory fraction',ylim=(0,1));ax.grid(alpha=.2)
axes[0].legend();fig.savefig(a.errors/'support_error_cdf.png',dpi=160);plt.close(fig);lines+=['![逐组误差分布](support_error_cdf.png)','','|Method|Group|Horizon s|Trajectories/roots|P95 support m|Worst support m|P95 historical global3D bound m|','|---|---|---:|---:|---:|---:|---:|']
for x in summary:lines.append(f'|{x["method"]}|{x["group"]}|{x["horizon_s"]}|{x["trajectories"]}/{x["roots"]}|{x["p95_support_error_m"]:.4f}|{x["max_support_error_m"]:.4f}|{x["p95_global3d_bound_m"]:.4f}|')
lines+=['','|Method|Horizon s|Frozen margin m|','|---|---:|---:|']
for x in margins:lines.append(f'|{x["method"]}|{x["horizon_s"]}|{x["margin_m"]:.4f}|')
lines+=['','逐轨迹误差见records.json，缺失/完整窗口见coverage.json，预算和适用域见protocol.json。先前3.8秒校准与本次4.5秒延长分目录保留；后者同一批192个DEV初态/曲线重放到4.6秒，不能称新增独立初态。无策略或模型续训。']
(a.errors/'RESULTS.md').write_text('\n'.join(lines)+'\n')
