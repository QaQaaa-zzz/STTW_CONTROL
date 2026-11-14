"""Matched full-departure prediction comparison across all model seeds and DEV cases."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
m=json.loads((a.root/'manifest.json').read_text());models=[];rows=[];conditions=None;case_rows=[]
for e in m['entries']:
    folder=Path(e['output']);result=json.loads((folder/'summary.json').read_text());cases=result['cases'];keys=[(r['evaluation'],r['case'],r['anchor_s'],r['steps']) for r in cases]
    if conditions is None:conditions=keys
    else:assert keys==conditions,'Model conditions differ'
    values=np.array([r['max_xy_m'] for r in cases]);support=np.array([r['max_support_xyz_m'] for r in cases])
    rows.append(dict(model=e['name'],cases=len(cases),mean_max_xy_m=float(values.mean()),p95_max_xy_m=float(np.quantile(values,.95)),worst_xy_m=float(values.max()),worst_support_xyz_m=float(support.max()),min_horizon_s=min(r['observed_horizon_s'] for r in cases),max_horizon_s=max(r['observed_horizon_s'] for r in cases),checkpoint=e['checkpoint'],model_root=e['model_root'],checkpoint_sha256=result['checkpoint_sha256']))
    case_rows.extend(dict(model=e['name'],**r) for r in cases);models.append(dict(name=e['name'],cases=cases,arrays=dict(np.load(folder/'predictions.npz'))))
reference=models[0];cases=reference['cases']
for source_id,source in enumerate(dict.fromkeys(r['evaluation'] for r in cases)):
    selected=[i for i,r in enumerate(cases) if r['evaluation']==source];cfg=json.loads((Path(source)/'resolved_config.json').read_text());bank=np.load(cfg['ground_handoff']['reference_path'])
    fig,axes=plt.subplots((len(selected)+1)//2,2,figsize=(14,len(selected)),layout='constrained')
    for ax,i in zip(axes.flat,selected):
        r=cases[i];end=r['steps']+1
        with np.load(Path(source)/r['case']/'trajectory.npz') as d:actual=d['qpos'][round(r['anchor_s']/.005):,:2]
        ref=bank['xy'][r['route']];keep=(ref[:,0]>=actual[:,0].min()-.1)&(ref[:,0]<=actual[:,0].max()+.1)
        ax.plot(ref[keep,0],ref[keep,1],':',color='gray',label='original reference');ax.plot(actual[:,0],actual[:,1],color='black',lw=2,label='actual 5ms');ax.plot(*actual[-1],marker='o' if r['physical_outcome']=='goal' else 'x',color='black')
        for model in models:
            name=model['name'];pred=model['arrays']['predicted'][i,:end];color='#009e73' if name=='M0' else '#0072b2' if name.startswith('residual') else '#d55e00';style='-.' if name=='M0' else ['-','--',':'][int(name[-1])]
            ax.plot(pred[:,0],pred[:,1],style,color=color,lw=1,label=name)
        ax.set(title=f"{r['case']} {r['request']}m/s {r['observed_horizon_s']:.2f}s",xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6,ncol=3);fig.savefig(a.output/f'map{source_id:02d}_paired_xy.png',dpi=130);plt.close(fig)
for name,data in [('models',rows),('cases',case_rows)]:
    with (a.output/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
(a.output/'summary.json').write_text(json.dumps(dict(models=rows,source=str(a.root),role=m['role']),indent=2)+'\n')
lines=['# 多地图模型：完整剩余离场预测配对比较','','两个既有DEV_MAP共72条件；全部模型同一个真实入口状态、历史、速度请求、路线和实际观察终点。每例3.22–4.00秒，超过训练最长3秒；不同例观察长度不同，不能把本表当共同固定4秒误差。第一/末个20ms样本与真实5ms事件端点最多相差15ms；实际XY画到真实终点，补齐区不计误差或成功。','']
for i in range(len({r['evaluation'] for r in cases})):lines+=['',f'![地图{i}全部模型同图实际/参考/预测XY](map{i:02d}_paired_xy.png)']
lines+=['','|模型|例数|平均最大XY m|P95 m|最差XY m|最差整车支持m|','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['model']}|{r['cases']}|{r['mean_max_xy_m']:.5f}|{r['p95_max_xy_m']:.5f}|{r['worst_xy_m']:.5f}|{r['worst_support_xyz_m']:.5f}|")
lines+=['','这是已知地图下离线预测比较，未增加新物理成功率，尚无规划器采用结论。M1/M2的三个初始化全部保留，不能把最好种子的某一项误差当统一优势。每例结果见cases.csv，存档身份见models.csv，原始预测和有效样本mask保留在外盘。', '',f'原始完整数组、逐模型报告与冻结合同：`{a.root}`。']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rows,indent=2))
