"""Compare same-data predictor seeds and saved checkpoints without new simulation."""
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
import torch
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import NominalRollout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--geometry',type=Path);p.add_argument('--speed-geometry',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(1)
m=json.loads((a.root/'manifest.json').read_text());sources=[];rows=[];curves=[];case_rows=[];colors={'residual':'#0072b2','direct':'#d55e00'}
for entry in m['entries']:
    folder=Path(entry['output']);status=json.loads((folder/'status.json').read_text());best=json.loads((folder/'best.json').read_text());windows=json.loads((folder/'dev_windows.json').read_text());logs=[json.loads(s) for s in (folder/'scalars.jsonl').read_text().splitlines()]
    if a.previous:
        prior=a.previous/f"{entry['kind']}_seed{entry['seed']}";logs=[json.loads(s) for s in (prior/'scalars.jsonl').read_text().splitlines()]+logs
    validation={r['update']:r for r in logs if r['kind']=='validation'};vs=list(validation.values())
    curves.extend(dict(model=entry['kind'],seed=entry['seed'],**r) for r in logs)
    for horizon in [1.,3.]:
        selected=[r for r in windows if r['horizon_s']==horizon];errors=[r['max_xy_m'] for r in selected]
        rows.append(dict(model=entry['kind'],seed=entry['seed'],best_update=best['update'],horizon_s=horizon,windows=len(errors),cases=len({r['case'] for r in selected}),mean_max_xy_m=float(np.mean(errors)),p95_max_xy_m=float(np.quantile(errors,.95)),worst_xy_m=max(errors),last25_relative_mean_reduction=(vs[-2]['dev_mean_max_xy_m'][str(horizon)]-vs[-1]['dev_mean_max_xy_m'][str(horizon)])/vs[-2]['dev_mean_max_xy_m'][str(horizon)],best_at_budget_end=best['update']==max(validation)))
        for case in sorted({r['case'] for r in selected}):
            errors=[r['max_xy_m'] for r in selected if r['case']==case];case_rows.append(dict(model=entry['kind'],seed=entry['seed'],checkpoint=best['update'],horizon_s=horizon,case=case,windows=len(errors),mean_max_xy_m=float(np.mean(errors)),worst_xy_m=max(errors)))
    sources.append(dict(entry=entry,folder=folder,examples=dict(np.load(folder/'dev_examples.npz'))))
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for seed in range(3):
    for kind in colors:
        selected=[r for r in curves if r['model']==kind and r['seed']==seed];validation={r['update']:r for r in selected if r['kind']=='validation'}
        for h,style in [('1.0','-'),('3.0','--')]:axes[0,seed].plot(list(validation),[r['dev_mean_max_xy_m'][h] for r in validation.values()],style,color=colors[kind],label=f'{kind} {h}s')
        for h,style in [(5,':'),(15,'--'),(50,'-.'),(150,'-')]:
            train=[r for r in selected if r['kind']=='training' and r['horizon_steps']==h];values=np.array([r['loss'] for r in train]);smoothed=np.array([np.median(values[max(0,i-4):i+1]) for i in range(len(values))]);axes[1,seed].plot([r['update'] for r in train],smoothed,style,color=colors[kind],lw=1,label=f'{kind} H{h}')
    axes[0,seed].set(title=f'Seed{seed}: fixed DEV',xlabel='Update',ylabel='Mean maximum XY error (m)',yscale='log');axes[1,seed].set(title=f'Seed{seed}: supervised loss by horizon',xlabel='Update',ylabel='Loss, trailing5 per-horizon median',yscale='log')
    for ax in axes[:,seed]:ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
fig.savefig(a.output/'training_curves.png',dpi=150);plt.close(fig)
first=a.previous if a.previous else a.root
source_eval=Path(m['evaluation']);source_cfg=json.loads((source_eval/'resolved_config.json').read_text());schema=json.loads((source_eval/'model_state_schema.json').read_text());reference=dict(np.load(source_cfg['ground_handoff']['reference_path']))
cache=first/'m0_report_cache.json'
if cache.exists():
    baseline_rows=json.loads(cache.read_text());baseline_examples=dict(np.load(first/'m0_report_examples.npz'))
else:
    contract=json.loads((first/'residual_seed0/contract.json').read_text());nominal=NominalRollout(dict(np.load(first/'residual_seed0/nominal.npz')),contract['follower'],'cpu')
    baseline_rows=[];baseline_examples={};windows=json.loads((first/'residual_seed0/dev_windows.json').read_text())
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32)
    with torch.no_grad():
        for case in sorted({r['case'] for r in windows}):
            record=next(r for r in windows if r['case']==case)
            actual_root=Path(record.get('evaluation',str(source_eval)));actual_case=record.get('source_case',case)
            with np.load(actual_root/actual_case/'trajectory.npz') as d:pose,xi=states(d['qpos'][::4],d['qvel'][::4],schema);requests=d['trace'][::4,18]
            for h in [50,150]:
                chosen=[r for r in windows if r['case']==case and r['horizon_s']==h*.02]
                if not chosen:continue
                ids=np.array([round(r['anchor_s']/.02) for r in chosen]);route=chosen[0]['route'];n=len(ids)
                prediction,_=nominal(tensor(pose[ids]),tensor(xi[ids]),tensor(xi[ids-1]),tensor(reference['s']),tensor(reference['xy'][route])[None].expand(n,-1,-1),tensor(reference['heading'][route])[None].expand(n,-1),tensor(reference['curvature'][route])[None].expand(n,-1),h,speed_request=tensor(requests[ids]))
                actual=pose[ids[:,None]+np.arange(h+1)];error=np.linalg.norm(prediction.numpy()[:,:,:2]-actual[:,:,:2],axis=-1).max(1)
                baseline_rows.extend(dict(case=case,route=route,anchor_s=r['anchor_s'],horizon_s=h*.02,max_xy_m=float(error[i])) for i,r in enumerate(chosen))
                baseline_examples[f'{case}_h{h}_prediction']=prediction[0].numpy()
    cache.write_text(json.dumps(baseline_rows,indent=2)+'\n');np.savez_compressed(first/'m0_report_examples.npz',**baseline_examples)
example_keys=sorted(k for k in sources[0]['examples'] if k.endswith('_h50_actual'))
for horizon in [50,150]:
    fig,axes=plt.subplots((len(example_keys)+1)//2,2,figsize=(14,len(example_keys)),layout='constrained')
    for row_index,key50 in enumerate(example_keys):
        key=key50.replace('_h50_actual',f'_h{horizon}_actual');ax=axes.flat[row_index]
        if key not in sources[0]['examples']:ax.text(.1,.5,'No complete actual window');ax.set_axis_off();continue
        actual=sources[0]['examples'][key];case=key.rsplit('_h',1)[0];route_id=int(case.split('_route')[1].split('_')[0]);ref=reference['xy'][route_id];mask=(ref[:,0]>=actual[:,0].min()-.1)&(ref[:,0]<=actual[:,0].max()+.1);ax.plot(ref[mask,0],ref[mask,1],':',color='gray',label='original reference');baseline_pred=baseline_examples[key.replace('_actual','_prediction')];ax.plot(baseline_pred[:,0],baseline_pred[:,1],'-.' ,color='#009e73',label='M0 same data');ax.plot(actual[:,0],actual[:,1],color='black',lw=2,label='actual')
        for source in sources:
            entry=source['entry'];pred=source['examples'][key.replace('_actual','_prediction')];ax.plot(pred[:,0],pred[:,1],color=colors[entry['kind']],ls=['-','--',':'][entry['seed']],lw=1,label=f"{entry['kind']} seed{entry['seed']}")
        ax.set(title=key.replace('_actual',''),xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        if row_index==0:ax.legend(fontsize=7,ncol=2)
    fig.suptitle(f'Earliest complete{horizon*.02:g}s window per DEV case; all model seeds; free predicted rollout')
    fig.savefig(a.output/f'prediction_xy_{horizon}.png',dpi=130);plt.close(fig)
first=a.previous if a.previous else a.root
baseline=[json.loads(s) for s in (first/'residual_seed0/scalars.jsonl').read_text().splitlines()][0]['dev_mean_max_xy_m']
(a.output/'summary.json').write_text(json.dumps(dict(M0_same_data_initial_mean=baseline,M0_windows=baseline_rows,rows=rows,cases=case_rows,source=str(a.root),previous=str(a.previous)),indent=2)+'\n')
with (a.output/'cases.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(case_rows[0]),lineterminator='\n');w.writeheader();w.writerows(case_rows)
lines=['# P3同数据结构模型与直接模型','', '![全部DEV条件1秒实际/预测XY](prediction_xy_50.png)','', '![全部可用DEV条件3秒实际/预测XY](prediction_xy_150.png)','', '![三初始化训练与开发曲线](training_curves.png)','',f"同数据M0线性基线初始1秒/3秒平均最大XY误差为{baseline['1.0']:.5f}/{baseline['3.0']:.5f}m。M0用相同粗糙TRAIN阶段2转移拟合，区别于此前平地名义模型。M1为M0＋地形/历史修正，M2用相同输入、GRU64/128×128头及姿态积分直接预测动态增量；两个学习模型各三个独立初始化、相同采样和更新预算。",'', '|模型|种子|最佳已评估更新|时域s|窗/例|平均最大XY m|P95 m|最坏m|末25次相对改善|','|---|---|---|---|---|---|---|---|---|']
for r in rows:lines.append(f"|{r['model']}|{r['seed']}|{r['best_update']}|{r['horizon_s']}|{r['windows']}/{r['cases']}|{r['mean_max_xy_m']:.5f}|{r['p95_max_xy_m']:.5f}|{r['worst_xy_m']:.5f}|{r['last25_relative_mean_reduction']:.2%}|")
lines+=['','最佳按固定DEV1秒均值在每25次存档中选择，3秒不单独选模；不是所有更新全程最优。重叠窗口非独立样本，各时域覆盖不同；逐例最坏值保存在cases.csv。损失是监督预测损失，不是PPO，没有策略reward/KL指标。学习模型读取真实381维历史作初始化，未来只用预测状态及自身隐状态查询固定地形，不读未来真实历史。', '', '地形描述是预测位置附近15点的高度、坡度、局部高度变化及支撑mask；沟与图外未标为平地支撑。双线性栅格只是模型输入，最终碰撞几何不替换。此处同一地图、既有TRAIN/DEV初态（来源及速度以本轮manifest为准），尚无独立新地图或规划收益证明。没有用平均误差宣称全程净空合格、训练正式收敛或完整P3已完成。', '',f'原始存档/标量/逐窗口：`{a.root}`；M0、M1、M2均未修改RB/H75或物理。']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rows,indent=2))

for geometry_root,subdir in [(a.geometry,'geometry'),(a.speed_geometry,'speed_geometry')]:
    if geometry_root is None:continue
    out=a.output/subdir;out.mkdir(exist_ok=True);aggregate=[];per_route=[];worst=[]
    lines=['# 整车支持误差与滚动历史复测','','固定600轮候选；M0同数据ARX。静态18个既有DEV条件，滚动2个既有DEV条件。所有预测只使用窗口入口真实历史，之后自由展开；无新物理成功率。','', '|集合/模型|时域s|窗/例|均值XY m|P95 XY m|最差XY m|最差整车XYZ支持m|','|---|---:|---|---:|---:|---:|---:|']
    for folder in sorted(geometry_root.glob('*')):
        if not (folder/'windows.json').exists():continue
        name=folder.name;windows=json.loads((folder/'windows.json').read_text())
        shutil.copyfile(folder/'prediction_xy.png',out/f'{name}_xy.png')
        for r in json.loads((folder/'summary.json').read_text()):per_route.append(dict(panel=name,**r))
        for h in [.1,.3,1.,2.,3.]:
            ws=[r for r in windows if r['horizon_s']==h];xy=np.array([r['max_xy_m'] for r in ws])
            r=dict(panel=name,horizon_s=h,windows=len(ws),cases=len({r['case'] for r in ws}),mean_xy_m=float(xy.mean()),p95_xy_m=float(np.quantile(xy,.95)),max_xy_m=float(xy.max()),max_support_xyz_m=max(r['max_support_xyz_m'] for r in ws));aggregate.append(r)
            worst.append(dict(panel=name,**max(ws,key=lambda r:r['max_support_xyz_m'])))
            lines.append(f"|{name}|{h}|{len(ws)}/{r['cases']}|{r['mean_xy_m']:.5f}|{r['p95_xy_m']:.5f}|{r['max_xy_m']:.5f}|{r['max_support_xyz_m']:.5f}|")
    lines+=['','逐路线完整结果见per_route.csv；worst_windows.csv保留各模型各时域最差整车支持窗口的路线、初态和真实锚点。窗口重叠，不能把窗口数当独立样本。', '', 'M1在静态DEV的1秒整车支持最差误差为0.190–0.252m，3秒为0.357–0.365m。滚动两例1秒为0.025–0.033m，但两例不足以证明覆盖整个滚动域；原覆盖失败仍保留。M2的3秒尾部有种子优于M1，不能宣称M1在所有误差指标占优。当前没有以这些数值自动扩充安全裕量或采用长时域规划。', '', '全覆盖XY：每个模型/条件选择自己的最差1秒XY窗口，窗口锚点可不同，不能视为逐点配对误差图。原始完整轨迹及参考为底图，预测曲线仅覆盖所选有效窗口。']
    for name in sorted({r['panel'] for r in aggregate}):lines+=['',f'![{name} 全路线实际与预测]({name}_xy.png)']
    lines+=['',f'原始逐窗口/模型身份/数组：`{geometry_root}`。']
    if subdir=='speed_geometry':
        lines[0]='# 速度域迁移：冻结1.8模型尚不支持1.9预测'
        lines[2]='冻结600轮候选；M0同数据ARX。三组速度条件各18个既有DEV初态；模型只训练过1.8m/s。本次使用真实phase2速度请求作未来固定命令，不读取未来物理/控制历史，无新增仿真。'
        lines=[line for line in lines if not line.startswith('M1在静态DEV')]
        lines.insert(3,'当phase2请求1.9时，M1三种子1秒平均最大XY约0.086m、3秒约0.29m；返回1.8时为约0.015m/0.03m。所有均值及最差值保留如下。物理144/144成功不能推出旧预测器跨速度可用；下一步需要包含实际速度输入的数据拟合，不能继续只在旧1.8数据堆训练轮数。')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    for name,rs in [('aggregate',aggregate),('per_route',per_route),('worst_windows',worst)]:
        with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
    (out/'summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    with (a.output/'RESULTS.md').open('a') as f:f.write(f'\n[逐路线整车预测诊断：{subdir}]({subdir}/RESULTS.md)。六模型best均在600预算末端，尚未证明收敛；扩大实际数据覆盖后再训练。\n')
