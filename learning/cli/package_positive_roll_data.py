"""Export only selected figure data; never copy the multi-GB contact archive."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--redraw',type=Path,required=True);args=p.parse_args();run=args.run;out=run/'roll_contact_analysis';dest=out/'positive_figure_data';(dest/'data').mkdir(parents=True,exist_ok=True);(dest/'figures').mkdir(exist_ok=True);(dest/'source').mkdir(exist_ok=True)
selections=json.loads((out/'positive_selection.json').read_text());outcomes=list(csv.DictReader((out/'positive_cases.csv').open()));events=list(csv.DictReader((out/'events.csv').open()));derived=np.load(out/'positive_derived.npz');meta={'scope':'two positive examples and six matched control lanes;8 lanes total','cases':[]}
for i,s in enumerate(selections):
 name=s['case'];lanes=s['lanes'];c=json.loads((run/name/'contract.json').read_text());d={}
 with np.load(out/(name+'.npz')) as z:
  for key in z.files:
   a=z[key]
   if key=='labels':a=a[lanes]
   elif a.ndim>=2 and a.shape[1]==18:a=a[:,lanes]
   d[key]=a
 for key in ['z','unstable','restoring','nominal_demand','goal_distance']:d[key]=derived[name+'__'+key][:,lanes]
 route=np.load(Path(c['bundle'])/'route.npz');goal=float(route['goal']);d['reference_xy']=route['xy'];d['reference_s']=route['s'];d['reference_curvature']=route['curvature'];d['goal_point']=np.array([np.interp(goal,route['s'],route['xy'][:,j]) for j in (0,1)])
 np.savez_compressed(dest/'data'/(name+'.npz'),**d)
 fields=['time_s','label','source_lane','phi_rad','euler_roll_rate_rad_s','speed_m_s','delta_rad','delta_dot_rad_s','path_error_m','goal_distance_m','x_m','y_m','z_rad_s','nominal_required_rate_rad_s','rate_command_rad_s','pred_nominal100_rad','pred_contact100_rad']+d['contact_fields'].tolist()
 with (dest/'data'/(name+'.csv')).open('w') as f:
  writer=csv.writer(f);writer.writerow(fields)
  for lane,label in enumerate(d['labels']):
   for t in np.flatnonzero(d['valid'][:,lane]):
    row=[d['time'][t,lane],str(label),lanes[lane],*[d[k][t,lane] for k in ['phi','euler_roll_rate','speed','delta','delta_dot','path_error','goal_distance']],*d['xy'][t,lane],*[d[k][t,lane] for k in ['z','nominal_demand','command','pred_nominal100','pred_contact100']],*d['contact_mean'][t,lane]]
    writer.writerow([format(float(v),'.9g') if isinstance(v,(float,np.floating)) else v for v in row])
 matched=[next(x for x in outcomes if x['case']==name and x['label']==label) for label in d['labels']]
 meta['cases'].append(dict(file=name+'.npz',title=s['title'],positive=s['positive'],source_case=str(run/name),original_lanes=lanes,local_lanes=list(range(4)),selection_reason=s['reason'],path_task_bound_m=.4 if i==0 else .3,outcomes=matched,events=[e for e in events if e['case']==name and e['label'] in d['labels']]))
 shutil.copy2(run/name/'contract.json',dest/'source'/(name+'_contract.json'))
meta['units']={'angles':'rad in data; plotted in degrees where labelled','forces':'N','moments':'Nm','positions':'m','time':'s','contact_fields':'5ms mean/min/max from native0.2ms records; contact_fields gives exact order'}
(dest/'metadata.json').write_text(json.dumps(meta,indent=2))
for path in out.glob('POSITIVE_*.png'):shutil.copy2(path,dest/'figures'/path.name)
for name in ['positive_cases.csv','positive_selection.json','contact_correction_models.json','validation.json']:shutil.copy2(out/name,dest/name)
shutil.copy2(args.redraw,dest/'redraw.py');shutil.copy2(Path(__file__).with_name('analyze_roll_contacts.py'),dest/'source/analyze_roll_contacts.py');shutil.copy2('runs/isaac_signals_parity/dynamic.json',dest/'source/controller_dynamic.json')
readme='''# 两组正面例子：图中数据与重绘

本包对应POSITIVE_XY、POSITIVE_STATES、POSITIVE_CONTACT_TASK、POSITIVE_RECOVERY_DEMAND、POSITIVE_MODEL_CHECK五张图。只含两工况、8条所画轨迹，不含5.71GB原始接触。

1. 单脊10cm、2.6m/s、seed101：full_0.5为正例，通过任务C；对照full_1、speed_only_0.5、steer_only_0.5。
2. 石子10mm、2.3m/s、seed101：full_0.75为正例，通过A/B/C；对照speed_only_0.75、steer_only_0.75、Zero。仅转向也通过A/B/C，完整保留。

## 使用
- data/*.csv：直接用于Excel/绘图；仅含真实有效步。空/NaN接触或预测表示未记录/无完整预测窗口，不是零。
- data/*.npz：全部图中数组、有效掩码、接触均值/最小/最大、参考XY、目标锚点、模型100/500ms结果。维度通常[time,local_lane,...]；local_lane0为正例。原车道映射见metadata.json。
- figures/：已生成的5张PNG。
- positive_cases.csv：8条轨迹的任务结果、真实终点、目标截面穿越时刻。
- metadata.json：事件、图例、单位、原路径、选例依据。
- source/：原合同、控制参数和计算源代码；原始高频数据路径在合同中。
- contact_correction_models.json：接触修正模型系数、训练物理组与留出组。

运行 `python redraw.py`（numpy、matplotlib）即可仅依赖本包数据重画五图到redrawn/；布局可能与figures略有差异，数值相同。仅需看数据无需运行代码。

## 时间、力矩和模型含义
状态每5ms后步记录；time保留真正失败末步时刻，valid=false之后不可画成存活。接触来自5kHz记录的5ms统计，保留min/max，不是重新采样仿真。高频原始接触窗口0.8–10s，事件在5ms精度上识别。
接触事件定义：至少一个轮地法向力>1N且接触点世界z>2mm，相隔<=50ms合并。图中灰带只表示正例接触；对照自身事件见metadata。绿色线为事件后首次连续200ms满足|phi|<=.18rad、|路径误差|<=.3m、|速度误差|<=.2m/s的入带时刻；若事件中未离带，不解释为新恢复。蓝带9–10s是原最终保持窗口。
轮接触侧倾力矩是关于整车质心、沿负车架x轴的投影；它不等于已识别的地形扰动d。motor扭矩是执行器施加项，不含被动弹簧阻尼。转角范围±.8rad；±3rad/s是伺服命令范围，不是受冲击时实际角速度硬上限；电机扭矩限幅±1Nm。
目标是穿越固定目标截面并在末段保持原路径/航向/速度，不是停在目标点或进入新设1m圆。距离曲线在穿越后上升可能是沿原路线继续前进；不得据此改判任务。

z=Euler侧倾导数+sqrt(a4)*phi；nominal_required_rate令名义dot(z)=0且d=0。它用于看恢复需求，不能当严格不可恢复边界。Euler侧倾导数由四元数与世界角速度变换，不直接拿IMU x分量代替。
动态曲线是不断以实测状态起算的100ms实测输入回放，不是10s自由预测，也不是只给alpha即可预测。接触修正用其他两个物理seed拟合4个接触力矩特征，留出当前seed验证，不使用当前回合反算残差来拟合。500ms误差明显增大；本包保留对应数组。标定过程见source，重拟合需原始运行数据；redraw只重画保存结果。

原始数据、失败与重复未修改。NPZ是主要精度数据，CSV输出9位有效数字以减小体积。SHA256SUMS用于文件完整性验证。
'''
(dest/'README.md').write_text(readme)
manifest=[]
for file in sorted(dest.rglob('*')):
 if file.is_file() and file.name!='SHA256SUMS':manifest.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+file.relative_to(dest).as_posix())
(dest/'SHA256SUMS').write_text('\n'.join(manifest)+'\n')
archive=out/'positive_figure_data.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for file in sorted(dest.rglob('*')):
  if file.is_file():z.write(file,Path('positive_figure_data')/file.relative_to(dest))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps(dict(zip=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),crc='passed'),indent=2))
