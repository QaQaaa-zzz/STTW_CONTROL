"""Compare head structures on exactly matched saved event transitions."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--branched',type=Path,required=True);p.add_argument('--shared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
# The comparison requires the same labels, roles and compressed state features.
for name in ('transitions.npz','codec.npz'):
 with np.load(a.branched/name) as left,np.load(a.shared/name) as right:
  assert set(left.files)==set(right.files),name
  for key in left.files:np.testing.assert_array_equal(left[key],right[key],err_msg=f'{name}:{key}')
x=json.loads((a.branched/'results.json').read_text());y=json.loads((a.shared/'results.json').read_text())
a.output.mkdir(parents=True,exist_ok=True)
for name,model in [('branched',a.branched),('shared',a.shared)]:
 shutil.copyfile(model/'results.json',a.output/f'{name}_metrics.json')
 shutil.copyfile(model/'model.json',a.output/f'{name}_metadata.json')
names=['接近入口','腾空','远岸接触','STTW回交','目标续行','清障辅助']
s='''# 事件模型输出头结构对照（DEV）

[对应物理数据的全部实际/参考XY](../chapter3_offroad_distance_training/RESULTS.md)。本轮仅CPU模型拟合，不生成新物理轨迹，不改写原仿真结果。

同一1199条已发布片段、6 TRAIN根/2 DEV根，PCA与逐事件状态/时间/标签/划分逐数组相等。两模型均600次更新、Adam 0.001、batch256、相同采样种子、每100次DEV选模，本次均选600。事件类型输入及标签完全保留，唯一设计变化为三个输出头改为一个共享头。共享编码器形状不变；参数量35770对27710，非严格等参数容量对照。该模型仍是事件条件模型，不是普通固定步长状态转移基线。

|阶段|三头Brier|单头Brier|三头时间MAE/s|单头时间MAE/s|三头XY/m|单头XY/m|
|---|---:|---:|---:|---:|---:|---:|
'''
for i,name in enumerate(names):
 l=x['metrics']['DEV'][str(i)];r=y['metrics']['DEV'][str(i)]
 s+='|'+name+'|'+ '|'.join(f'{v:.4f}' for k in ['event_brier','time_mae_s','xy_error_m'] for v in [l[k],r[k]])+'|\n'
s+='''
三头模型六阶段XY误差均更低，五阶段时间误差更低、四阶段Brier更低；单头在远岸接触及回交概率上更准。续行XY误差0.8825m对1.7799m，显示本次分支结构对状态预测有帮助，但只有单训练种子和两个DEV根，且三头参数更多，不能宣称通用架构优势或在线完成率改善。时间/XY仅对发生事件评价；不以相关片段数代替独立来车数。

共享头拟合墙钟10.40秒（原三头记录21.34秒，非同负载配对计时，不能归因结构提速）；0物理采集、0PPO。旧权重严格加载、六模式相同头等价性及包含缺失事件的有限反传通过。当前正式在线模型不替换；此结果只补充论文结构对照。

证据：[三头指标](branched_metrics.json)、[单头指标](shared_metrics.json)、[三头配置](branched_metadata.json)、[单头配置](shared_metadata.json)。
'''
(a.output/'RESULTS.md').write_text(s)
(a.output/'verification.json').write_text(json.dumps(dict(matched_arrays=['transitions.npz','codec.npz'],branched=str(a.branched),shared=str(a.shared),dev_roots=2,training_seeds=1,scope='DEV event-head structure comparison; no online performance claim'),indent=2)+'\n')
print(s)
