"""Summarize held-out event prediction, separately from physical planning benefit."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
r=json.loads((a.model/'results.json').read_text());meta=json.loads((a.model/'model.json').read_text())
names=['Approach entry','Flight','Far contact','STTW handoff','Goal continuation','Clearance']
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');x=np.arange(6)
for role,color in [('TRAIN','#0072B2'),('DEV','#D55E00')]:
 rows=[r['metrics'][role][str(i)] for i in range(6)]
 for ax,key,label in zip(axes.ravel(),['event_brier','constant_rate_brier','time_mae_s','xy_error_m'],['Event Brier (lower better)','Constant TRAIN-rate Brier','Event time MAE (s; observed events)','Event XY mean error (m; observed events)']):
  y=[float('nan') if row[key] is None else row[key] for row in rows];ax.plot(x,y,'o-',color=color,label=role);ax.set(xticks=x,xticklabels=names,ylabel=label);ax.tick_params(axis='x',rotation=30);ax.legend()
fig.suptitle('Same-map event prediction; grouped starts; no planning benefit established')
fig.savefig(a.output/'prediction_metrics.png',dpi=150);plt.close(fig)
shutil.copy2(a.model/'results.json',a.output/'prediction_metrics.json')
text='# 越野事件模型首轮预测结果\n\n[全部物理实际／参考XY与任务结果](RESULTS.md)\n\n![预测误差与常数基线](prediction_metrics.png)\n\n'
text+=f"按完整来车初态划分TRAIN/DEV，600次CPU更新，DEV选择更新{r['selected_update']}；耗时{r['elapsed_s']:.2f}秒。测试集未参与拟合或选择。六种预定义阶段的事件发生、状态及时间预测，尚非任意接触模式发现；不等同已验证的在线规划收益。\n\n"
text+='|阶段|DEV片段/发生|模型Brier|TRAIN发生率基线Brier|时间MAE/s|XY误差/m|\n|---|---:|---:|---:|---:|---:|\n'
def fmt(v):return '—' if v is None else f'{v:.4f}'
for i,name in enumerate(names):
 row=r['metrics']['DEV'][str(i)];text+=f"|{name}|{row['n']}/{row['events']}|{fmt(row['event_brier'])}|{fmt(row['constant_rate_brier'])}|{fmt(row['time_mae_s'])}|{fmt(row['xy_error_m'])}|\n"
text+='\n时间和状态误差仅对确实发生的事件计算；Brier包括未发生事件。前缀片段相关，不能把片段数当独立初态数。常数基线只使用TRAIN各阶段发生率。352维控制历史与观测压缩成16维，不能据此保证Markov性；没有物理接触求解缓存。\n\n'
text+=f"数据：`{meta['collection']}`；模型：`{a.model}`。完整指标见[prediction_metrics.json](prediction_metrics.json)。后续仍需同输入直接成败/普通转移模型对照及独立整程规划验证。\n"
(a.output/'PREDICTION.md').write_text(text);print(text)
