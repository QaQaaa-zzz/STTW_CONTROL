"""Report all route, speed and alpha cases on the expanded physical map."""
import argparse,json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
protocol=json.loads((a.run/'protocol.json').read_text());contract=json.loads((a.run/'map_contract.json').read_text());summary=json.loads((a.run/'evaluation/summary.json').read_text());d=np.load(a.run/'evaluation/trajectories.npz');routes=np.load(a.run/'roads.npz')['xy'];cases=protocol['cases'];speeds=contract['speeds_m_s'];alphas=contract['alphas'];names=contract['routes'];per=len(speeds)*len(alphas)
points=d['root_xyzw'][d['valid']][:,:2];xlimits=(min(0.,float(points[:,0].min())-.2),max(19.,float(points[:,0].max())+.2));ylimits=(min(-4.,float(points[:,1].min())-.2),max(4.,float(points[:,1].max())+.2))
for kind in ('xy','reward','cumulative_reward','speed'):
 fig,axs=plt.subplots(len(names),len(speeds),figsize=(20,12),layout='constrained')
 for case in cases:
  lane=case['lane'];candidate=case['candidate_id'];route=candidate//per;speed=candidate%per//len(alphas);alpha=candidate%len(alphas);ax=axs[route,speed];ok=d['valid'][:,lane];root=d['root_xyzw'][ok,lane];rew=d['reward'][ok,lane];t=d['time'][ok];color=['#0072B2','#D55E00','#009E73'][alpha];style=['-','--'][case['root']]
  if kind=='xy':
   ax.plot(root[:,0],root[:,1],color=color,ls=style,lw=1);ax.scatter(*root[-1,:2],marker='o' if summary['cases'][lane]['success'] else 'x',color=color,s=18)
   if case['root']==0 and alpha==0:ax.plot(routes[candidate,:,0],routes[candidate,:,1],'k:',lw=.9);ax.axvspan(10.3,10.6,color='gray',alpha=.15)
   ax.set(xlim=xlimits,ylim=ylimits,xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal',adjustable='box')
  else:
   value=rew if kind=='reward' else (rew.cumsum() if kind=='cumulative_reward' else root[:,7]);ax.plot(t,value,color=color,ls=style,lw=.8);ax.set(xlabel='time (s)',ylabel={'reward':'reward / step','cumulative_reward':'cumulative reward','speed':'world vx (m/s)'}[kind])
  ax.set_title(f'{names[route]} / {speeds[speed]} m/s',fontsize=9)
 fig.suptitle('Exploratory DEV: alpha 0 blue / 0.5 orange / 1 green; root 0 solid / 1 dashed; actual failure endpoints retained')
 fig.savefig(a.output/f'{kind}.png',dpi=130);plt.close(fig)
rows=[]
for case in cases:
 result=summary['cases'][case['lane']];rows.append(dict(**case,result=result))
(a.output/'cases.json').write_text(json.dumps(rows,indent=2))
text='# 越野任务地图与首次全候选执行\n\n![全部实际/参考XY](xy.png)\n\n![物理高程与路线](map_overview.png)\n\n四条路线共享同地图、同目标：直线跨沟、S弯接近跨沟、圆角90度弯接近跨沟、连续地面绕行。五速度1.5/1.8/2.0/2.3/2.6m/s来自STTW既有配置；α0/.5/1。共60候选×2初态=120条探索性DEV，不是独立泛化统计。候选在x1.8m实际发布，旧世界模型仅加载共用入口，不为新地图评分。原道路半宽0.6m、目标和失败判据保留；跳跃模型身份见下文冻结配置，首次远岸接触后交回STTW。\n\n地形：真实30cm开口沟；沟外1–4mm圆顶石子（碰撞高度网格而非视觉装饰），2cm及8cm脊坡；x8.5–12m平整跳跃区。同高两岸，无整体高岸。碰撞网格水平分辨率约2.5cm，高程图5cm、单位米，沟缺口保留NaN及support=false；图显示厘米。圆角90度路径半径见本运行map_contract.json，不要求车辆瞬间转向。\n\n|路线|速度m/s|α0 /2|α0.5 /2|α1 /2|\n|---|---:|---:|---:|---:|\n'
for i,name in enumerate(names):
 for j,speed in enumerate(speeds):
  totals=[sum(c['result']['success'] for c in rows if c['candidate_id']==i*per+j*len(alphas)+k) for k in range(len(alphas))];text+=f'|{name}|{speed}|'+ '|'.join(map(str,totals))+'|\n'
text+=f"\n完整任务完成 {sum(c['result']['success'] for c in rows)}/{len(rows)}。耗时 {summary['elapsed_s']:.2f}s，物理环境步 {summary['physics_lane_steps']}。未参与训练，不按失败删除路线或放松道路。此轮同时扩展路线与地形，用于任务覆盖与能力采集，不归因某单因素。\n\n[每步奖励](reward.png) · [累计奖励](cumulative_reward.png) · [真实速度](speed.png) · [全部原始终止码与逐条件结果](cases.json)。\n\n原始高程、支持掩码、碰撞网格、路线与逐步数据：`{a.run}`。后续世界模型需要使用本地图数据重训/校准，当前不能宣称已学会在新地图自主决策。\n"
cfg=json.loads((a.run/'config.json').read_text());text+='\n实际跳跃权重：`'+cfg.get('evaluation_checkpoint',cfg['bundle'])+'`。\n'
(a.output/'RESULTS.md').write_text(text)
print(text)
