"""Compare completed geometry conditions, preserving cases and terminal meanings."""
import argparse,json,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[];detail=[]
for run in a.runs:
 s=json.loads((run/'evaluation/summary.json').read_text());p=json.loads((run/'protocol.json').read_text());contract=json.loads((run/'map_contract.json').read_text());d=np.load(run/'evaluation/trajectories.npz');per=len(contract['speeds_m_s'])*len(contract['alphas'])
 for case in p['cases']:
  lane=case['lane'];result=s['cases'][lane];valid=d['valid'][:,lane];root=d['root_xyzw'][valid,lane];phase=d['phase'][valid,lane]
  detail.append(dict(run=run.name,route=contract['routes'][case['candidate_id']//per],root=case['root'],candidate=case['candidate'],success=result['success'],end_code=result['end_code'],entered_jump=result['entry_s'] is not None,handed_back=result['handoff_s'] is not None,last_phase=int(phase[-1]),last_time_s=float(d['time'][valid][-1]),last_x_m=float(root[-1,0]),max_x_m=float(root[:,0].max()),max_abs_y_m=float(abs(root[:,1]).max()),road_violation=result['road_violation']))
 for route in contract['routes']:
  cases=[c for c in detail if c['run']==run.name and c['route']==route]
  rows.append(dict(run=run.name,route=route,n=len(cases),success=sum(c['success'] for c in cases),entered_jump=sum(c['entered_jump'] for c in cases),handed_back=sum(c['handed_back'] for c in cases),passed_approach_ridge=sum(c['max_x_m']>=5.5 for c in cases),reached_far_ridge=sum(c['max_x_m']>=13 for c in cases),end_codes=dict(collections.Counter(c['end_code'] for c in cases))))
(a.output/'geometry_comparison.json').write_text(json.dumps(dict(groups=rows,cases=detail),indent=2))
text='# 越野弯道几何条件与阶段结果\n\n两批分别为120条探索DEV，共用种子、来车初态与地形；更缓条件仅改变S弯与90°圆角几何。不同批次重复初态不计作独立样本；不合并成成功率显著性证据。\n\n'
for run in a.runs:text+=f'- [{run.name}全部实际/参考XY与逐候选结果](../{run.name}/RESULTS.md)\n'
text+='\n|条件|路线|完成|进入跳跃|交回STTW|曾到达x≥5.5|曾到达x≥13|终止码分布|\n|---|---|---:|---:|---:|---:|---:|---|\n'
for r in rows:text+=f"|{r['run']}|{r['route']}|{r['success']}/{r['n']}|{r['entered_jump']}|{r['handed_back']}|{r['passed_approach_ridge']}|{r['reached_far_ridge']}|{r['end_codes']}|\n"
text+='\n进入跳跃是控制阶段切换，不等于真实离地或成功清障；交回STTW由首次远岸接触触发，不等于整程完成。地面绕行不要求进入跳跃。x阈值只记录实际到达位置，不能单独证明跨越脊坡稳定。终止码15为运行域退出，3/4为侧倾/俯仰阈值，18为到目标但曾越廊，9为观察窗口结束未成功，13为完成。\n'
(a.output/'GEOMETRY_COMPARISON.md').write_text(text);print(text)
