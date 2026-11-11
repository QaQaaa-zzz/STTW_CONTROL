"""Exact lower-envelope intervals for the saved complete endpoint costs."""
import argparse,json
from pathlib import Path
import numpy as np

def preference_intervals(f0,f1):
    f0=np.asarray(f0);slope=np.asarray(f1)-f0;result=[]
    for i in range(len(f0)):
        lo,hi=0.,1.
        for j in range(len(f0)):
            m=slope[i]-slope[j];rhs=f0[j]-f0[i]
            if m>0:hi=min(hi,rhs/m)
            elif m<0:lo=max(lo,rhs/m)
            elif rhs<0:hi=-1.
        result.append(None if hi<lo else [float(lo),float(hi)])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=json.loads(a.source.read_text());rows=[];alphas=[0,.25,.5,.75,1]
    for panel in data['panels']:
        bounds=preference_intervals(panel['F0'],panel['F1']);members=[]
        for alpha,interval in zip(alphas,bounds):members.append(interval is not None and interval[0]-1e-10<=alpha<=interval[1]+1e-10)
        for i,interval in enumerate(bounds):
            if interval is None:continue
            theta=np.mean(interval);scores=(1-theta)*np.array(panel['F0'])+theta*np.array(panel['F1']);assert scores[i]<=scores.min()+1e-8
        rows.append(dict(terrain=panel['terrain'],speed_m_s=panel['speed_m_s'],theta_intervals=bounds,own_alpha_in_interval=members,F0=panel['F0'],F1=panel['F1']))
    result=dict(definition='theta weights the complete discounted endpoints: (1-theta)F0+theta F1, including shared costs, gates, heading and terminal replacement; not the physical-RMSE eta',source=str(a.source),panels=rows,own_alpha_optimal_count=sum(sum(r['own_alpha_in_interval']) for r in rows),total_policy_panel_pairs=100)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'complete_intervals.json').write_text(json.dumps(result,indent=2))
    lines=['# S1：完整端点代价的偏好区间','','[全部配对XY](xy_all.png) · [完整代价重评分](OBJECTIVES.md) · [eta来源](ETA_ORIGIN.md)','','对全部五档同时求解F0_i+θ(F1_i−F0_i)≤F0_j+θ(F1_j−F0_j)，θ∈[0,1]。θ是评价偏好，α是被评策略输入；两者分别列出。保留姿态、动作、χ门控、航向及终止项。','','|地形|请求速度|α0|α.25|α.5|α.75|α1|','|---|---:|---|---|---|---|---|']
    for row in rows:
        cells=['空集' if b is None else f'{b[0]:.4f}–{b[1]:.4f}' for b in row['theta_intervals']];lines.append('|'+row['terrain']+'|'+str(row['speed_m_s'])+'|'+'|'.join(cells)+'|')
    lines+=['',f'100个策略—面板组合中，{result["own_alpha_optimal_count"]}个在θ=自身α时是五候选中的最优。区间为空表示该候选在本面板整个θ范围都不占优，原始轨迹仍保留。','', 'η用于Ev²+η²Ey²，其量纲是s⁻¹；本表θ用于完整训练端点，不能将两张表当作同一个权重。只有额外满足固定门控、相同共同代价、相同代价形状等条件，才可将训练速度/路径权重比换成物理误差η。当前共享策略的精确最优前提未满足，因此S1用区间检验实际族内选择，不推出α=θ的普适结论。']
    (a.output/'COMPLETE_INTERVALS.md').write_text('\n'.join(lines)+'\n');print(result['own_alpha_optimal_count'],'/100 own-alpha candidates are optimal')
if __name__=='__main__':main()
