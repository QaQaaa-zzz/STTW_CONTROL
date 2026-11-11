"""Recompute all archived paired panels and their exact quadratic selection intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def intervals(v,p):
    result=[]
    for i in range(len(v)):
        lo,hi=0.,np.inf
        for j in range(len(v)):
            slope=p[i]-p[j]; rhs=v[j]-v[i]
            if slope>0: hi=min(hi,rhs/slope)
            elif slope<0: lo=max(lo,rhs/slope)
            elif rhs<0: hi=-1.
        result.append(None if hi<lo else [float(np.sqrt(lo)),None if np.isinf(hi) else float(np.sqrt(hi))])
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    original=json.loads(args.source.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    alpha=np.array([0,.25,.5,.75,1]);colors=["#0072B2","#56B4E9","#009E73","#E69F00","#D55E00","#777777"]
    fig,axs=plt.subplots(4,5,figsize=(19,14),layout="constrained")
    xyfig,xyaxs=plt.subplots(4,5,figsize=(19,15),layout="constrained")
    rows=[];raw_error=0.
    for panel,(case,ax,xax) in enumerate(zip(original["cases"],axs.flat,xyaxs.flat)):
        speed=case["speed_m_s"];count=round(case["common_window_s"]/.005);vs=[];ps=[]
        for seed in case["per_seed"]:
            run=Path(seed["run"])
            with np.load(run/"trajectory.npz") as raw:
                tr=raw["lower_trace"].reshape(-1,6,11)
                v=np.mean((tr[:count,:5,2].astype(float)-speed)**2,axis=0);p=np.mean(tr[:count,:5,3].astype(float)**2,axis=0)
                vs.append(v);ps.append(p)
                raw_error=max(raw_error,float(np.max(abs(np.sqrt(v)-np.array(seed["speed_rmse"])))),float(np.max(abs(np.sqrt(p)-np.array(seed["path_rmse"])))))
                for lane in range(6):
                    valid=tr[:,lane,6].astype(bool);curve=tr[valid,lane,:2]
                    xax.plot(*curve.T,color=colors[lane],alpha=.30,lw=.6)
                    if valid.sum()<len(valid):xax.scatter(*curve[-1],marker="x",s=12,color=colors[lane])
            ax.scatter(np.sqrt(p),np.sqrt(v),c=colors[:5],s=12,alpha=.3)
        contract=json.loads((run/"contract.json").read_text())
        with np.load(Path(contract["bundle"])/"route.npz") as route:xax.plot(*route["xy"].T,"k:",lw=.9)
        v=np.mean(vs,axis=0);p=np.mean(ps,axis=0);ranges=intervals(v,p)
        # Candidate intervals verified independently at an interior point.
        for i,bounds in enumerate(ranges):
            if bounds is None:continue
            lo,hi=bounds;eta=(lo+hi)/2 if hi is not None else lo+max(1.,lo)
            assert (v+eta**2*p)[i] <= np.min(v+eta**2*p)+1e-10
        joint=(np.diff(v)<=0)&(np.diff(p)>=0)
        ax.plot(np.sqrt(p),np.sqrt(v),color=".5",lw=1)
        for i in range(5):
            ax.scatter(np.sqrt(p[i]),np.sqrt(v[i]),color=colors[i],s=32)
            ax.annotate(str(alpha[i]),(np.sqrt(p[i]),np.sqrt(v[i])),fontsize=8)
        title=f'{case["terrain"]} / {speed:g} m/s'
        ax.set(title=title,xlabel="Path RMS (m)",ylabel="Speed RMS (m/s)")
        xax.set(title=title,xlabel="World x (m)",ylabel="World y (m)");xax.set_aspect("equal");xax.grid(alpha=.15)
        rows.append(dict(terrain=case["terrain"],speed_m_s=speed,common_window_s=case["common_window_s"],seeds=[s["seed"] for s in case["per_seed"]],speed_mse=v.tolist(),path_mse=p.tolist(),optimal_eta_intervals=ranges,adjacent_tradeoff=joint.tolist(),all_seed_feasible_alpha=case["full_window_feasible_alpha"],source_runs=[s["run"] for s in case["per_seed"]]))
    fig.suptitle("Five-alpha preference frontiers: all 20 archived panels\nFaint dots: five paired physical groups; large points: root mean squared errors pooled across groups")
    xyfig.suptitle("All paired actual/reference XY trajectories | five alphas + Zero\nBlue to red: alpha 0 / .25 / .5 / .75 / 1; gray: Zero; dotted: original path; crosses: early endpoints")
    fig.savefig(out/"frontiers.png",dpi=170);xyfig.savefig(out/"xy_all.png",dpi=170);plt.close("all")
    joint=sum(sum(r["adjacent_tradeoff"]) for r in rows)
    summary=dict(panels=len(rows),runs=sum(len(r["seeds"]) for r in rows),alpha=[float(a) for a in alpha],raw_archive_rmse_max_difference=raw_error,adjacent_tradeoff_edges=joint,total_edges=4*len(rows),panels_with_all_four_tradeoff_edges=sum(all(r["adjacent_tradeoff"]) for r in rows),source=str(args.source),new_simulations=0,panels_data=rows)
    (out/"frontiers.json").write_text(json.dumps(summary,indent=2))
    text=["# S1：全部20面板的偏好前沿","","![实际与参考XY](xy_all.png)","","![五档前沿](frontiers.png)","",f"从100份原始轨迹重算20个面板，五档α与Zero共同有效窗口；与旧档案RMSE最大差{raw_error:.3g}。{joint}/80相邻档同时表现为速度误差不增、路径误差不减；{summary['panels_with_all_four_tradeoff_edges']}/20面板的四条相邻边全部满足该方向。","", "## 各档可占优的任务尺度","","下表对所有其他候选同时求解 Vᵢ+η²Pᵢ≤Vⱼ+η²Pⱼ，不只比较相邻档。η单位s⁻¹；空集表示该档不在该面板的加权平方误差下包络上。窗口为原协议的共同观察窗口，任务可行性另外保留，不能靠短观察窗口得到任务成功。","","|地形|速度|α=0|α=.25|α=.5|α=.75|α=1|","|---|---:|---|---|---|---|---|"]
    for r in rows:
        cells=["空集" if b is None else f"{b[0]:.4g}–"+("∞" if b[1] is None else f"{b[1]:.4g}") for b in r["optimal_eta_intervals"]]
        text.append("|"+str(r["terrain"])+"|"+str(r["speed_m_s"])+"|"+"|".join(cells)+"|")
    text+=["","## 当前判断与后续","","五档形成条件相关的离散前沿，应按具体任务尺度选档。物理误差前沿不等同包含姿态、修正与终止代价的训练目标；下一步重建两个完整端点代价，检验标量化最优性关系。η=.533出现在旧分析脚本默认参数和两个校准面板说明中，尚无证据表明它是事先独立确定的任务参数；本轮不依赖该数值，输出全部η区间。","", "当前只完成S1的原始误差/候选包络部分；精确奖励、弯道分配、S2接触映射、S3预测冻结和细扫描、S4独立测试继续按合同推进。"]
    (out/"README.md").write_text("\n".join(text)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="panels_data"},indent=2))
if __name__=="__main__":main()
