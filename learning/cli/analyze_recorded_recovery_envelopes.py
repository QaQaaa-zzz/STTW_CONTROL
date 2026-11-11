"""Apply the verified nominal steering envelope to all archived full-policy lanes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from verify_nominal_recovery import classify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_ridge_summary(source, output, rows):
    labels=['full_0','full_0.25','full_0.5','full_0.75','full_1']
    colors=['#D55E00','#CC79A7','#009E73','#0072B2','#B22222']
    fig,axs=plt.subplots(3,2,figsize=(12,10),sharex=True,sharey='col',layout='constrained')
    for row,seed in enumerate([101,102,103]):
        name=f'single10_v2.6_seed{seed}';z=np.load(source/(name+'.npz'))
        for lab,color,style in zip(labels,colors,['--',':','-','-.','--']):
            lane=list(z['labels']).index(lab);mask=z['valid'][:,lane]&(z['time'][:,lane]>=1.1)&(z['time'][:,lane]<=3.3);t=z['time'][mask,lane]
            for col,key in enumerate(['phi','euler_roll_rate']):
                axs[row,col].plot(t,np.degrees(z[key][mask,lane]),color=color,ls=style,lw=1.7,label=lab.replace('full_','alpha='))
            bad=[r for r in rows if r['case']==name and r['label']==lab and r['outcome']=='unavoidable_nominal_limit']
            if bad:
                first=bad[0]
                for col,key in enumerate(['phi_deg','omega_deg_s']):axs[row,col].scatter(float(first['time_s']),float(first[key]),marker='s',s=27,color=color,zorder=5)
        axs[row,0].set_ylabel(f'Group {seed} | roll (deg)');axs[row,1].set_ylabel('Roll rate (deg/s)')
        for ax in axs[row]:ax.axhline(0,color='.6',lw=.7);ax.grid(alpha=.15);ax.set_xlim(1.1,3.3)
        for bound in [-np.degrees(.7),np.degrees(.7)]:axs[row,0].axhline(bound,color='.4',ls=':',lw=.7)
    axs[0,0].legend(ncol=3,fontsize=9);axs[0,0].set_title('Actual roll: intermediate preferences recover');axs[0,1].set_title('Roll rate: accumulation and recovery')
    for ax in axs[-1]:ax.set_xlabel('Time (s)')
    fig.suptitle('10 cm ridge | requested 2.6 m/s | all five preferences, three physical groups\nSquares: first sampled state excluded by optimal nominal steering envelope; curves end at true termination or 3.3 s',fontsize=12)
    fig.savefig(output/'ridge_mechanism.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.controller.read_text());rows=[];summary=[]
    labels=['full_0','full_0.25','full_0.5','full_0.75','full_1','Zero']
    colors=['#D55E00','#CC79A7','#009E73','#0072B2','#B22222','#555555']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})
    for path in sorted(a.source.glob('*_seed*.npz')):
        z=np.load(path);raw=np.load(a.source.parent/path.stem/'trajectory.npz');fig,axs=plt.subplots(3,2,figsize=(14,10),layout='constrained')
        for n,label in enumerate(labels):
            lane=list(z['labels']).index(label);valid=np.flatnonzero(z['valid'][:,lane]);ids=valid[::4]
            selected=ids[(z['time'][ids,lane]>=1.1)&(abs(z['phi'][ids,lane])<.7)]
            rr=[]
            for i in selected:
                v=float(z['speed'][i,lane]);x=[float(z[k][i,lane]) for k in ['phi','euler_roll_rate','delta']]
                # Analytic kernel sign is established only for forward speed >=0.5 m/s.
                if v<.5 or abs(x[2])>.8:continue
                r=dict(case=path.stem,label=label,time_s=float(z['time'][i,lane]),speed_m_s=v,phi_deg=float(np.degrees(x[0])),omega_deg_s=float(np.degrees(x[1])),delta_rad=x[2]);r.update(classify(v,x,cfg));rr.append(r)
            rows.extend(rr);bad=[r for r in rr if r['outcome']=='unavoidable_nominal_limit'];last=valid[-1]
            summary.append(dict(case=path.stem,label=label,end_s=float(z['time'][last,lane]),end_phi_deg=float(np.degrees(z['phi'][last,lane])),domain_exit=bool(raw['domain_exit'][:last+1,lane].any()),roll_limit_observed=bool((abs(z['phi'][valid,lane])>=.7).any()),tested_states=len(rr),unavoidable_states=len(bad),first_nominal_exclusion_s=bad[0]['time_s'] if bad else None,first_predicted_crossing_s=bad[0]['time_s']+bad[0]['event_s'] if bad else None))
            ax=axs.flat[n];t=z['time'][valid,lane];w=np.degrees(z['euler_roll_rate'][valid,lane]);ax.plot(t,w,color=colors[n],lw=1.6,label='Recorded Euler roll rate')
            if bad:ax.scatter([r['time_s'] for r in bad],[r['omega_deg_s'] for r in bad],s=12,color='black',marker='x',label='Nominal best steering still crosses')
            ax.axhline(0,color='.6',lw=.7);ax.set_title(label);ax.set_xlabel('Time (s)');ax.set_ylabel('Roll rate (deg/s)');ax.legend(fontsize=8);ax.set_xlim(1.1,10)
        limits=[ax.get_ylim() for ax in axs.flat]
        for ax in axs.flat:ax.set_ylim(min(x[0] for x in limits),max(x[1] for x in limits))
        fig.suptitle(path.stem+' | five preferences + Zero, complete valid observations\nBlack crosses: frozen-speed, no-future-contact envelope; not full-physics impossibility')
        fig.savefig(a.output/(path.stem+'.png'),dpi=150);plt.close(fig)
        print(path.stem,len(rows),flush=True)
    plot_ridge_summary(a.source,a.output,rows)
    for name,data in [('states.csv',rows),('summary.csv',summary)]:
        with (a.output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (a.output/'definition.json').write_text(json.dumps(dict(source=str(a.source),controller=str(a.controller),sample_interval_s=.02,start_s=1.1,horizon_s=1,model_time_grid_s=.00025,steering_rate_rad_s=3,steering_angle_rad=.8,roll_limit_rad=.7,lanes=len(summary),states=len(rows),note='Current-direction braking only; fixed measured speed and zero future external disturbance. Full valid horizon retained; states outside v>=.5, abs(delta)<=.8, abs(phi)<.7 omitted from nominal classification only.'),indent=2))
    lines=['# S2：完整记录中的最有利名义恢复包络','','[原始全部XY]('+str(a.source/'INDEX.md')+')。五档完整网络及Zero，共72条；每20ms从实际状态计算一次，不调整任何参数。','','每个时刻保持当时速度、去掉未来接触扰动，并给出最有利的转向（速度上限3rad/s，转角上限0.8rad）。计算未来1秒：若当前方向尚未刹住就越过0.7rad，标记名义不可避免越限。黑叉不等于完整PhysX必败；刹住当前方向也不等于完成任务。','','## 单脊2.6m/s，全物理组和五档','','|组|档位|实际末时刻 s|末侧倾 °|首次名义排除 s|该状态预测越限 s|排除/检测状态|','|---|---|---:|---:|---|---|---|']
    for r in summary:
        if r['case'].startswith('single10_v2.6'):
            lines.append('|'+ '|'.join([r['case'].split('seed')[-1],r['label'],f"{r['end_s']:.3f}",f"{r['end_phi_deg']:.2f}",f"{r['first_nominal_exclusion_s']:.3f}" if r['first_nominal_exclusion_s'] is not None else '—',f"{r['first_predicted_crossing_s']:.3f}" if r['first_predicted_crossing_s'] is not None else '—',f"{r['unavoidable_states']}/{r['tested_states']}"])+'|')
    lines+=['','## 所有工况','']
    for name in sorted(set(r['case'] for r in summary)):lines.append(f'[{name}]({name}.png)')
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
