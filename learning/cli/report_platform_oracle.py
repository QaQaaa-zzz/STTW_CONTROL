"""Publish the per-root physical runs, paired task outcomes and actual XY on the step."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from run_playground_task_reversal import write_rows


def compare_entry_states(raw, baseline, dst):
    records = []
    for skill in ('source','rb_run1'):
        for rootdir in (baseline,raw):
            cap=rootdir/(skill+'_capture')
            with (rootdir/'prefixes.csv').open() as stream:
                audited={row['lane']:row for row in csv.DictReader(stream) if row['skill']==skill}
            for r in json.loads((cap/'status.json').read_text())['results']:
                with np.load(cap/r['case_dir']/'trajectory.npz') as z:
                    trace=z['trace']
                rec=dict(run='baseline' if rootdir==baseline else 'signed_tangent',skill=skill,
                         root=r['variant']['bank_row'],approach=('left','right')[r['route']%2])
                physical=audited[str(r['route'])]
                events=[('entry',r['rb_entry_s']),('takeoff',physical['takeoff_s']),('flight_end',physical['flight_end_s']),
                        ('loaded',physical['loaded_s']),('handoff',r['handoff_s'])]
                for name,t in events:
                    event=trace[np.argmin(abs(trace[:,0]-float(t)))] if t not in (None,'') else None
                    for k,index in [('y_m',2),('yaw_rad',6),('world_vy_m_s',8)]:
                        rec[name+'_'+k]=float(event[index]) if event is not None else None
                records.append(rec)
    write_rows(dst/'paired_entry_states.csv',records)


def report(raw, dest):
    dest.mkdir(exist_ok=True,parents=True)
    plan = json.loads((raw/'plan.json').read_text())
    counts = json.loads((raw/'complete.json').read_text())
    datasets = {}
    for name in ('prefixes','physical_suffixes','oracle_task_reversal','paired_reversal'):
        with (raw/(name+'.csv')).open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for k in ('prefix','trajectory'):
                if k in row:
                    row[k] = str(Path(row[k]).relative_to(raw))
        write_rows(dest/(name+'.csv'),rows)
        datasets[name] = rows
    skills, roots = plan['skills'], plan['rows']
    fig,axes = plt.subplots(len(roots),len(skills)*2,figsize=(15,1.55*len(roots)),squeeze=False,layout='constrained')
    colors = {'left':'#176d89','straight':'#61556b','right':'#bd6123'}
    sections = plan['contracts']['x_sections_m']
    for row in datasets['prefixes']:
        i = roots.index(int(row['root']))
        j = skills.index(row['skill'])*2+(row['approach']=='right')
        ax = axes[i,j]
        with np.load(raw/row['prefix']) as z:
            q,t,ref = z['qpos'],z['trace'][:,0],z['reference_xy']
        last = len(q) if row['verified']=='0' else np.searchsorted(t,float(row['handoff_s'])+.0025)
        ax.plot(*ref.T,color='.55',ls=':',lw=.9)
        ax.plot(*q[:last,:2].T,color='black',lw=1)
        ax.plot(*q[last-1,:2],'o',color='black',ms=2.7)
        ax.add_patch(Rectangle((13.3,-.06),.2,.12,color='black'))
        for x in sections:
            ax.axvline(x,color='.75',lw=.5)
        ax.set_aspect('equal',adjustable='box')
        ax.set(xlim=(8,19),ylim=(-1.05,1.05),yticks=[-1,0,1])
        if not i: ax.set_title(row['skill']+' / '+row['approach']+' approach')
        if i==len(roots)-1: ax.set_xlabel('world x (m)')
        if not j: ax.set_ylabel(f'root {roots[i]}\ny (m)')
    for row in datasets['physical_suffixes']:
        ax = axes[roots.index(int(row['root'])),skills.index(row['skill'])*2+(row['approach']=='right')]
        with np.load(raw/row['trajectory']) as z:
            trace = z['trace']
            xy = trace[np.rint(trace[:,0]/.005).astype(int)>=int(row['branch_tick']),1:3]
            ref = z['published_reference_xy']
        color = colors[row['departure']]
        ax.plot(*ref.T,color=color,lw=.55,ls='--')
        ax.plot(*xy.T,color=color,lw=1)
        ax.plot(*xy[-1],marker='o' if row['goal']=='True' else 'x',color=color,ms=3)
    fig.suptitle('Measured platform trajectories | dashed: requested | dot: goal hold | cross: termination')
    fig.legend(handles=[Line2D([],[],color=c,label=k) for k,c in colors.items()],loc='outside lower center',ncol=3)
    fig.savefig(dest/'xy.png',dpi=125)
    plt.close(fig)
    fig,axes = plt.subplots(1,len(skills),figsize=(6.5,3.7),layout='constrained')
    labels = ['near','mid','far']
    from matplotlib.colors import ListedColormap
    for ax,sk in zip(np.atleast_1d(axes),skills,strict=True):
        mat = np.full((len(roots),3),-1)
        for r in datasets['paired_reversal']:
            if r['skill']==sk and r['reversal']:
                mat[roots.index(int(r['root'])),labels.index(r['bin'])]=int(r['reversal'])
        ax.imshow(mat,vmin=-1,vmax=1,cmap=ListedColormap(['#ddd','#f5eee0','#17735b']),aspect='auto')
        for (y,x),v in np.ndenumerate(mat):
            ax.text(x,y,{-1:'NA',0:'0',1:'1'}[v],ha='center',va='center',color='white' if v==1 else 'black')
        ax.set(xticks=range(3),xticklabels=labels,yticks=range(len(roots)),yticklabels=roots,title=sk,ylabel='DEV root')
    fig.savefig(dest/'h1_phase_map.png',dpi=135)
    plt.close(fig)
    doc = ['# 高台同图左右任务实测','',
        '[各根实际与参考 XY](xy.png) · [近中远双向反转图](h1_phase_map.png) · [逐合同结果](oracle_task_reversal.csv)','',
        f"同一张粗糙高台中央墙地图，{len(roots)} 个开发初态。{counts['prefixes']} 条接近中，{counts['verified']} 条经腾空、台面双轮承重和锁摆交回核验。每条合格前缀实际执行三种驶离，共 {counts['physical_suffixes']} 条物理后缀、{counts['scored_contracts']} 项合同评分。每个方向在首次真实交回后下一个 5 ms 控制步发布。将分支在切换时的位姿、速度、RB 和地面控制器历史与原轨迹逐项配对。",'',
        '| 策略 | 左接近 / 右接近合格 | 左绕 / 直行 / 右绕完成终点保持 | 近 / 中 / 远双向反转根数（分母为两种接近均合格的根） |','|---|---|---|---|']
    for sk in skills:
        pref=[r for r in datasets['prefixes'] if r['skill']==sk]
        tested=[r for r in datasets['physical_suffixes'] if r['skill']==sk]
        h1=[r for r in datasets['paired_reversal'] if r['skill']==sk and r['reversal']!='']
        p=['%d/%d'%(sum(r['verified']=='1' for r in pref if r['approach']==d),sum(r['approach']==d for r in pref)) for d in ('left','right')]
        g=['%d/%d'%(sum(r['goal']=='True' for r in tested if r['departure']==d),sum(r['departure']==d for r in tested)) for d in ('left','straight','right')]
        h=['%d/%d'%(sum(r['reversal']=='1' for r in h1 if r['bin']==b),sum(r['bin']==b for r in h1)) for b in labels]
        doc.append('| '+' | '.join((sk,' · '.join(p),' · '.join(g),' · '.join(h)))+' |')
    doc.extend(['','同根同截面内，对每一种接近，分别检查三种实际驶离中是否有候选能完成左、右合同。两种接近分别只能完成相反一侧时计一次双向反转。失败前缀只记一次，根覆盖不足标为 NA。终点保持仍取实测 0.5 s，并要求通过相应检查点及全程无禁区接触。',
        '','截面位置固定，交回到过面的实测时间见逐合同表。当前证据来自单张已知开发图和有限候选，跨图泛化、预测规划收益与受扰条件下 Full RB 的增益仍需各自配对试验。','',
        '[接近与腾空记录](prefixes.csv) · [每次真实驶离](physical_suffixes.csv) · [逐根反转判定](paired_reversal.csv) · [地图和预算](../protocol.json)'])
    from collections import Counter
    import mujoco
    categories, scales, handbacks = [], [], []
    for sk in skills:
        model = mujoco.MjModel.from_binary_path(str(raw/(sk+'_capture')/'terrain.mjb'))
        md = mujoco.MjData(model)
        robot = np.flatnonzero((model.geom_bodyid>0)&((model.geom_contype|model.geom_conaffinity)!=0))
        wall = next(i for i in range(model.ngeom) if model.geom(i).name.startswith('planning_obstacle_'))
        left_edge = model.geom_pos[wall,1]+model.geom_size[wall,1]
        right_edge = model.geom_pos[wall,1]-model.geom_size[wall,1]
        for label in labels:
            stats=Counter()
            for r in datasets['paired_reversal']:
                if (r['skill'],r['bin'])!=(sk,label): continue
                masks={r['left_mask'],r['right_mask']}
                category=('insufficient' if 'NA' in masks else 'reversal' if masks=={'01','10'}
                          else 'universal' if masks=={'11'} else 'infeasible' if masks=={'00'}
                          else 'shared_one_side' if len(masks)==1 else 'one_way')
                stats[category]+=1
            usable=[r for r in datasets['oracle_task_reversal'] if (r['skill'],r['bin'],r['task'])==(sk,label,'left')]
            times=[float(r['measured_T_s']) for r in usable if r['measured_T_s']]
            categories.append(dict(skill=sk,bin=label,roots=len(roots),**{k:stats[k] for k in (
                'reversal','one_way','universal','infeasible','shared_one_side','insufficient')},
                crossed=len(times),suffixes=len(usable),T_min_s=min(times,default=None),
                T_median_s=np.median(times) if times else None,T_max_s=max(times,default=None)))
        for p in datasets['prefixes']:
            if p['skill']!=sk or p['verified']=='0': continue
            hand=float(p['handoff_s'])
            with np.load(raw/p['prefix']) as z:
                trace=z['trace']
                k=np.argmin(abs(trace[:,0]-hand))
                md.qpos[:]=z['qpos'][k]
                vy0=trace[k,8]
                handbacks.append(dict(skill=sk,root=p['root'],approach=p['approach'],handoff_s=hand,x_m=trace[k,1],
                                     y_m=trace[k,2],yaw_deg=np.rad2deg(trace[k,6]),world_vy_m_s=vy0))
            mujoco.mj_kinematics(model,md)
            rot=md.geom_xmat[robot].reshape(-1,3,3)
            mid=md.geom_xpos[robot]+(rot@model.geom_aabb[robot,:3,None])[...,0]
            half=(abs(rot)@model.geom_aabb[robot,3:,None])[...,0]
            need={'left':max(0.,left_edge-(mid-half)[:,1].min()),
                  'right':max(0.,(mid+half)[:,1].max()-right_edge)}
            scored=[r for r in datasets['oracle_task_reversal'] if (r['skill'],r['lane'],r['task'])==(sk,p['lane'],'left')]
            tracks={}
            for r in scored:
                if r['bin']=='near':
                    with np.load(raw/r['trajectory']) as z:
                        a=z['trace']
                        tracks[r['departure']]=a[a[:,0]>=hand-.00001]
            base=tracks['straight']
            for direction,sign in (('left',1),('right',-1)):
                tr=tracks[direction]
                stop=min(tr[-1,0],base[-1,0])
                ids=tr[:,0]<=stop
                t,vy=tr[ids,0],tr[ids,8]
                delta=abs(vy-np.interp(t,base[:,0],base[:,8]))
                triggers=np.flatnonzero(np.convolve(delta>=.05,np.ones(11,int),'valid')==11) if len(t)>=11 else []
                delay=float(t[triggers[0]]-hand) if len(triggers) else None
                for r in scored:
                    if r['departure']!=direction:continue
                    T=float(r['measured_T_s']) if r['measured_T_s'] else None
                    acc=point=delayed=None
                    if T is not None:
                        v=tr[tr[:,0]<=hand+T+1e-6,8]
                        if len(v)>20:
                            acc=max(0.,float(np.max(sign*(v[20:]-v[:-20])/.1)))
                            point=sign*vy0*T+.5*acc*T*T
                            if delay is not None:delayed=sign*vy0*T+.5*acc*max(T-delay,0.)**2
                    scales.append(dict(skill=sk,root=p['root'],approach=p['approach'],departure=direction,
                        bin=r['bin'],T_s=T,delay_s=delay,max_signed_100ms_accel_m_s2=acc,
                        point_shift_m=point,delayed_shift_m=delayed,whole_body_shift_to_clear_wall_m=need[direction]))
    write_rows(dest/'handoff_states.csv',handbacks)
    write_rows(dest/'coverage_by_section.csv',categories)
    write_rows(dest/'lateral_scales.csv',scales)
    doc.extend(['','[逐档覆盖、反转类别和实测 T 最小/中位/最大值](coverage_by_section.csv) · [点模型横移、含响应延迟横移及整车外廓所需横移](lateral_scales.csv)。横移用交回实测侧速及该开发轨迹的最大同向 100 ms 平均加速度估算。响应延迟取同前缀相对直行的侧速差首次连续 11 帧达到 0.05 m/s 的起点；缺失延迟保留空值。整车横移按交回姿态的碰撞外廓与中央墙边缘计算，这些量用于解释开发轨迹。'])
    doc.extend(['','| 策略及接近 | 交回横向位置中位数 m | 航向中位数 ° | 世界横向速度中位数 m/s |','|---|---:|---:|---:|'])
    for skill in skills:
        for approach in ('left','right'):
            states=[r for r in handbacks if (r['skill'],r['approach'])==(skill,approach)]
            if states:
                med=[np.median([r[k] for r in states]) for k in ('y_m','yaw_deg','world_vy_m_s')]
                doc.append(f'| {skill} / {approach} | {med[0]:+.3f} | {med[1]:+.1f} | {med[2]:+.3f} |')
    doc.extend(['','[逐根交回位置、航向与横向速度](handoff_states.csv)。'])
    if (dest/'full_rb_root1.mp4').exists():
        doc.extend(['','[Full RB 初态 1 的整程录像](full_rb_root1.mp4)。左弯接近和左绕请求，全车腾空后上台、锁摆交回并完成终点保持。录像叠加实测前后轮法向力；各检查点的左右任务成绩另见逐合同表。'])
    if 'entry_tangent_delta' in plan:
        compare_entry_states(raw,Path(plan['baseline']),dest)
        source_paths = {'baseline':Path(plan['baseline']), 'signed_tangent':raw}
        options = {}
        for name,path in source_paths.items():
            with (path/'paired_reversal.csv').open() as f:
                for r in csv.DictReader(f):
                    key = r['skill'],r['root'],r['bin']
                    options.setdefault(key,{}).update({name+'_left':r['left_mask'],name+'_right':r['right_mask']})
        pooled=[]
        for (skill,root,group),masks in options.items():
            valid = [m for m in masks.values() if m!='NA']
            pooled.append(dict(skill=skill,root=root,bin=group,**masks,eligible=len(valid),
                               reversal=int('10' in valid and '01' in valid) if len(valid)>=2 else None))
        write_rows(dest/'four_prefixes.csv',pooled)
        doc.extend(['','[同根四条已实跑接近的合并判定](four_prefixes.csv)。这项开发集分析允许从两轮共四条候选接近中选择；两轮使用同一份完整中立起步快照。'])
        doc.extend(['',f"本轮在 x={plan['entry_x_m']:.2f} m 入口对左右参考施加 ±{plan['entry_tangent_delta']:.2f} 的切线斜率增量。原图、冻结权重、初态和三种驶离沿用基线。[同根入口与交回状态对照](paired_entry_states.csv)"])
    (dest/'RESULTS.md').write_text('\n'.join(doc)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path)
    p.add_argument('dest',type=Path)
    a=p.parse_args()
    report(a.raw,a.dest)
