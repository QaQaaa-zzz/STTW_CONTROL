"""Cross-experiment STTW evidence atlas; no new simulation or result-based fitting."""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_roll_braking_boundary import margin


def csvout(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(repo,base,out):
    out.mkdir(exist_ok=True,parents=True);plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    entries=json.loads((repo/'docs/evidence/sttw_retained_experiments/catalog.json').read_text())['entries'];inventory=[]
    for e in entries:
        p=Path(e['source']);text=p.read_text() if p.exists() else '';inventory.append(dict(id=e['id'],experiment=e['experiment'],model=e['model'],source=str(p),source_exists=p.exists(),report_chars=len(text),claim=e['result'],scope=e['boundary'],level='report_read' if text else 'missing'))
    csvout(out/'historical_inventory.csv',inventory)
    metrics=list(csv.DictReader((repo/'docs/evidence/h75_evidence_queue_20261110/final_results/metrics.csv').open()));csvout(out/'all1432.csv',metrics)
    stats=[]
    def fraction(rows,key,group):
        n=len(rows);k=sum(r[key]=='True' for r in rows);stats.append(dict(group=group,metric=key,n=n,k=k));return k,n
    colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00'];alphas=['0','0.25','0.5','0.75','1'];fullfixed=['fixed'+a for a in alphas]
    # Same pulse, different terrain; and unchanged policy over height.
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for lab,color in zip(fullfixed,colors):
        vals=[]
        for h in [0,5,10,20]:
            rs=[r for r in metrics if r['stage']=='h75_height_scan' and r['label']==lab and r['case'].startswith(f'height{h:02d}mm')];k,n=fraction(rs,'final_hold',f'height{h}:{lab}');vals.append(k);assert n==6
        axes[0].plot([0,5,10,20],vals,'o-',color=color,label='α='+lab[5:])
    for terrain,color in [('flat','#0072B2'),('dense10','#D55E00')]:
        cases=[r for r in metrics if r['stage']=='h75_force_scan' and r['label'] in fullfixed and r['case'].startswith(terrain+'_force')];forces=sorted(set(int(re.search(r'force([+-]?\d+)',r['case'])[1]) for r in cases));vals=[]
        for force in forces:
            rs=[r for r in cases if int(re.search(r'force([+-]?\d+)',r['case'])[1])==force];k,n=fraction(rs,'full_window',f'force{force}:{terrain}');vals.append(k);assert n==30
        axes[1].plot(forces,vals,'o-',color=color,label='平地' if terrain=='flat' else '10 mm石子')
    axes[0].set(xlabel='石子高度（mm）',ylabel='末段保持条数 / 6',ylim=(-.2,6.4),title='同一H75：地形难度改变可完成范围');axes[1].set(xlabel='侧推力（N）',ylabel='完整10秒条数 / 30',ylim=(-1,32),title='相同侧推：地形改变受扰后的结局')
    for ax in axes:ax.legend();ax.grid(alpha=.18)
    fig.suptitle('证据一：地形与外扰共同决定任务难度',fontsize=18);fig.savefig(out/'01_TERRAIN_FORCE.png',dpi=180);plt.close(fig)
    # Channel interaction, every alpha and all six contexts.
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,terrain in zip(axes,['dense10','single10']):
        matrix=[]
        for mode in ['full','speed_only','steer_only']:
            line=[]
            for alpha in alphas:
                rs=[r for r in metrics if r['stage']=='h75_channel_scan' and r['case'].startswith(terrain) and r['label']==mode+'_'+alpha];k,n=fraction(rs,'final_hold',terrain+':'+mode+':'+alpha);assert n==6;line.append(k)
            matrix.append(line)
        im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=6,aspect='auto')
        for i in range(3):
            for j in range(5):ax.text(j,i,f'{matrix[i][j]}/6',ha='center',va='center',color='white' if matrix[i][j]>=4 else 'black',fontsize=14)
        ax.set(xticks=range(5),xticklabels=alphas,yticks=range(3),yticklabels=['速度＋转向','仅速度','仅转向'],xlabel='α',title='10 mm石子' if terrain=='dense10' else '10 cm单脊')
    fig.suptitle('证据二：速度与转向的配合效果取决于地形\n每格为末段保持；相同两速度×三物理组',fontsize=16);fig.savefig(out/'02_CHANNELS.png',dpi=180);plt.close(fig)
    # Task preferences and honest selector comparison.
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    for key,color,label in [('task_A','#0072B2','任务A：路径更严'),('task_B','#D55E00','任务B：速度更严')]:
        vals=[]
        for lab in fullfixed:
            rs=[r for r in metrics if r['stage']=='h75_height_scan' and r['case'].startswith('height05mm') and r['label']==lab];k,n=fraction(rs,key,'height05:'+lab);vals.append(k)
        axes[0].plot(list(map(float,alphas)),vals,'o-',color=color,label=label)
    best=[];rules=[]
    for task in ['A','B','C']:
        vals=[]
        for lab in fullfixed+['rule_'+task]:
            rs=[r for r in metrics if r['stage']=='h75_selector_layout_test' and 'height10mm' in r['case'] and r['label']==lab];k,n=fraction(rs,'task_'+task,'layout10:'+lab);assert n==16;vals.append(k)
        best.append(max(vals[:-1]));rules.append(vals[-1])
    x=np.arange(3);axes[1].bar(x-.17,best,.34,label='测试后最佳固定档',color='#0072B2');axes[1].bar(x+.17,rules,.34,label='预先固定的规则',color='#E69F00');axes[1].set(xticks=x,xticklabels=['A','B','C'],ylabel='任务通过条数 / 16',ylim=(0,16),title='新布局：简单选档尚未普遍优于固定档')
    axes[0].set(xlabel='α',ylabel='任务通过条数 / 6',ylim=(-.2,6.4),title='5 mm石子：同一α的优势随任务改变')
    for ax in axes:ax.legend(fontsize=9);ax.grid(axis='y',alpha=.18)
    fig.suptitle('证据三：可调取舍成立，选档收益需要单独验证',fontsize=17);fig.savefig(out/'03_TASK_SELECTION.png',dpi=180);plt.close(fig)
    # Recompute all216 physical model diagnostics plus exact pursuit invalid flags.
    run=base/'h75_channel_scan';src=run/'roll_contact_analysis';route=np.load(repo/'runs/h75_selected_local300_flat/bundle/route.npz');cfg=json.loads((repo/'runs/h75_selected_local300_flat/bundle/config.json').read_text())['follower'];diag=[];data={};B=6146.053403416946;limit=np.degrees(.7)
    for p in sorted(src.glob('*_seed*.npz')):
        z=np.load(p);raw=np.load(run/p.stem/'trajectory.npz');data[p.stem]={k:z[k] for k in z.files}
        for lane,lab in enumerate(z['labels']):
            ok=z['valid'][:,lane];ii=np.flatnonzero(ok);last=ii[-1];phi=np.degrees(z['phi'][ok,lane]);omega=np.degrees(z['euler_roll_rate'][ok,lane]);h=margin(phi,omega,limit,B)[2];t=z['time'][ok,lane];rs=next(r for r in metrics if r['stage']=='h75_channel_scan' and r['case']==p.stem and r['label']==lab);pose=raw['pose'][last,lane];progress=raw['progress'][last,lane];v=z['speed'][last,lane];look=np.clip(cfg['lookahead_time_s']*max(v,cfg['lookahead_min_speed_m_s']),cfg['lookahead_min_m'],cfg['lookahead_max_m']);point=np.array([np.interp(progress+look,route['s'],route['xy'][:,j]) for j in range(2)]);delta=point-pose[:2];px=np.cos(pose[2])*delta[0]+np.sin(pose[2])*delta[1];hd=np.interp(progress,route['s'],route['heading'])-pose[2];hd=np.arctan2(np.sin(hd),np.cos(hd));pre=np.flatnonzero((h<0)&(abs(phi)<limit))
            diag.append(dict(case=p.stem,label=str(lab),cause=rs['cause'],task_C=rs['task_C'],final_hold=rs['final_hold'],H_min_deg=float(h.min()),first_H_negative_pre_limit_s=float(t[pre[0]]) if len(pre) else None,end_s=float(t[-1]),preview_body_x_m=float(px),heading_error_deg=float(np.degrees(hd)),preview_invalid=bool(px<=cfg['preview_body_x_min_m']),heading_invalid=bool(abs(hd)>=np.pi/2),policy_fault=bool(raw['policy_fault'][last,lane]),domain_logged=bool(raw['domain_exit'][last,lane])))
    csvout(out/'all216_diagnostics.csv',diag)
    # Selected contrasting behaviors, fixed representative seeds from prior work.
    examples=[('single10_v2.6_seed101',['full_0.5','full_1'],'平衡恢复与侧倾超限'),('single10_v2.6_seed103',['full_0.5','full_1'],'H为正仍可能退出跟踪域'),('single10_v2.6_seed101',['full_0.5','speed_only_0.5','steer_only_0.5'],'单脊：双通道协同'),('dense10_v2.3_seed101',['full_0.75','speed_only_0.75','steer_only_0.75'],'石子：仅转向也能完成任务')]
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');xydata={}
    for ax,(case,labs,title) in zip(axes.flat,examples):
        z=data[case];points=[]
        for lab,color in zip(labs+['Zero'],['#009E73','#D55E00','#0072B2','#444444'] if len(labs)==3 else ['#009E73','#D55E00','#444444']):
            lane=list(z['labels']).index(lab);ok=z['valid'][:,lane];xy=z['xy'][ok,lane];points.append(xy);ax.plot(xy[:,0],xy[:,1],color=color,label=lab,lw=1.8);ax.scatter(*xy[-1],marker='x',color=color);xydata[case+'__'+lab+'__xy']=xy
        points=np.concatenate(points);ax.plot(route['xy'][:,0],route['xy'][:,1],':',color='gray',label='原固定路径');ax.set(xlim=(points[:,0].min()-.4,points[:,0].max()+.4),ylim=(points[:,1].min()-.4,points[:,1].max()+.4),xlabel='X（m）',ylabel='Y（m）',title=title+'\n'+case);ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8);ax.grid(alpha=.15)
    for ax,(case,labs,title) in zip(axes.flat,examples):
        inset=ax.inset_axes([.81,.57,.17,.37])
        for lab,col in zip(labs+['Zero'],['#009E73','#D55E00','#0072B2','#444444'] if len(labs)==3 else ['#009E73','#D55E00','#444444']):
            xy=xydata[case+'__'+lab+'__xy'];inset.plot(xy[:,0],xy[:,1],color=col,lw=.8)
        inset.set_aspect('equal',adjustable='box');inset.tick_params(labelsize=5);inset.set_title('完整运行',fontsize=7)
        ax.set_xlim(8,15.5);ax.set_ylim(-.5,7);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('配对XY：主图放大越障与回正区域，右上小图保留完整运行',fontsize=15)
    fig.savefig(out/'00_XY_CASES.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,(case,labs,title) in zip(axes,examples[:2]):
        z=data[case]
        for lab,col in zip(labs,['#009E73','#D55E00']):
            lane=list(z['labels']).index(lab);valid=z['valid'][:,lane];t=z['time'][valid,lane];h=margin(np.degrees(z['phi'][valid,lane]),np.degrees(z['euler_roll_rate'][valid,lane]),limit,B)[2];sel=(t>=1.15)&(t<=3.2);ax.plot(t[sel],h[sel],color=col,label=lab,lw=2);xydata[case+'__'+lab+'__time']=t;xydata[case+'__'+lab+'__H']=h
            r=next(r for r in diag if r['case']==case and r['label']==lab)
            if t[-1]<3.2:ax.scatter(t[-1],h[-1],marker='X',s=70,color=col);ax.annotate('侧倾超限' if r['cause']=='roll_limit' else '前视域停机',xy=(t[-1],h[-1]),xytext=(16,18),textcoords='offset points',color=col,arrowprops={'arrowstyle':'->','color':col})
        ax.set_ylim(-35,43);ax.axhline(0,color='black',lw=1);ax.set(xlabel='时间（s）',ylabel='模型双侧H（°）',title=title+'\n'+case);ax.legend();ax.grid(alpha=.15)
    fig.suptitle('证据四：平衡余量与跟踪有效域是不同约束\n同一名义模型B=6146°/s²；φc=40.11°',fontsize=16);fig.savefig(out/'04_TWO_FAILURES.png',dpi=180);plt.close(fig);np.savez_compressed(out/'case_curves.npz',**xydata)
    # Historical low-speed physical-contact experiment is kept separate.
    old=next(e for e in inventory if e['id']=='D01');text=Path(old['source']).read_text();oldrows=[]
    for line in text.splitlines():
        cells=[x.strip() for x in line.split('|')[1:-1]]
        if len(cells)==8 and re.fullmatch(r'\d+\.\d+',cells[0]):oldrows.append(dict(v=float(cells[0]),steer=float(cells[1]),actual_v=float(cells[2]),nonzero_no_contact=int(cells[5]),n=8))
    assert len(oldrows)==12;csvout(out/'historical_low_speed.csv',oldrows);fig,ax=plt.subplots(figsize=(9,5),layout='constrained');speeds=sorted(set(r['v'] for r in oldrows));vals=[sum(r['nonzero_no_contact'] for r in oldrows if r['v']==v) for v in speeds];ax.bar([str(v) for v in speeds],vals,color=['#56B4E9','#0072B2','#009E73','#009E73'])
    for i,n in enumerate(vals):ax.text(i,n+.3,f'{n}/24',ha='center')
    ax.set(xlabel='参考速度（m/s）',ylabel='非零侧推后4秒无触地条数 / 24',ylim=(0,27),title='历史底层固定测试：低速区抗侧推较弱\n三转角×八非零力档；不同工作状态，未运行H75中层');fig.savefig(out/'05_LOW_SPEED.png',dpi=180);plt.close(fig)
    csvout(out/'aggregate_counts.csv',stats)
    summary=dict(historical_reports=len(inventory),historical_reports_accessible=sum(x['source_exists'] for x in inventory),recent_episodes=len(metrics),recent_contexts=len(set((r['stage'],r['case']) for r in metrics)),mechanism_trajectories=len(diag),domain_logged=sum(r['domain_logged'] for r in diag),domain_preview_trigger=sum(r['domain_logged'] and r['preview_invalid'] for r in diag),domain_heading_trigger=sum(r['domain_logged'] and r['heading_invalid'] for r in diag),domain_neither=sum(r['domain_logged'] and not(r['preview_invalid'] or r['heading_invalid']) for r in diag),new_simulation=0,new_training=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--base',type=Path,default=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.repo,a.base,a.output)
