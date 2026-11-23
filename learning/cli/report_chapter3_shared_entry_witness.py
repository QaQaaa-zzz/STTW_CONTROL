"""Reproduce the root-1174 fixed-entry witness PNG and its 108 archived source rows."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch
from matplotlib import font_manager


def report(p, out):
    out.mkdir(parents=True, exist_ok=True)
    f=p/'analysis/partial_identification/observed/right_offset/part_0/matched_exits.csv.gz'
    bounds=p/'analysis/partial_identification/shared_entry_bounds_by_root_gate.csv'
    rows=pd.read_csv(f).query('root==1174 and planner=="rolling"').copy()
    register=json.loads((p/'protocol.json').read_text())
    legacy={tuple(v) for v in register['old_actions']}
    b=p/'right_offset/part_0/physical'; status=json.loads((b/'status.json').read_text())['results']
    cfg=json.loads((p/'right_offset/world.json').read_text())
    geo=cfg['terrain']; goal=cfg['goal']
    rr=pd.read_csv(bounds)
    with np.load(b/'p0_branch_audit.npz') as z:
        ok=z['pair_ok']; bad=z['first_bad_tick']
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(font).exists(): font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#39434a',
                         'xtick.color':'#59666a','ytick.color':'#59666a','axes.edgecolor':'#a9b1ae'})
    blue,red='#00679a','#a3441b'
    specs=[('near',-1,'近右门',12.7,(0,0.),(3,1.)),('far',1,'远左门',15.,(1,0.),(4,.5))]
    titles=[('近右门','旧：基本右弯，α=0　新：其镜像曲线，α=1'),
            ('远左门','旧：原左弯，α=0　新：减半左弯，α=0.5')]
    fig=plt.figure(figsize=(13.3,5.95),facecolor='white')
    grid=fig.add_gridspec(3,2,left=.055,right=.985,top=.785,bottom=.145,hspace=.28,wspace=.13)
    fig.text(.055,.966,'同一起点，两道门分别找到一条新入口，三次试验各完成两次',fontsize=17,weight='bold',ha='left')
    fig.text(.055,.919,'开发地图 right_offset · 起点 1174 · 入口是事后找到的。每道门用各自固定入口跑三种侧推，落地后用同一滚动规划器。',fontsize=10,color='#374747')
    for j,(t,sub) in enumerate(titles):
        x=.055+j*.493
        fig.text(x,.856,t+'    旧库最佳 0/3，新入口 2/3',fontsize=12,weight='bold',ha='left')
        fig.text(x,.815,sub,fontsize=9,color='#374747',ha='left')
    labels={'goal':'完成','tracking_domain_exit':'超出跟踪范围','forbidden_obstacle_contact':'碰墙','physical_failure':'物理失败','timeout':'超时'}
    csv=[]
    for col,(sec,side,name,gate_x,old,new) in enumerate(specs):
        g=rows[(rows.section==sec)&(rows.side==side)].copy()
        assert len(g)==54 and not g.duplicated(['geometry','alpha','force']).any()
        g['is_legacy']=[(i,a) in legacy for i,a in zip(g.geometry,g.alpha)]
        g['fixed_across_forces_successes']=g.groupby(['geometry','alpha']).success.transform('sum')
        assert g[g.is_legacy].fixed_across_forces_successes.max()==0 and g.fixed_across_forces_successes.max()==2
        assert g[(g.geometry==new[0])&(g.alpha==new[1])].success.sum()==2
        for cell,limit in [('B',0),('D',2)]:
            t=rr[(rr.root==1174)&(rr.section==sec)&(rr.side==side)&(rr.rule=='strict')&(rr.measure==cell)]
            assert len(t)==1 and t.iloc[0].lower==t.iloc[0].upper==limit
        for q in g.itertuples():
            src=Path(q.trajectory); m=status[q.lane]; v=q.geometry,q.alpha; isdraw=v in (old,new)
            assert Path(m['case_dir'])==Path(src.parent.name) and ok[q.group_start_lane//7]
            with np.load(src) as z:
                xy=z['qpos'][:,:2]; ref=z['reference_xy']; held=z['published_reference_xy']; t=z['trace'][:,0]
            assert np.isfinite(xy).all() and np.isclose(q.duration_s,t[-1])
            csv.append(dict(layout=q.layout,batch=q.batch,root=q.root,gate=sec,side=side,gate_x_m=gate_x,
                solver=q.planner,entry_geometry=q.geometry,entry_shape=register['geometry'][q.geometry],
                entry_alpha=q.alpha,in_old_five=q.is_legacy,role_in_figure=('old_gate_default' if v==old else
                    'expanded_fixed_entry' if v==new else 'other_old' if q.is_legacy else 'other_expanded'),
                world_y_force_N=q.force,force_available_to_prejump_entry=False,
                selection='retrospective DEV shared entry across -5,0,+5 N; rollout of each specified frozen entry',
                success=q.success,success_count_same_entry_all_three=int(q.fixed_across_forces_successes),
                strict_seven_prefix_audit_ok=bool(ok[q.group_start_lane//7]),first_bad_tick=int(bad[q.group_start_lane//7]),
                bank_row=m['variant']['bank_row'],group_start_lane=q.group_start_lane,lane=q.lane,verified_handoff=q.verified,
                crossing_y_m=q.cross_y,correct_gate=q.gate,wall=q.wall,goal=m['goal'],hold_s=q.hold_s,
                termination=q.termination,duration_s=q.duration_s,terminal_x_m=xy[-1,0],terminal_y_m=xy[-1,1],
                rb_entry_s=m['rb_entry_s'],handoff_s=q.handoff_s,published_updates=int(q.published),
                trajectory_npz=str(src),actual_field='qpos[:, :2]',reference_fields='reference_xy before RB entry; published_reference_xy at final recorded state clipped from handback to endpoint',
                physical_status=str(b/'status.json'),pairing_file=str(b/'p0_branch_audit.npz'),matched_exits_file=str(f),
                strict_bounds_file=str(bounds),world=str(p/'right_offset/world.json')))
        for i,force in enumerate([-5,0,5]):
            ax=fig.add_subplot(grid[i,col],aspect='equal')
            ax.set(xlim=(1.,18.8),ylim=(-1.12,1.55),yticks=[-1,0,1],xticks=[2,5,8,11,14,17])
            ax.tick_params(length=2,labelsize=8,pad=2)
            ax.axhline(0,color='#e4e8e6',lw=.5,zorder=0)
            ax.axvline(geo['front_x_m'],c='#809096',ls=':',lw=.9,zorder=1)
            ax.add_patch(Rectangle((goal['x_min_m'],-goal['abs_y_max_m']),goal['x_max_m']-goal['x_min_m'],2*goal['abs_y_max_m'],fc='#d7ecdd',ec='#8ca99a',lw=.6,zorder=0))
            for x0,x1,y0,y1 in geo['forbidden_rectangles']:
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fc='#352f2b',zorder=7))
            ax.vlines(gate_x,*sorted((.06*side,3.9*side)),color='#705397',lw=1.9,zorder=1)
            remarks=[]
            for v,color,label in [(old,red,'旧'),(new,blue,'新')]:
                q=g[(g.force==force)&(g.geometry==v[0])&(g.alpha==v[1])].iloc[0]
                m=status[int(q.lane)]
                with np.load(q.trajectory) as z:
                    xy=z['qpos'][:,:2];ref=z['reference_xy'];held=z['published_reference_xy'];t=z['trace'][:,0]
                def clip(curve,lo,hi):
                    return curve[(curve[:,0]>=lo)&(curve[:,0]<=hi)]
                if m['rb_entry_s'] is not None:
                    take=int(np.abs(t-m['rb_entry_s']).argmin())
                    rr0=clip(ref,xy[0,0],xy[take,0])
                    ax.plot(*rr0.T,color=color,lw=1,ls=(0,(1,2)),zorder=2)
                if pd.notna(q.handoff_s):
                    hand=int(np.abs(t-q.handoff_s).argmin())
                    rr1=clip(held,xy[hand,0],xy[-1,0])
                    ax.plot(*rr1.T,color=color,lw=1,ls=(0,(5,3)),zorder=2)
                ax.plot(*xy.T,lw=1.9,color=color,zorder=4,alpha=.94)
                ax.plot(*xy[-1],marker='o' if q.success else 'x',markersize=6,markerfacecolor='white',mew=1.5,color=color,zorder=8)
                remarks.append(label+' '+labels[q.termination])
            force_label={-5:'−5 N',0:'0 N',5:'+5 N'}[force]
            ax.set_title(force_label+'    '+ '  /  '.join(remarks),loc='left',fontsize=9,pad=3,color='#26353d')
            if col==0: ax.set_ylabel('y（m）',fontsize=8)
            if i==2: ax.set_xlabel('x（m）',fontsize=8,labelpad=1)
            else: ax.tick_params(labelbottom=False)
    handles=[Line2D([],[],color=red,lw=2,label='旧入口实跑'),Line2D([],[],color=blue,lw=2,label='新入口实跑'),
             Line2D([],[],color='#465652',ls=':',lw=1.3,label='跳前参考'),Line2D([],[],color='#465652',ls='--',lw=1.3,label='末帧持有的出口参考'),
             Line2D([],[],color='#41524a',marker='o',ls='',mfc='white',label='任务完成'),Line2D([],[],color='#41524a',marker='x',ls='',label='失败终点'),
             Line2D([],[],color='#705397',lw=2,label='指定门侧'),Patch(facecolor='#352f2b',label='墙'),Patch(facecolor='#d7ecdd',edgecolor='#8ca99a',label='目标区')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.53,.076),frameon=False,ncol=9,columnspacing=1.15,handlelength=1.7,fontsize=8)
    fig.text(.055,.023,'等比例世界坐标，实线记录车体中心，灰色竖虚线为台阶前缘。参考线按阶段与行进范围裁切。图中每道门各画一条旧路线，源表列出全部 18 条入口。',fontsize=8,color='#465458')
    png=out/'root1174_shared_entry_xy.png'
    fig.savefig(png,dpi=180,metadata={'Description':'Paired archived physical endpoints from right_offset/part_0, root 1174 for near right and far left. Six trials per displayed gate, 108 archive rows in source CSV. Same fixed entry across forcing levels; selected retrospectively.'})
    plt.close(fig)
    source=out/'root1174_shared_entry_sources.csv'
    a=pd.DataFrame(csv).sort_values(['gate','entry_geometry','entry_alpha','world_y_force_N']);a.to_csv(source,index=False)
    print(png,source,len(a))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_continuous_p0b_20261123'),
                        help='P0-B archive with the physical traces and partial-identification tables')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2]/'docs/evidence/chapter3_continuous_planning/p0_b',
                        help='Directory for root1174_shared_entry_xy.png and root1174_shared_entry_sources.csv')
    args = parser.parse_args()
    report(args.run, args.output)
