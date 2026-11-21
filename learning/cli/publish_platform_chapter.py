"""Rebuild the six platform figures and four manuscript tables from frozen scores and saved poses."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import mujoco
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw

from report_platform_controls import flight
from sttw_control.playground_prediction import BodyEnvelope


ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'docs/chapter3_paper'
REVIEW=next((ROOT/'docs/evidence').rglob('review_114/p2_pairs.csv')).parent
BLUE,ORANGE='#195d91','#aa4a12'
SECTIONS=['near','mid','far']
SKILLS={'source':'Source','rb_run1':'Full RB (R9)'}


def wall(ax,world):
    for x0,x1,y0,y1 in world['terrain']['forbidden_rectangles']:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#4b4543',edgecolor='black',zorder=4))


def pathline(ax,z,color,label,style='-'):
    q=z['qpos'];ref=z['published_reference_xy']
    ax.plot(*ref.T,':',color=color,alpha=.65,lw=.9)
    ax.plot(*q[:,:2].T,style,color=color,label=label,lw=1.5)
    ax.scatter(*q[-1,:2],c=color,s=14,marker='x')


def finish(fig,name):
    fig.savefig(PAPER/'figures'/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def reversal(b,oracle):
    # First strict Source root on TEST A in the near bin; one success per task side.
    h=pd.read_csv(b/'h1_roots.csv').query("layout=='test_a' and skill=='source' and section=='near' and strict")
    root=h.root.min()
    cases=oracle.query("layout=='test_a' and skill=='source' and force==0 and root==@root and section=='near' and success")
    rows=cases.sort_values(['approach','planned_dep']).drop_duplicates('side').sort_values('side',ascending=False)
    world=json.loads((b/'test_a.json').read_text());edge=world['terrain']['front_x_m']
    fig=plt.figure(figsize=(7.1,4.7),layout='constrained')
    grid=fig.add_gridspec(3,3,height_ratios=[1,1,1.7]);xy=fig.add_subplot(grid[2,:])
    for i,r in enumerate(rows.itertuples()):
        side='Left' if r.side==1 else 'Right';color=(BLUE,ORANGE)[i]
        p=Path(r.trajectory);m=mujoco.MjModel.from_binary_path(str(p.parents[1]/'terrain.mjb'))
        with np.load(p) as z:
            q=z['qpos'];tr=z['trace'];t=tr[:,0];loads=z['contact_substep']
            support=z['wheel_contact_flags'][:,2]&(loads[:,-1,:2]>1e-5).all(1)
            take,end,land=flight(m,q,loads,support,t,edge,edge,.16,0.)
            airborne=np.where((t>=take)&(t<=end))[0]
            apex=airborne[q[airborne,2].argmax()]
            frames=[np.abs(t-take).argmin(),apex,np.abs(t-land).argmin()]
            data=mujoco.MjData(m);camera=mujoco.MjvCamera()
            camera.lookat[:]=[edge,0,.50];camera.azimuth=93;camera.elevation=-8;camera.distance=3.2
            with mujoco.Renderer(m,height=175,width=360) as renderer:
                for j,(idx,title) in enumerate(zip(frames,['Takeoff','Apex','Platform support'])):
                    data.qpos[:]=q[idx];mujoco.mj_forward(m,data);renderer.update_scene(data,camera=camera)
                    ax=fig.add_subplot(grid[i,j]);ax.imshow(renderer.render());ax.axis('off')
                    ax.set_title(f'{side} · {title} · {t[idx]:.3f} s',fontsize=8)
            pathline(xy,z,color,f'{side} gate, approach {r.approach}')
            pose=torch.from_numpy(q[::15,[0,1,2,3,4,5,6,8,9,7,10,11]].copy()).float()
            lo,hi=BodyEnvelope(m,device='cpu')(pose)
            a=lo.amin(1).numpy();c=hi.amax(1).numpy()
            polys=[[(x,y),(x,h),(w,h),(w,y)] for (x,y),(w,h) in zip(a,c)]
            xy.add_collection(PolyCollection(polys,facecolors=color,edgecolors='none',alpha=.10))
    wall(xy,world);xy.axvline(edge,c='.4',ls='--',lw=.7);xy.axvline(12.7,c='.4',ls='-.',lw=.7)
    xy.set(xlim=(8.6,17.8),ylim=(-.85,.85),aspect='equal',xlabel='World x (m)',ylabel='World y (m)')
    xy.legend(ncol=2,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.44));finish(fig,'platform_reversal')


def layouts(b,raw):
    old=pd.read_csv(REVIEW/'p1_roots.csv');new=pd.read_csv(b/'h1_roots.csv')
    verified=pd.read_csv(raw/'physical_scores.csv').query("section=='near'").drop_duplicates(['layout','skill','root','approach'])
    actual=pd.read_csv(b/'h1_approaches.csv').query("section=='near'")
    rates=pd.concat([verified.rename(columns={'verified_jump':'verified'})[['layout','skill','approach','verified']],
                     actual[['layout','skill','approach','verified']]]).groupby(['layout','skill','approach']).verified.mean()
    strict=pd.concat([old,new]).groupby(['layout','skill','section']).strict.mean()
    maps=['primary','alternate','test_a','test_b'];labels=['Primary DEV','19033 DEV','A TEST','B TEST']
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.4),layout='constrained',sharey=True)
    for ax,(skill,name) in zip(axs,SKILLS.items()):
        mat=np.array([[rates.loc[m,skill,a] for a in range(3)]+[strict.loc[m,skill,s] for s in SECTIONS] for m in maps])
        ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.axvline(2.5,color='black',lw=1.4)
        for (y,x),v in np.ndenumerate(mat):ax.text(x,y,f'{100*v:.0f}',va='center',ha='center',color='white' if v>.58 else 'black',fontsize=9)
        ax.set_xticks(range(6),['Base R','New L','New R','Near','Mid','Far'],rotation=38,ha='right');ax.set_yticks(range(4),labels)
        ax.set_title(name);ax.set_xlabel('Handback (%)       Strict reversal (%)')
    finish(fig,'platform_layouts')


def dynamics(raw):
    df=pd.read_csv(raw/'physical_scores.csv');pair=df.query("layout=='alternate' and skill=='rb_run1' and approach==1 and departure=='left' and section=='near' and root in [96,110]")
    fig,(xy,steer,bar)=plt.subplots(1,3,figsize=(8,2.7),layout='constrained',gridspec_kw={'width_ratios':[1.3,1,1]})
    for r in pair.itertuples():
        with np.load(r.trajectory) as z:
            k=round(r.handoff_s/.005)+1;tr=z['trace'];q=z['qpos'];c=BLUE if r.root==96 else ORANGE
            pathline(xy,z,c,f'{r.root} '+('goal' if r.root==96 else 'wall'));steer.plot(tr[k:,0]-tr[k,0],np.rad2deg(q[k:,8]),color=c)
    world=json.loads(Path(json.loads((raw/'plan.json').read_text())['maps']['alternate']).read_text());wall(xy,world)
    xy.set(xlim=(10.8,17.8),ylim=(-.7,.7),xlabel='World x (m)',ylabel='World y (m)',aspect='equal');xy.legend(fontsize=7,loc='lower left',bbox_to_anchor=(0,1))
    steer.set(xlabel='After handback (s)',ylabel='Steering (deg)')
    p=pd.read_csv(REVIEW/'p2_summary.csv').query("skill=='rb_run1' and margin=='shared' and split=='heldout44'").groupby('method').sum(numeric_only=True)
    for i,m in enumerate(['phase2_raw','phase2_relative']):
        r=p.loc[m];bar.bar(np.arange(2)+(i-.5)*.34,[100*r.goal_hits/r.goals,100*r.collision_clear/r.collisions],width=.34,label=['World z','Relative z'][i],color=[ORANGE,BLUE][i])
    bar.set_xticks([0,1],['Goal\nrecall','Walls called\nclear']);bar.set_ylabel('Physical cases (%)');bar.set_ylim(0,100);bar.legend(fontsize=7)
    finish(fig,'platform_state')


def tradeoff(raw,b,metrics):
    table=pd.read_csv(raw.parent/'chapter_mainline/s1/common_bank_summary.csv').query("planner=='C-R'").groupby('method').sum(numeric_only=True)
    old=pd.read_csv(raw/'online/online_cases.csv').query("skill=='rb_run1' and planner in ['fixed','nominal']")
    fig,(dev,test)=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    points=[(k,100*g.collision.mean(),100*g.success.mean()) for k,g in old.groupby('planner')]
    names={'kinematic':'Kinematic .102','phase2_relative':'Dynamic .102','original_0211':'Kinematic .211'}
    points.extend((names[k],100*r.wall/r.n,100*r.success/r.n) for k,r in table.iterrows())
    for i,(label,x,y) in enumerate(points):
        dev.plot(x,y,['o','s','^','D','v'][i],label=label)
    dev.legend(fontsize=7,loc='lower right');dev.set(xlabel='Wall contact (%)',ylabel='Task completion (%)',title='R9 · common DEV bank')
    for skill,name in SKILLS.items():
        g=metrics.query("metric=='success' and planner=='C-R' and skill==@skill and force!='all'").copy()
        g['force']=g.force.astype(int);g=g.sort_values('force');x=g.force.to_numpy()+(-.15 if skill=='source' else .15)
        test.errorbar(x,100*g.rate,yerr=100*np.stack((g.rate-g.low,g.high-g.rate)),marker='o' if skill=='source' else 's',capsize=3,label=name)
    test.set(xlabel='Assigned lateral push (N)',ylabel='Task completion (%)',xticks=[-5,0,5],title='C-R · independent TEST');test.legend(fontsize=7)
    finish(fig,'platform_tradeoff')


def rescues(b):
    m=pd.read_csv(b/'matched_cases.csv');cases=m[m.rescued&m.c_wall&~m.cr_wall].sort_values(['layout','root']).drop_duplicates('layout')
    fig,axes=plt.subplots(2,2,figsize=(7.1,3.9),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    for (xy,angle),r in zip(axes,cases.itertuples(),strict=True):
        world=json.loads((b/(r.layout+'.json')).read_text());wall(xy,world)
        for name,path,c in [('C',r.c_trajectory,ORANGE),('C-R',r.cr_trajectory,BLUE)]:
            with np.load(path) as z:
                pathline(xy,z,c,name);q=z['qpos'];tr=z['trace'];k=np.flatnonzero(tr[:,19]==2)[0]
                angle.plot(tr[k:,0]-tr[k,0],np.rad2deg(q[k:,8]),color=c,label=name)
        xy.set(xlim=(10.6,18),ylim=(-.85,.85),aspect='equal',xlabel='World x (m)',ylabel='World y (m)',title=f'{r.layout}, {SKILLS[r.skill]}, root {r.root}, {r.force:+} N')
        angle.set(xlabel='After lock (s)',ylabel='Steering (deg)');angle.axvline(.30,c='.4',lw=.8,ls=':')
    axes[0,1].legend();finish(fig,'platform_rescues')


def approved_walls(blind):
    from count_platform_histories import fingerprint
    cases = pd.read_csv(blind / 'online_cases.csv').query(
        "skill == 'rb_run1' and planner == 'C-R' and query_feasible and collision").copy()
    cases['history'] = cases.trajectory.map(fingerprint)
    paths = cases.drop_duplicates(['layout', 'root', 'force', 'side', 'history'])
    keys = ['layout','skill','planner','root','force','section','side']
    repeat = pd.read_csv(blind.parent / 'post_review/realtime_15ms/online_cases.csv').set_index(keys)
    fig, axes = plt.subplots(len(paths), 1, figsize=(7.0, 1.42 * len(paths)), layout='constrained')
    for ax, row in zip(axes, paths.itertuples(), strict=True):
        world = json.loads((blind / (row.layout + '.json')).read_text())
        wall(ax, world)
        match = repeat.loc[tuple(getattr(row,k) for k in keys)]
        for path, tick, color, label in [(row.trajectory,row.departure_applied_tick,BLUE,'305/300 ms'),
                                         (match.trajectory,match.departure_applied_tick,ORANGE,'15/15 ms')]:
            with np.load(path) as data:
                pathline(ax,data,color,label)
                q = data['qpos']
                ax.plot(q[tick,0],q[tick,1],'o',ms=3,color=color)
        ax.set(xlim=(10.5, 18.6), ylim=(-.8, .8), ylabel='World y (m)', aspect='equal',
               title=f'{row.layout}, root {row.root}, {row.force:+} N, ' + ('left gate' if row.side == 1 else 'right gate'))
    axes[-1].set_xlabel('World x (m)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    finish(fig, 'platform_approved_walls')


def tables(b,metrics):
    dest=PAPER/'figures'
    def write(name,header,rows,cols):
        lines=[r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
        lines+=[' & '.join(map(str,r))+r' \\' for r in rows]
        (dest/(name+'.tex')).write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    roots=pd.read_csv(b/'h1_roots.csv');g=roots.groupby(['skill','section'])
    rows=[]
    for skill in SKILLS:
        for sec in SECTIONS:
            h=g.get_group((skill,sec));rows.append([SKILLS[skill],sec,h.eligible.sum(),h.strict.sum(),h.fixed_goals.sum(),h.oracle_goals.sum()])
    write('reversal_table',['Skill','Gate','Eligible','Strict','Fixed','Oracle'],rows,'llrrrr')
    def row(skill,planner,metric,force='all'):
        return metrics[(metrics.skill==skill)&(metrics.planner==planner)&(metrics.metric==metric)&(metrics.force==force)].iloc[0]
    def ci(r):return f'{100*r.rate:.1f} [{100*r.low:.1f}, {100*r.high:.1f}]'
    def count(r):return f'{int(r.numerator)}/{int(r.denominator)}'
    rows=[];votes=[]
    for skill in SKILLS:
        for method in ['fixed','nominal','C','C-R']:
            success=row(skill,method,'success');wall=row(skill,method,'collision')
            rows.append([SKILLS[skill],method,int(success.numerator),ci(success),int(wall.numerator),ci(wall)])
        for method in ['C','C-R']:
            votes.append([SKILLS[skill],method,*[count(row(skill,method,k)) for k in ['admitted_success','admitted_wall','veto_oracle_unsolvable']]])
    write('system_table',['Skill','Plan','Done',r'Rate [95\% CI]','Wall',r'Rate [95\% CI]'],rows,'llrrrr')
    write('votes_table',['Skill','Plan','Done/admit','Wall/admit','No sol./veto'],votes,'llrrr')
    rows=[]
    for force in ['-5','0','5']:
        g=metrics[(metrics.metric=='R9_minus_Source')&(metrics.force==force)&(metrics.planner=='C-R')].iloc[0]
        rows.append([force,count(row('source','C-R','success',force)),count(row('rb_run1','C-R','success',force)),ci(g)])
    write('skill_table',['Push (N)','Source','R9',r'Difference [95\% CI]'],rows,'rrrr')


def replay_video(b):
    # First R9 blind wall-to-completion rescue in the near bin (same case as Fig. 5).
    pairs=pd.read_csv(b/'matched_cases.csv')
    case=pairs.query("skill=='rb_run1' and rescued and c_wall and not cr_wall and section=='near'").sort_values(['layout','root']).iloc[0]
    path=Path(case.cr_trajectory);dest=REVIEW/'final_stages'/'verified_test_jump.mp4'
    label=f'TEST A | R9 | C-R | root {case.root} | {case.force:+} N | left gate'
    render_jump_video(path,b/(case.layout+'.json'),dest,label)


def realtime_video(blind):
    repeat=blind.parent/'post_review/realtime_15ms'
    keys=['layout','root','force','section','side']
    old=pd.read_csv(blind/'online_cases.csv').query("skill=='rb_run1' and planner=='C-R' and side==1 and section=='near' and query_feasible and collision")
    new=pd.read_csv(repeat/'online_cases.csv').query("skill=='rb_run1' and planner=='C-R' and verified and success and not collision")
    case=old[keys].merge(new,on=keys,validate='one_to_one').sort_values(['layout','root']).iloc[0]
    path=Path(case.trajectory)
    output=REVIEW/'final_stages'/'realtime_15ms'
    label=f'{case.layout.upper()} | R9 C-R | {case.root} | {case.force:+} N | left gate | 15 ms'
    events=render_jump_video(path,blind/(case.layout+'.json'),output/'verified_jump.mp4',label)
    shutil.copyfile(path,output/'verified_jump.npz')
    receipt=dict(layout=case.layout,root=int(case.root),force_N=int(case.force),section=case.section,
                 side=int(case.side),skill=case.skill,planner=case.planner,trajectory=str(path.relative_to(repeat)),
                 handback_s=case.handoff_s,verified_jump=bool(case.verified),task_success=bool(case.success),
                 wall_contact=bool(case.collision),**events)
    (output/'verified_jump.json').write_text(json.dumps(receipt,indent=2)+'\n')


def render_jump_video(path,map_path,dest,label):
    model=mujoco.MjModel.from_binary_path(str(path.parents[1]/'terrain.mjb'))
    edge=json.loads(map_path.read_text())['terrain']['front_x_m']
    with np.load(path) as z:
        q=z['qpos'];tr=z['trace'];loads=z['contact_substep']
        take,end,land=flight(model,q,loads,z['wheel_contact_flags'][:,2]&(loads[:,-1,:2]>1e-5).all(1),tr[:,0],edge,edge,.16,0.)
    p=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.azimuth=98;camera.elevation=-12;camera.distance=2.85
    samples=np.unique(np.r_[np.arange(0,len(q),10),len(q)-1])
    cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','560x280','-r','20','-i','-',
         '-an','-c:v','libx264','-crf','27','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder, mujoco.Renderer(model,height=280,width=560) as renderer:
        for k in samples:
            t=tr[k,0];camera.lookat[:]=[q[k,0]+.20,q[k,1],max(.28,q[k,2]+.05)]
            p.qpos[:]=q[k];mujoco.mj_forward(model,p);renderer.update_scene(p,camera=camera)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            event='Flight: wheels unloaded' if take<=t<=end else ('Platform support verified' if t>=land else 'Approach / jump transition')
            draw.rectangle((0,0,560,33),fill='#17232c')
            draw.text((6,2),label,fill='white')
            draw.text((6,17),f't = {t:.2f} s    {event}',fill='white')
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():raise RuntimeError('Video encoder failed')
    return dict(takeoff_s=take,airborne_end_s=end,loaded_landing_s=land)


def publish(wide):
    b=wide/'chapter_mainline/blind_test';raw=wide/'review_114';metrics=pd.read_csv(b/'blind_metrics.csv')
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    reversal(b,pd.read_csv(b/'oracle_cases.csv'));layouts(b,raw);dynamics(raw);tradeoff(raw,b,metrics);rescues(b);approved_walls(b);tables(b,metrics);replay_video(b);realtime_video(b)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('wide',type=Path)
    publish(parser.parse_args().wide)
