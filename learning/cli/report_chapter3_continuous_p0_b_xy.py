"""Root-paged paired XY evidence for all physical executions in the P0-B experiment."""
import argparse
import base64
import gzip
import html
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from report_chapter3_continuous_p0_xy import COLOURS, sample_mm


def scene(run, layout):
    world=json.loads((run/layout/'world.json').read_text())
    return dict(walls=world['terrain']['forbidden_rectangles'],edge=world['terrain']['front_x_m'],
                goal=world['goal'],section={'near':12.7,'far':15.})


def compact_xy(path, handoff, takeoff):
    with np.load(path) as z:
        xy=z['qpos'][:,:2]
        time=z['trace'][:,0]
        final=z['published_reference_xy']
        approach=z['reference_xy']
    finite=np.flatnonzero(np.isfinite(xy).all(1))
    if not len(finite):
        raise ValueError(f'No measured positions in {path}')
    markers={}
    event_indices=[]
    for key, value in [('handback',handoff),('jump_entry',takeoff)]:
        idx=int(abs(time-value).argmin()) if pd.notna(value) else None
        if idx is not None and np.isfinite(xy[idx]).all():
            markers[key]=np.rint(xy[idx]*1000).astype(int).tolist()
            event_indices.append(idx)
        else:
            markers[key]=None
    segments=[]
    for seq in np.split(finite, np.flatnonzero(np.diff(finite)!=1)+1):
        extras=[idx-seq[0] for idx in event_indices if seq[0]<=idx<=seq[-1]]
        segments.append(sample_mm(xy[seq],10,extras))
    return dict(parts=segments,**markers,finite_end_s=float(time[finite[-1]]),
                end_s=float(time[-1]),missing_frames=len(time)-len(finite)),approach,final


def pack(data):
    raw=json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    return base64.b64encode(gzip.compress(raw,mtime=0)).decode()


def displayed_outcome(row):
    if row.success:
        return 'completed'
    if row.termination == 'goal':
        return 'goal reached; gate or verified jump unmet'
    return str(row.termination)


def representatives(outcomes):
    cols=['layout','root','force','geometry','alpha','section','side']
    old=outcomes[outcomes.planner=='old_filter']
    cem=outcomes[outcomes.planner=='rolling']
    paired=old.merge(cem,on=cols,suffixes=('_old','_cem'),validate='one_to_one')
    picked=[]
    for _,grp in paired.sort_values(cols).groupby(['layout','section','side']):
        disagree=False
        for flag,subset in [('lost',grp[grp.success_old & ~grp.success_cem]),
                            ('rescued',grp[~grp.success_old & grp.success_cem])]:
            if len(subset):
                picked.append(dict(subset.iloc[0],comparison=flag))
                disagree=True
        if not disagree:
            failed=grp[~grp.success_cem]
            picked.append(dict((failed if len(failed) else grp).iloc[0],comparison='both_failed' if len(failed) else 'both_completed'))
    return pd.DataFrame(picked)


def save_examples(picked, scenes, folders, out, labels=('Old exit filter','Rolling CEM')):
    files=[]
    for (layout,section), group in picked.groupby(['layout','section']):
        geo=scenes[layout]
        fig, axes=plt.subplots(len(group),1,figsize=(12,2.8*len(group)),layout='constrained',squeeze=False)
        for ax,row in zip(axes[:,0],group.itertuples(),strict=True):
            g=geo['goal']
            ax.add_patch(Rectangle((g['x_min_m'],-g['abs_y_max_m']),g['x_max_m']-g['x_min_m'],2*g['abs_y_max_m'],
                                   color='#d7eada',zorder=0))
            for x0,x1,y0,y1 in geo['walls']:
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#464646',zorder=5))
            ax.vlines(geo['section'][section],min(.06*row.side,3.9*row.side),max(.06*row.side,3.9*row.side),
                      color='#80651d',lw=1)
            ax.axvline(geo['edge'],color='#777777',ls=':',lw=.8)
            low,high=-1.4,1.4
            xlo,xhi=geo['edge'],g['x_max_m']
            for suffix,col,marker in [('old',COLOURS[0],'o'),('cem',COLOURS[1],'s')]:
                path=Path(getattr(row,'trajectory_'+suffix))
                with np.load(path) as z:
                    actual=z['qpos'][:,:2]
                    times=z['trace'][:,0]
                    ref=z['published_reference_xy']
                low=min(low,np.nanmin(actual[:,1])-.2)
                high=max(high,np.nanmax(actual[:,1])+.2)
                xlo=min(xlo,float(np.nanmin(actual[:,0]))-.2); xhi=max(xhi,float(np.nanmax(actual[:,0]))+.2)
                ax.plot(*ref.T,color=col,ls=':',alpha=.85,lw=.9)
                ax.plot(*actual.T,color=col,lw=1.6)
                end=np.flatnonzero(np.isfinite(actual).all(1))[-1]
                ax.plot(*actual[end],marker=marker,mfc='white' if getattr(row,'success_'+suffix) else col,color=col,ms=6)
                for t,symbol in [(getattr(row,'handoff_s_'+suffix),'|'),(folders[(row.batch_old,row.group_start_lane_old) ],'^')]:
                    if pd.notna(t):
                        idx=int(abs(times-t).argmin())
                        ax.plot(*actual[idx],marker=symbol,color=col,ms=7,ls='none',mfc='none')
            side='left' if row.side>0 else 'right'
            ax.set(title=f'{side} {section} gate / {row.comparison} / start {row.root}, {row.force:+g} N, curve {row.geometry}, alpha {row.alpha:g}',
                   xlim=(xlo,xhi),ylim=(low,high),aspect='equal',xlabel='World x (m)',ylabel='World y (m)')
        axes[0,0].legend(handles=[Line2D([],[],color=COLOURS[0],marker='o',label=labels[0]),Line2D([],[],color=COLOURS[1],marker='s',label=labels[1]),
            Line2D([],[],color='.3',ls=':',label='Last published reference'),Line2D([],[],color='.3',ls='',marker='^',label='Jump handoff'),
            Line2D([],[],color='.3',ls='',marker='|',label='Ground handback')], loc='upper left',frameon=False,ncol=5,fontsize=8)
        fig.suptitle(f'{layout} / paired trajectories from an identical approach; full dynamics through the real endpoint')
        path=out/f'{layout}_{section}_paired_xy.png';fig.savefig(path,dpi=144);plt.close(fig);files.append(path.name)
    return files


def write_history(run,out,events):
    """Read each accepted publication file once. A small script for a single tick is loaded on demand."""
    timelines={}
    for (batch,tick,kind), rows in events.groupby(['batch','change_tick','kind']):
        batchdir=run/batch/'physical'/'p0_published'
        prefix='tick' if kind=='accepted_plan' else 'fallback'
        source=batchdir/f'{prefix}_{int(tick):04d}.npz'
        path=out/'versions'/batch/f'{prefix}_{int(tick):04d}.js'
        path.parent.mkdir(parents=True,exist_ok=True)
        with np.load(source) as z:
            ids=z['lanes']; curves=z['xy']
            if set(map(int,ids))!=set(rows.lane.astype(int)):
                raise ValueError(f'Publication receipts disagree with trajectory file: {source}')
            encoded=pack({int(lane):sample_mm(curve,6) for lane,curve in zip(ids,curves,strict=True)})
        relative=path.relative_to(out).as_posix()
        path.write_text(f'window.p0Versions[{json.dumps(relative)}]={json.dumps(encoded)};')
        for r in rows.itertuples():
            receipt=lambda t: int(t) if pd.notna(t) else None
            timelines.setdefault((batch,int(r.lane)),[]).append([int(tick),relative,os.path.relpath(source,out),
                    kind,receipt(r.h75_actor_consumed_tick),receipt(r.pursuit_consumed_tick)])
    return timelines


def export(run, analysis):
    out=analysis/'xy';out.mkdir(parents=True,exist_ok=True)
    lanes=pd.read_csv(analysis/'physical_lanes.csv.gz')
    decisions=pd.read_csv(analysis/'measured_exits.csv.gz')
    prefixes=pd.read_csv(analysis/'prefixes.csv')
    changes=pd.read_csv(analysis/'route_changes_and_consumers.csv.gz')
    plan=json.loads((run/'protocol.json').read_text())
    if len(lanes)!=plan['trajectory_lanes'] or len(prefixes)!=plan['unique_prefixes']:
        raise ValueError('XY exporter requires a complete physical lane index')
    index=prefixes.set_index(['batch','group_start_lane']).rb_entry_s.to_dict()
    geo={layout:scene(run,layout) for layout in plan['layouts']}
    timelines=write_history(run,out,changes) if len(changes) else {}
    template=Path(__file__).with_name('chapter3_p0_b_xy_viewer.html').read_text()
    pages=[]; lane_index=[]
    for (layout,root), group in lanes.groupby(['layout','root']):
        refids={};refs=[]; trajectories={}; pairs={}
        def addref(ref):
            key=ref.tobytes()
            if key not in refids:
                if not np.isfinite(ref).all():
                    raise ValueError('Non-finite reference coordinates')
                refids[key]=len(refs);refs.append(sample_mm(ref,6))
            return refids[key]
        p=out/f'{layout}_root_{root}.html'
        score=decisions[(decisions.layout==layout)&(decisions.root==root)]
        for r in group.itertuples():
            k=(r.batch,r.group_start_lane)
            data,approach,final=compact_xy(Path(r.trajectory),r.handoff_s,r.rb_entry_s)
            key=f'{r.force:g}/{r.geometry}/{r.alpha:g}'
            laneset=pairs.setdefault(key,dict(arms={},task={},paired=bool(r.pair_ok)))
            laneset['arms'][r.arm]=int(r.lane)
            raw=os.path.relpath(r.trajectory,out)
            data.update(lane=int(r.lane),arm=r.arm,termination=r.termination,wall=bool(r.wall),
                        verified=bool(r.verified),goal=bool(r.goal),entry_ref=addref(approach),final_ref=addref(final),
                        hist=timelines.get((r.batch,r.lane),[]),raw=raw,initial_default_ticks=int(r.initial_default_ticks),
                        expired_default_ticks=int(r.expired_default_ticks),accepted_count=int(r.published),
                        gates={f'{sec}/{side}':bool(.06 <= getattr(r,sec+'_cross_y')*sign <= 3.9)
                               for sec in ['near','far'] for side,sign in [('left',1),('right',-1)]})
            trajectories[int(r.lane)]=data
            lane_index.append(dict(layout=layout,root=root,force=r.force,geometry=r.geometry,alpha=r.alpha,

                arm=r.arm,lane=r.lane,trajectory=r.trajectory,page=p.name,selection_key=key,
                viewer_link=p.name+'?'+urlencode(dict(case=key,old=r.arm if r.arm in ('left','straight','right') else 'auto',
                    task=r.arm.replace('_','/') if '_' in r.arm else 'near/left')),
                accepted_plans=r.published,physical_route_changes=len(data['hist']),initial_default_ticks=r.initial_default_ticks,
                expired_default_ticks=r.expired_default_ticks,missing_xy_frames=data['missing_frames']))
        for r in score[score.planner.isin(['old_filter','rolling'])].itertuples():
            key=f'{r.force:g}/{r.geometry}/{r.alpha:g}';task=f'{r.section}/{"left" if r.side>0 else "right"}'
            comparison=pairs[key]['task'].setdefault(task,{})
            comparison[r.planner]=dict(lane=int(r.lane),result=displayed_outcome(r),gate=bool(r.gate))
        for g in pairs.values():
            if not g['paired']:
                for task,arm in [('near/left','near_left'),('near/right','near_right'),
                                 ('far/left','far_left'),('far/right','far_right')]:
                    cem=trajectories[g['arms'][arm]]
                    completed=cem['verified'] and cem['goal'] and not cem['wall'] and cem['gates'][task]
                    endpoint='completed' if completed else ('goal reached; gate or verified jump unmet'
                                   if cem['termination']=='goal' else cem['termination'])
                    g['task'][task]={'old_filter':{'lane':g['arms']['straight'],'result':'未配对。仅展示实际直行分支'},
                                    'rolling':{'lane':cem['lane'],'result':endpoint}}
        if any(len(v['arms'])!=7 or len(v['task'])!=4 or any(len(t)!=2 for t in v['task'].values()) for v in pairs.values()):
            raise ValueError(f'Missing paired trajectories: {p.name}')
        title=f'{layout} / start {root}'
        data=dict(title=title,scene=geo[layout],pairs=pairs,trajectories=trajectories,refs=refs)
        p.write_text(template.replace('{{TITLE}}',html.escape(title)).replace('{{PAYLOAD}}',pack(data)))
        pages.append((p.name,title,len(group),len(pairs)))
    selected=representatives(decisions) if len(decisions) else pd.DataFrame()
    selected.to_csv(out/'representative_cases.csv',index=False)
    figures=save_examples(selected,geo,index,out) if len(selected) else []
    lane_df=pd.DataFrame(lane_index)
    lane_df.to_csv(out/'physical_lane_index.csv.gz',index=False)
    page_lines=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P0-B 实际行驶轨迹</title>',
        '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:64rem;margin:2rem auto;padding:0 1rem}li{margin:.45rem 0}</style>',
        '<h1>P0-B 实际与参考 XY 轨迹</h1>',
        '<p>各页按起点、接近曲线和扰动列出物理分支。可对比同一接手状态下旧三出口打分选中的路径与滚动规划，逐次切换查看实际发布的参考路径。</p>',
        f'<p>共享前缀校验未通过 {int((~prefixes.handbacks_equal).sum())} 组。这些组只能检查各自实际终点，在页面明确标记。</p>',
        '<h2>每种门条件的实际案例</h2><p>每张地图、近远与左右分开，在存在时分别取排序最前的救回与丢失合同；无胜负差异时优先展示失败。图中均使用相同接近路线。</p><ul>']
    page_lines.extend(f'<li><a href="{p}">{p}</a></li>' for p in figures)
    page_lines.append('</ul><h2>全部物理轨迹</h2><p><a href="physical_lane_index.csv.gz">下载逐轨迹索引</a> · <a href="representative_cases.csv">代表案例清单</a></p><ul>')
    page_lines.extend(f'<li><a href="{path}">{title}</a> · {n} 条物理轨迹，{m} 组前缀</li>' for path,title,n,m in pages)
    page_lines.append('</ul></html>')
    (out/'index.html').write_text('\n'.join(page_lines))
    measured=dict(physical_lanes=len(lanes),prefixes=len(prefixes),root_pages=len(pages),
         physically_accepted_plans=int((changes.kind=='accepted_plan').sum()),physical_default_restorations=int((changes.kind=='expired_to_default').sum()),static_figures=figures,
         full_coverage=not lane_df.duplicated(['layout','root','force','geometry','alpha','arm']).any(),
         displayed_actual_sampling_ms=50,reference_stride=6,coordinate_quantization_m=.001,
         unmatched_prefixes=int((~prefixes.handbacks_equal).sum()),raw_trace_paths='physical_lane_index.csv.gz',root_index='index.html',
         static_selection='Within layout, section, side, and paired outcome choose lowest root/force/geometry/alpha; prioritize joint failures if no discordant outcomes.',
         reference='Frozen entry/final published geometry plus each accepted rolling reference or expiry-to-default change at its physical change time; archive path in the inspector.')
    (out/'manifest.json').write_text(json.dumps(measured,ensure_ascii=False,indent=2)+'\n')
    return measured


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--analysis',type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    print(json.dumps(export(run,args.analysis.resolve() if args.analysis else run/'analysis'),ensure_ascii=False))


if __name__=='__main__':
    main()
