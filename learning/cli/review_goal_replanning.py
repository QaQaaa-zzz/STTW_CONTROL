"""Compare the completed fixed-route and periodic-replanning176 panels.

Original-route distance below is closest-polyline geometry, not the historical
bounded-progress V3 cost. All physical traces and score aliases remain distinct.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.goal_endpoint_analysis import _read_panel,_validate_h75_comparison
from sttw_control.preference_best import digest


def original_distance(xy,path,*,chunk_size=128):
    """Exact closest sampled-polyline segment distance, with bounded memory."""
    xy=np.asarray(xy,float);points=np.asarray(path['xy'],float)
    if xy.ndim!=2 or xy.shape[1]!=2 or points.ndim!=2 or points.shape[1]!=2 or len(points)<2:
        raise ValueError('XY rows and at least two route vertices required')
    if chunk_size<=0 or not np.isfinite(xy).all() or not np.isfinite(points).all():
        raise ValueError('finite coordinates and positive chunk size required')
    a=points[:-1];v=points[1:]-a;den=np.sum(v*v,axis=1)
    result=np.empty(len(xy))
    for start in range(0,len(xy),chunk_size):
        delta=xy[start:start+chunk_size,None,:]-a
        t=np.divide(np.sum(delta*v,axis=2),den,out=np.zeros(delta.shape[:2]),where=den>0)
        difference=delta-np.clip(t,0,1)[...,None]*v
        result[start:start+chunk_size]=np.sqrt(np.min(np.sum(difference*difference,axis=2),axis=1))
    return result


# Only these declared integration/analysis files may differ across the panels.
_SOURCE_EXEMPTIONS=frozenset(('learning/src/sttw_control/goal_endpoint_run.py',
    'learning/src/sttw_control/goal_endpoint_analysis.py',
    'learning/src/sttw_control/goal_endpoint_replanning.py',
    'learning/cli/evaluate_goal_endpoint.py',
    'learning/cli/review_goal_replanning.py'))


def validate_shared_sources(fixed,replanned):
    def shared(plan):
        sources=plan.get('sources')
        if not sources:raise ValueError('sealed runtime source inventory required')
        return {name:sha for name,sha in sources.items()
                if not any(name==allowed or name.endswith('/'+allowed) for allowed in _SOURCE_EXEMPTIONS)}
    if shared(fixed)!=shared(replanned):
        raise ValueError('fixed/replanned shared runtime dependency differs')


def load_pair(main,h75):
    p,index,paths,cfg=_read_panel(main,128);q,other,op,oc=_read_panel(h75,48)
    _validate_h75_comparison(p,q,paths,op)
    result={}
    for root,items in [(main,index),(h75,other)]:
        for item in items:
            if digest(root/item['file'])!=item['sha256']:raise ValueError('original trajectory changed: '+item['file'])
            with np.load(root/item['file']) as z:data={k:z[k] for k in z.files}
            key=item['case']['case_id'];result[key]=(root,item,data)
    return p,paths,cfg,result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixed',type=Path,required=True);ap.add_argument('--fixed-h75',type=Path,required=True)
    ap.add_argument('--replan',type=Path,required=True);ap.add_argument('--replan-h75',type=Path,required=True)
    a=ap.parse_args();p,paths,cfg,old=load_pair(a.fixed,a.fixed_h75);q,newpaths,ncfg,new=load_pair(a.replan,a.replan_h75)
    if p.get('replanning') or not q.get('replanning') or p['cases']!=q['cases'] or cfg!=ncfg:raise ValueError('only declared replanning comparison permitted')
    validate_shared_sources(p,q)
    if old.keys()!=new.keys():raise ValueError('complete same-case176 required')
    for key,sha in p['inputs'].items():
        if q['inputs'].get(key)!=sha:raise ValueError('fixed scientific input differs '+key)
    out=a.replan/'paired_replanning_review';out.mkdir(exist_ok=False)
    scores={}
    for label,root in [('fixed',a.fixed),('replan',a.replan)]:
        r=json.loads((root/'analysis_with_h75/results.json').read_text())
        scores[label]={x['case_id']:x for x in r['score_views']}
    rows=[];groups={};coordinate_data={};max_prefix=0.
    for index,key in enumerate(old):
        _,oi,od=old[key];nr,ni,nd=new[key];c=oi['case']
        assert oi['case']==ni['case'] and np.allclose(oi['original_goal_xy'],ni['original_goal_xy'],atol=1e-6,rtol=0)
        pathkey=c['suite']+'/'+c['context']
        with np.load(a.fixed/paths[pathkey]['file']) as z:original={k:z[k] for k in z.files}
        with np.load(a.replan/newpaths[pathkey]['file']) as z:
            if any(not np.array_equal(z[k],original[k]) for k in original):raise ValueError('original route changed')
        prefix=float(np.max(np.abs(od['actual_xy'][:200]-nd['actual_xy'][:200])));max_prefix=max(max_prefix,prefix)
        row=dict(case_id=key,role=c['role_name'],suite=c['suite'],context=c['context'],alpha=c['controller_alpha'],force_N=c['force_N'],onset=c['pulse_onset_task_tick'],prefix1s_xy_max_difference_m=prefix)
        for label,data in [('fixed',od),('replan',nd)]:
            s=scores[label][key];mask=data['active_tick'];dist=original_distance(data['actual_xy'][mask],original)
            row[label+'_goal']=s['goal'];row[label+'_survival']=s['survival'];row[label+'_goal_time_s']=s['goal_time_s']
            row[label+'_original_polyline_distance_rmse_m']=float(np.sqrt(np.mean(dist**2)))
            row[label+'_speed_rmse_m_s']=s['speed_rmse_m_s'];row[label+'_observed_s']=float(mask.sum()*.005)
            row[label+'_closest_prefatal_goal_m']=s['minimum_prefatal_goal_distance_m'] if 'minimum_prefatal_goal_distance_m' in s else s.get('minimum_prefatal_goal_distance')
            coordinate_data[f'case{index:03d}_{label}_xy']=data['actual_xy'][mask]
        rows.append(row)
        g=tuple(c[k] for k in ('suite','context','force_N','pulse_onset_task_tick'))
        groups.setdefault(g,[]).append((ni,nd,nr))
    np.savez_compressed(out/'actual_coordinates.npz',**coordinate_data)
    with (out/'paired_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for role in sorted({r['role'] for r in rows}):
        rr=[r for r in rows if r['role']==role]
        summary.append(dict(role=role,n=len(rr),goal_gain=sum(r['fixed_goal']=='not_arrived' and r['replan_goal']=='arrived' for r in rr),
            goal_loss=sum(r['fixed_goal']=='arrived' and r['replan_goal']=='not_arrived' for r in rr),
            survival_gain=sum(r['fixed_survival']=='contact' and r['replan_survival']=='survived' for r in rr),
            survival_loss=sum(r['fixed_survival']=='survived' and r['replan_survival']=='contact' for r in rr)))
    colors=['#0072B2','#D55E00','#333333','#009E73','#AA4499'];styles={0.:'-',.5:'--',1.:'-.'};figures=[]
    for g,items in groups.items():
        with plt.rc_context({'font.size':8}):
            fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
            pathkey=g[0]+'/'+g[1]
            with np.load(a.replan/newpaths[pathkey]['file']) as z:original={k:z[k] for k in z.files}
            show=original['s']<=float(original['goal'])+5
            ax[0,0].plot(*original['xy'][show].T,':',color='grey',label='Original route')
            goal=items[0][0]['original_goal_xy'];ax[0,0].plot(*goal,'k*',label='Original goal')
            for item,data,root in items:
                c=item['case'];mask=data['active_tick'];role=c['role'];alpha=c['controller_alpha'];color=colors[role];style=styles[alpha]
                label=c['role_name']+(f' a{alpha:g}' if role in (0,1,4) else '')
                xy=data['actual_xy'][mask];t=np.arange(1,len(xy)+1)*.005
                ax[0,0].plot(*xy.T,color=color,ls=style,label=label)
                ax[0,0].plot(*xy[-1],marker='x' if np.any(data['contact_failure'][mask]) else 'o',color=color,ms=4)
                ax[0,1].plot(t,original_distance(xy,original),color=color,ls=style)
                ax[1,0].plot(t,data['actual_forward_speed'][mask]-data['requested_speed'][mask],color=color,ls=style)
                ax[1,1].plot(t,data['path_cross_track'][mask],color=color,ls=style)
            ax[0,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Periodic replanning: actual XY; original goal unchanged');ax[0,0].set_aspect('equal',adjustable='datalim');ax[0,0].legend(fontsize=5)
            ax[0,1].set(xlabel='Time (s)',ylabel='Distance (m)',title='Distance to ORIGINAL polyline (descriptive)')
            ax[1,0].set(xlabel='Time (s)',ylabel='Speed error (m/s)',title='Actual minus original requested speed')
            ax[1,1].set(xlabel='Time (s)',ylabel='Signed cross-track error (m)',title='EXECUTION-path error; resets geometry at replans')
            for axis in [ax[0,1],ax[1,0],ax[1,1]]:
                if g[2]:axis.axvspan(g[3]*.005,g[3]*.005+.1,color='grey',alpha=.2)
                axis.grid(alpha=.2)
            fig.suptitle(f'{g}; H10; fixed0064 / parent64 / zero / support / H75u77; related DEV')
            name=f'{g[0]}_{g[1]}_F{g[2]:+g}_t{g[3]}_original_vs_execution.png';fig.savefig(out/name,dpi=150);plt.close(fig);figures.append(name)
    report=dict(summary=summary,max_before_first_plan_xy_difference_m=max_prefix,physical_pairs=len(rows),figures=figures,
        interpretation='Same original goals/models/cases/physics and H10; new periodic planner shared by all methods. Original-polyline distance is a descriptive nearest-segment metric, not original bounded-progress V3 cost. No independent trials or safety certification.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    lines=['# 共同重规划与固定路径：176条逐物理轨迹配对','',report['interpretation'],'','|方法|物理配对|到达新增|到达丢失|不触地新增|不触地丢失|','|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('|'+ '|'.join(str(r[k]) for k in ('role','n','goal_gain','goal_loss','survival_gain','survival_loss'))+'|')
    lines+=['',f'首次1s规划前XY最大差{max_prefix:.8g}m。全部实际坐标见actual_coordinates.npz；逐例paired_cases.csv。未知/观察时长不能当零误差。',
        '执行路径V3费用不能直接跨固定/重规划比较为原用户效用；原路径几何偏离、请求速度误差及原目标G/S分别保留。上图没有把重规划后近零执行误差冒作原路径误差。','']
    lines += [f'- [{name}]({name})' for name in figures]
    (out/'REVIEW.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
