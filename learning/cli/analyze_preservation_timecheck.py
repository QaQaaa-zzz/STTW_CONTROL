"""Authenticate all unique traces then expand declared logical sharing, no fit."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.reference_preservation_confirmation import read_inputs,costs,audit_pulse
from sttw_control.reference_bias_runtime import score_raw,encode_objective
from sttw_control.reference_bias_optimization import bias_targets
from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics


def add_radius_sensitivity(scored,raw,path):
    """Predeclared descriptive radius; original1m decision stays unchanged."""
    point=[np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in range(2)]
    scored['goals']['1.25']=goal_arrival_metrics(raw,goal_xy=point,radius_m=1.25,dt=.005,horizon_s=10.)


def expand_logical_rows(audited,logical):
    # source_role belongs to the physical feedback contract; the outer role
    # names the old candidate chosen by a selector. Preserve both meanings.
    return [{**audited[r['physical_index']],**r['case'],
             'physical_index':r['physical_index'],'chosen_source_role':r['source_role']}
            for r in logical]


def analyze(parent):
    parent=parent.resolve();out,protocol,logical,physical=read_inputs(parent)
    state=json.loads((out/'status.json').read_text())
    if state['phase']!='completed' or state['completed_physical_cases']!=240:raise ValueError('complete fixed run required')
    target=out/'analysis';target.mkdir(exist_ok=False)
    plan=json.loads((parent/'plan.json').read_text());cfg=json.loads((parent/'common_environment_config.json').read_text());paths={}
    for ctx,c in plan['contexts'].items():
        if digest(c['path'])!=c['path_sha256']:raise ValueError('path changed')
        with np.load(c['path']) as z:paths[ctx]={k:z[k] for k in z.files}
    measured=json.loads((out/'physical_results.json').read_text());audited=[]
    if len(measured)!=240:raise ValueError('all240 raw required')
    for row,item in zip(measured,physical):
        if row['case_id']!=item['case']['case_id'] or row['vector']!=item['vector']:raise ValueError('misaligned physical result')
        if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('raw changed')
        with np.load(row['raw_file']) as z:raw={k:z[k] for k in z.files}
        ctx=row['context'];q=score_raw(raw,item['case'],cfg,paths[ctx],targets=bias_targets(item['vector']),positions=plan['contexts'][ctx]['bias_positions_m'],feedback_spec=plan['feedback_spec'])
        add_radius_sensitivity(q,raw,paths[ctx])
        if abs(q['scored_cost']-row['scored_cost'])>1e-8:raise ValueError('reward changed')
        q.update(pulse_audit=audit_pulse(raw,item['case']))
        audited.append(dict(encode_objective(q),vector=item['vector'],raw_file=row['raw_file'],raw_sha256=row['raw_sha256'],physical_id=item['physical_id']))
    if sum(r['physical_ticks'] for r in audited)!=state['active_lower_ticks']:raise ValueError('physical cost mismatch')
    for field,value in costs(out).items():
        if state[field]!=value:raise ValueError('allpartial/batch cost mismatch')
    rows=expand_logical_rows(audited,logical)
    original=json.loads((parent/'production/analysis/results.json').read_text())
    for c in json.loads((out/'cached_nominal.json').read_text()):
        with np.load(c['raw_file']) as z:raw={k:z[k] for k in z.files}
        source=next(r for r in original if r['context']==c['context'] and r['alpha']==c['alpha'] and r['role']==c['chosen_role'] and not r['force_N'])
        if source['raw_sha256']!=c['raw_sha256']:raise ValueError('nominal cache identity mismatch')
        case=dict(source,role=c['method'],case_id=c['context']+f'/a{c["alpha"]}/nominal/'+c['method'],event_id='nominal',force_N=0.)
        q=score_raw(raw,case,cfg,paths[c['context']])
        add_radius_sensitivity(q,raw,paths[c['context']])
        if abs(q['scored_cost']-c['cost'])>1e-8:raise ValueError('cached nominal reward changed')
        rows.append(dict(encode_objective(q),raw_file=c['raw_file'],raw_sha256=c['raw_sha256'],inherited_nominal=True))
    methods=protocol['methods'];summary={};groups={}
    for r in rows:
        if r['force_N']:groups.setdefault((r['context'],r['alpha'],r['event_id']),{})[r['role']]=r
    if len(groups)!=96 or any(set(g)!=set(methods) for g in groups.values()):raise ValueError('fourmethods on all96 events required')
    for method in methods:
        rr=[g[method] for g in groups.values()]
        summary[method]=dict(n=96,arrived={str(t):sum(r['goals'][str(t)]['goal_arrived'] for r in rr) for t in [.5,.75,1.,1.25]},survived=sum(r['physical_outcome']=='survived' for r in rr),
            nominal_user_cost={str(a):float(np.mean([r['scored_cost'] for r in rows if not r['force_N'] and r['role']==method and r['alpha']==a])) for a in [0.,.5,1.]},
            arrival_gain_vs_full=sum(g[method]['goals']['1.0']['goal_arrived'] and not g['full']['goals']['1.0']['goal_arrived'] for g in groups.values()),
            arrival_loss_vs_full=sum(g['full']['goals']['1.0']['goal_arrived'] and not g[method]['goals']['1.0']['goal_arrived'] for g in groups.values()),
            survival_gain_vs_full=sum(g[method]['physical_outcome']=='survived' and g['full']['physical_outcome']!='survived' for g in groups.values()),
            survival_loss_vs_full=sum(g['full']['physical_outcome']=='survived' and g[method]['physical_outcome']!='survived' for g in groups.values()))
    strata={}
    for alpha in [0.,.5,1.]:
        for phase in ['early','late']:
            selected=[g for g in groups.values() if g['full']['alpha']==alpha and g['full']['event_id'].split('_')[0]==phase]
            strata[f'alpha{alpha}/{phase}']={m:dict(n=len(selected),arrived=sum(g[m]['goals']['1.0']['goal_arrived'] for g in selected),survived=sum(g[m]['physical_outcome']=='survived' for g in selected),scored_cost_mean=float(np.mean([g[m]['scored_cost'] for g in selected]))) for m in methods}
    comparisons={}
    for candidate,baseline in [('context_choice','full'),('context_choice','geometry_alpha'),('context_choice','alpha_only'),('geometry_alpha','alpha_only')]:
        delta=[g[candidate]['goals']['1.0']['minimum_upright_goal_distance_m']-g[baseline]['goals']['1.0']['minimum_upright_goal_distance_m'] for g in groups.values()]
        common=[g for g in groups.values() if all(g[m]['physical_outcome']=='survived' for m in [candidate,baseline])]
        comparisons[candidate+'_vs_'+baseline]=dict(n=96,arrival_gain=sum(g[candidate]['goals']['1.0']['goal_arrived'] and not g[baseline]['goals']['1.0']['goal_arrived'] for g in groups.values()),arrival_loss=sum(g[baseline]['goals']['1.0']['goal_arrived'] and not g[candidate]['goals']['1.0']['goal_arrived'] for g in groups.values()),survival_gain=sum(g[candidate]['physical_outcome']=='survived' and g[baseline]['physical_outcome']!='survived' for g in groups.values()),survival_loss=sum(g[baseline]['physical_outcome']=='survived' and g[candidate]['physical_outcome']!='survived' for g in groups.values()),minimum_goal_distance_change_median=float(np.median(delta)),minimum_goal_distance_change_mean=float(np.mean(delta)),common_survivor_n=len(common),common_survivor_cost_change_mean=float(np.mean([g[candidate]['scored_cost']-g[baseline]['scored_cost'] for g in common])) if common else None)
    atomic_json(target/'results.json',rows);atomic_json(target/'summary.json',dict(methods=summary,by_original_alpha_and_phase=strata,paired=comparisons,cost=state,inherited_parent_lower=3330433,total_project_new_lower=state['active_lower_ticks']+3330433,
        scope='Fixed old-data maps on new out-of-range onset times. Same old states/geometries/force ranges; not independent TEST or novel selection algorithm. Nominal costs inherited. Role-choicealpha-only still uses context-specific candidate trajectories.',visual_review_complete=False))
    render(target,rows,paths,methods)
    lines=['# 固定选择的范围外新时刻确认','', '[全部无扰XY（继承）](xy_nominal_+0.png)','', '|方法|新时刻事件|1m到达|全窗存活|','|---|---:|---:|---:|']
    for m,q in summary.items():lines.append(f'|{m}|96|{q["arrived"]["1.0"]}|{q["survived"]}|')
    for phase in ['early','late']:
        for force in [-16.,16.,-32.,32.]:lines.append(f'\n[全部XY {phase}/{force:g}N](xy_{phase}_{force:+g}.png)')
    lines+=['','240唯一物理/384受扰逻辑；48逻辑无扰来自30旧raw，未当新仿真/独立样本。旧映射已提前冻结，未用本轮重选。全部损失保留，无零损失硬门槛。图生成不等于目视；奖励/跟踪诊断另由统一诊断器生成。']
    (target/'README.md').write_text('\n'.join(lines)+'\n');atomic_json(target/'status.json',dict(phase='completed',new_physics_steps=0,physical_traces=240,logical_rows=432,xy_figures=9,visual_review_complete=False))
    print(json.dumps(summary,ensure_ascii=False))


def render(out,rows,paths,methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    colors=['#222222','#0072B2','#D55E00','#009E73'];styles=['-','-','--',':']
    for phase,force in [('nominal',0.)]+[(p,f) for p in ['early','late'] for f in [-16.,16.,-32.,32.]]:
        fig,axes=plt.subplots(4,3,figsize=(16,15),layout='constrained')
        for i,(ctx,path) in enumerate(paths.items()):
            mask=path['s']<=float(path['goal']);point=[np.interp(float(path['goal']),path['s'],path['xy'][:,j]) for j in range(2)]
            for j,alpha in enumerate([0.,.5,1.]):
                ax=axes[i,j];ax.plot(path['xy'][mask,0],path['xy'][mask,1],ls=':',color='gray');ax.add_patch(Circle(point,1.,fill=False,ec='gray',lw=.6));ax.scatter(*point,c='black',marker='*',s=20)
                rr=[r for r in rows if r['context']==ctx and r['alpha']==alpha and r['force_N']==force and r['event_id'].split('_')[0]==phase]
                if len(rr)!=4:raise ValueError('allfour lines per XY panel required')
                for m,c,style in zip(methods,colors,styles):
                    r=next(r for r in rr if r['role']==m)
                    with np.load(r['raw_file']) as z:xy=z['actual_xy'][z['active_tick'].astype(bool)]
                    ax.plot(xy[:,0],xy[:,1],color=c,ls=style,lw=1);ax.scatter(*xy[-1],color=c,marker='x' if r['physical_outcome']=='contact' else 'o',s=14)
                ax.set(title=ctx+f' / alpha{alpha}',xlabel='World X (m)',ylabel='World Y (m)');ax.title.set_fontsize(7);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        fig.legend(handles=[Line2D([],[],c=c,ls=s,label=m) for m,c,s in zip(methods,colors,styles)],loc='outside lower center',ncol=4,fontsize=8)
        fig.suptitle(f'Fixed old reference choices / {phase} / {force:g}N,0.1s\nOriginal worldgoal1m; actual contact endpoints; same ancestors/geometries; shared traces disclosed',fontsize=11)
        fig.savefig(out/f'xy_{phase}_{force:+g}.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);analyze(p.parse_args().parent)
