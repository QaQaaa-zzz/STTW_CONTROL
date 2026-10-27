"""Read existing common-state branches; never fit or evaluate a new selector.

Online snapshots use the last completed lower tick before the fork. Post-fork
counterfactual changes are retrospective mechanism diagnostics, never features.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json, digest

ROLES=('full_current','pause_neural_steer','pause_neural_both')
FIELDS=('phi','phi_dot','actual_delta','actual_forward_speed','wheel_speed_proxy',
        'path_cross_track','path_heading_error','reference','filtered_offset')


def run(root,output_name):
    root=Path(root).resolve()
    if Path(output_name).name!=output_name:raise ValueError('simple output directory required')
    out=root/output_name;out.mkdir(exist_ok=False)
    atomic_json(out/'analysis_contract.json',dict(question='Do already logged causal measurements and short-term consequences establish an actionable relation beyond existing simple feedback?',
        data_role='Outcome-inspected DEV; descriptive only, not new selector or independent validation',
        events=120,branches=360,goal_radius_m=1.,horizon_s=10.,
        online_snapshot='Last completed lower tick before privileged fork; no event metadata is an online feature',
        counterfactual_windows_lower_ticks=[20,80],existing_rate_trigger_rad_s=.8,
        no_fit=True,no_threshold_search=True,no_selected_rule_rollout=True,
        physics_steps=0,optimizer_updates=0))
    source=root/'analysis_partial_block_fix_v1/results.json'
    rows=json.loads(source.read_text());fork=json.loads((root/'physical/fork_receipt.json').read_text())
    if len(rows)!=360 or not fork['all_carry_leaves_copied'] or fork['clock_reset']:raise ValueError('complete authenticated fork cohort required')
    groups={}
    for r in rows:
        g=groups.setdefault(r['prefix_index'],{})
        if r['role'] in g:raise ValueError('duplicate branch')
        g[r['role']]=r
    if set(groups)!=set(range(120)) or any(set(g)!=set(ROLES) for g in groups.values()):raise ValueError('all120 three-way events required')
    events=[];details=[];raw_checks=[]
    for idx,g in sorted(groups.items()):
        b=g[ROLES[0]];f=int(b['fork_task_tick']);raw={}
        for role,r in g.items():
            for k in ['context','alpha','event_id','force_N','fork_task_tick','pulse_onset_task_tick','world_direction','pulse_duration_s']:
                if r[k]!=b[k]:raise ValueError('unmatched fork metadata')
            file=Path(r['raw_file'])
            if digest(file)!=r['raw_sha256']:raise ValueError('changed raw '+r['case_id'])
            with np.load(file) as z:raw[role]={k:np.asarray(z[k]) for k in z.files}
            d=raw[role];active=np.asarray(d['active_tick'],bool)
            if not r['reached_fork'] or not active[f-1] or not active[f]:raise ValueError('fork snapshot unavailable')
            np.testing.assert_array_equal(d['physical_tick'][active],np.arange(active.sum()))
            raw_checks.append(dict(case_id=r['case_id'],raw_sha256=r['raw_sha256']))
        full=raw[ROLES[0]]
        for role in ROLES[1:]:
            for k in FIELDS:np.testing.assert_array_equal(full[k][:f],raw[role][k][:f])
        features={k:float(full[k][f-1]) for k in FIELDS if full[k].ndim==1}
        features.update(reference_speed=float(full['reference'][f-1,0]),reference_steer=float(full['reference'][f-1,1]),
                        existing_relay_active_before_fork=bool(full['relay_active'][f-1]),
                        prior_rate_above_existing_trigger=bool(abs(full['phi_dot'][f-1])>.8),
                        actor_legal_before_fork=bool(full['actor_action_valid'][f-1]))
        event=dict(prefix_index=idx,case_id=b['case_id'],context=b['context'],alpha=b['alpha'],
                   force_N_metadata_not_feature=b['force_N'],fork_task_tick_metadata_not_feature=f,
                   online_features=features)
        events.append(event)
        for role in ROLES[1:]:
            r=g[role];candidate=raw[role]
            effect=dict(prefix_index=idx,role=role,context=b['context'],alpha=b['alpha'],nominal=(b['force_N']==0),
                arrival_change=int(r['minimum_upright_goal_distance_m']<=1.)-int(b['minimum_upright_goal_distance_m']<=1.),
                survival_change=int(r['physical_outcome']=='survived')-int(b['physical_outcome']=='survived'),
                original_user_total_cost_change=r['scored_cost']-b['scored_cost'],
                minimum_goal_distance_change_m=r['minimum_upright_goal_distance_m']-b['minimum_upright_goal_distance_m'],
                retrospective_windows={})
            for n in [20,80]:
                end=f+n
                available=bool(len(full['active_tick'])>=end and len(candidate['active_tick'])>=end and np.all(full['active_tick'][f:end]) and np.all(candidate['active_tick'][f:end]))
                item=dict(complete_paired_window=available)
                if available:
                    for key,fn in [('peak_abs_roll_rad',lambda d:np.max(np.abs(d['phi'][f:end]))),
                                   ('end_abs_roll_rate_rad_s',lambda d:abs(d['phi_dot'][end-1])),
                                   ('end_abs_path_error_m',lambda d:abs(d['path_cross_track'][end-1])),
                                   ('end_abs_heading_error_rad',lambda d:abs(d['path_heading_error'][end-1]))]:
                        item[key]=dict(full=float(fn(full)),candidate=float(fn(candidate)),change=float(fn(candidate)-fn(full)))
                    item['mean_reference_steer_change_rad']=float(np.mean(candidate['reference'][f:end,1]-full['reference'][f:end,1]))
                    item['mean_actual_steer_change_rad']=float(np.mean(candidate['actual_delta'][f:end]-full['actual_delta'][f:end]))
                effect['retrospective_windows'][str(n)]=item
            details.append(effect)
    summary={}
    for role in ROLES[1:]:
        rr=[d for d in details if d['role']==role];positive=[d for d in rr if d['arrival_change']>0 or d['survival_change']>0]
        summary[role]=dict(n=len(rr),disturbed_n=sum(not d['nominal'] for d in rr),
            arrival_gain=sum(d['arrival_change']>0 for d in rr),arrival_loss=sum(d['arrival_change']<0 for d in rr),
            survival_gain=sum(d['survival_change']>0 for d in rr),survival_loss=sum(d['survival_change']<0 for d in rr),
            positive_events=[dict(effect=d,online_features=events[d['prefix_index']]['online_features']) for d in positive],
            nominal_cost_by_alpha={str(a):dict(n=len(q:=[d for d in rr if d['nominal'] and d['alpha']==a]),mean_change=float(np.mean([d['original_user_total_cost_change'] for d in q]))) for a in [0.,1.]},
            existing_trigger_coverage=dict(rate_above=sum(e['online_features']['prior_rate_above_existing_trigger'] for e in events),
                relay_active=sum(e['online_features']['existing_relay_active_before_fork'] for e in events),
                warning='Coverage at privileged fork only; no switched trajectory evaluated, no binary danger estimator certified'),
            surrogate_direction={str(n):{key:dict(paired_available=sum(d['retrospective_windows'][str(n)]['complete_paired_window'] for d in rr),
                lower=sum(d['retrospective_windows'][str(n)]['complete_paired_window'] and d['retrospective_windows'][str(n)][key]['change']< -1e-6 for d in rr),
                higher=sum(d['retrospective_windows'][str(n)]['complete_paired_window'] and d['retrospective_windows'][str(n)][key]['change']>1e-6 for d in rr)) for key in ['peak_abs_roll_rad','end_abs_roll_rate_rad_s']} for n in [20,80]})
    atomic_json(out/'events.json',events);atomic_json(out/'effects.json',details);atomic_json(out/'summary.json',summary)
    atomic_json(out/'receipt.json',dict(source=str(source),source_sha256=digest(source),raw_checks=raw_checks,
        shared_logged_prefix_fields=list(FIELDS),feature_last_completed_tick=True,physics_steps=0,optimizer_updates=0,
        no_new_controller=True,no_performance_claim_for_a_selector=True))
    with (out/'online_features.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['prefix_index','case_id',*events[0]['online_features']]);writer.writeheader()
        for e in events:writer.writerow(dict(prefix_index=e['prefix_index'],case_id=e['case_id'],**e['online_features']))
    print(json.dumps({role:{k:v for k,v in d.items() if k not in ['positive_events','surrogate_direction']} for role,d in summary.items()},ensure_ascii=False))


def render(root,output_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    root=Path(root).resolve();out=root/output_name
    events=json.loads((out/'events.json').read_text());details=json.loads((out/'effects.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for row,role in enumerate(ROLES[1:]):
        rr=[d for d in details if d['role']==role]
        for col,n in enumerate([20,80]):
            ax=axes[row,col];count=0
            for d in rr:
                w=d['retrospective_windows'][str(n)]
                if not w['complete_paired_window']:continue
                count+=1;x=abs(events[d['prefix_index']]['online_features']['phi_dot']);y=w['peak_abs_roll_rad']['change']
                label=None
                if d['survival_change']>0:color,marker,size,label='#D55E00','^',100,'Survival gain (one event)'
                elif d['arrival_change']>0:color,marker,size,label='#0072B2','*',120,'Arrival gain (one other event)'
                else:color,marker,size=('#777777','o',20) if d['nominal'] else ('#222222','s',15)
                ax.scatter(x,y,c=color,marker=marker,s=size,alpha=.8,label=label,zorder=3 if label else 2)
            ax.axhline(0,color='#555555',lw=.8);ax.axvline(.8,color='#777777',ls=':',lw=1)
            ax.set(xlabel='Pre-fork |roll rate| (rad/s)',ylabel='Pause minus full: max sampled |roll| (rad)',title=f'{role}: first {n*.005:g}s; {count}/120 complete pairs')
            ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('All common-state events: short-term roll changes are retrospective, not an online score\nDotted line: existing 0.8 rad/s trigger; nominal circles, disturbed squares; two masks are paired')
    fig.savefig(out/'all_events_short_term_changes.png',dpi=160);plt.close(fig)
    summary=json.loads((out/'summary.json').read_text());rows=json.loads((root/'analysis_partial_block_fix_v1/results.json').read_text())
    by={(r['prefix_index'],r['role']):r for r in rows}
    positives=summary['pause_neural_steer']['positive_events'];plan=json.loads((root/'plan.json').read_text())
    fig,axes=plt.subplots(len(positives),4,figsize=(17,8),layout='constrained')
    for row,positive in enumerate(positives):
        effect=positive['effect'];idx=effect['prefix_index'];context=effect['context']
        r=by[idx,'full_current'];f=r['fork_task_tick']
        with np.load(plan['contexts'][context]['path']) as z:path={k:z[k] for k in z.files}
        goal=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in [0,1]])
        ax=axes[row,0];mask=path['s']<=path['goal'];ax.plot(*path['xy'][mask].T,c='black',ls=':',label='Original path')
        ax.add_patch(Circle(goal,1.,fill=False,ec='black',lw=.8));ax.scatter(*goal,c='black',marker='*',s=25)
        for role,color,style in [('full_current','#333333','-'),('pause_neural_steer','#D55E00','--')]:
            rr=by[idx,role];file=Path(rr['raw_file'])
            if digest(file)!=rr['raw_sha256']:raise ValueError('changed plotted raw')
            with np.load(file) as z:d={k:z[k] for k in z.files}
            active=d['active_tick'].astype(bool);t=(d['physical_tick']+1-f)*.005
            xy=d['actual_xy'][active];ax.plot(*xy.T,c=color,ls=style,label=role)
            ax.scatter(*xy[-1],c=color,marker='x' if rr['physical_outcome']!='survived' else 'o',s=35)
            ax.scatter(*d['actual_xy'][f-1],c=color,marker='s',s=20)
            window=active&(t>=-.2)&(t<=1.6)
            axes[row,1].plot(t[window],d['phi'][window],c=color,ls=style,label=role)
            axes[row,2].plot(t[window],d['phi_dot'][window],c=color,ls=style,label=role)
            axes[row,3].plot(t[window],d['reference'][window,1],c=color,ls=':',label=role+' reference')
            axes[row,3].plot(t[window],d['actual_delta'][window],c=color,ls=style,label=role+' actual')
        ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=('Survival gain; no arrival' if effect['survival_change'] else 'Arrival gain; both survived'))
        ax.legend(fontsize=6)
        for col,label in [(1,'Roll (rad)'),(2,'Roll rate (rad/s)'),(3,'Steering (rad)')]:
            a=axes[row,col];a.axvspan(-.1,0,color='#BBBBBB',alpha=.3);a.axvspan(0,.4,color='#E69F00',alpha=.12);a.axvline(0,c='#555555',lw=.8)
            a.set(xlabel='Physical time from fork (s)',ylabel=label,title=f'{context.split("__")[0]}, alpha1, {r["force_N"]:+g}N');a.grid(alpha=.15)
        axes[row,3].legend(fontsize=6)
    fig.suptitle('Two distinct common-state witnesses: full 10s XY, local unsmoothed responses\nSquare: privileged fork; orange band: 0.4s neural steer pause; gray band: preceding pulse; circle=10s end, cross=contact')
    fig.savefig(out/'two_witnesses_xy_and_response.png',dpi=160);plt.close(fig)
    atomic_json(out/'figure_provenance.json',dict(scope='Provisional internal research figures; journal requirements unknown',all_event_points=120,distinct_positive_events=2,
        masks_not_independent_replicates=True,missing_04s_pairs=3,smoothing='none',time_definition='Post-tick physical time=(physical_tick+1-fork_tick)*0.005',
        xy_scope='Two specified witnesses, full 10s or real contact endpoint; all-event XY remains inherited',physics_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output-name',default='observability_audit_v1');p.add_argument('--render-only',action='store_true');args=p.parse_args();render(args.root,args.output_name) if args.render_only else run(args.root,args.output_name)
