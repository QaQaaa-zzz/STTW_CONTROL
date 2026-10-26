"""Post-hoc finite-library reward/strict-task ordering; zero new physical steps."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.live_endpoint_analysis import paired_endpoint_report,ROLES
from prepare_short_recovery_campaign import register

SOURCE=Path('runs/matched_live_ppo_v1_fixed64_evaluation').resolve()
OUT=Path('runs/objective_ordering_audit_v1').resolve()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),physics_steps=0,optimizer_updates=0)
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(OUT/'status.json',status)
    update();register(OUT/'ACTIVE_RUN.json','STTW_CONTROL 历史200例任务代价与保持条件排序审计',OUT/'status.json')
    try:
        directory=SOURCE/'analysis'
        source_status=json.loads((directory/'status.json').read_text())
        assert source_status['phase']=='completed' and source_status['completed_cases']==200
        rows=json.loads((directory/'results.json').read_text());assert len(rows)==200
        original=paired_endpoint_report(rows)
        stored=json.loads((directory/'summary.json').read_text())['all']
        assert original['role_counts']==stored['role_counts']
        with (SOURCE/'left/common_environment_config.json').open() as stream:cfg=json.load(stream)
        band=cfg['stage_B_baseline'];dt=cfg['timing']['lower_dt_s']
        groups={}
        for row in rows:groups.setdefault((row['context'],row['alpha'],row['event_id']),[]).append(row)
        assert len(groups)==50 and all(len(g)==4 and {r['role'] for r in g}==set(ROLES) for g in groups.values())
        safe=lambda r:r['category'] in ['safe_task','safe_task_failed']
        summaries={};details=[]
        for label,selection in [('all',list(groups.values())),('nominal',[g for g in groups.values() if not g[0]['force_N']]),('disturbed',[g for g in groups.values() if g[0]['force_N']])]:
            tally=dict(groups=len(selection),unknown_groups=0,groups_with_surviving_candidate=0,
                groups_with_safe_task_candidate=0,least_cost_has_safe_task=0,
                least_cost_has_only_task_failures=0,mixed_task_tie=0)
            for g in selection:
                if any(r['category']=='unknown' for r in g):tally['unknown_groups']+=1;continue
                available=[r for r in g if safe(r)]
                if not available:continue
                tally['groups_with_surviving_candidate']+=1
                successful=[r for r in available if r['category']=='safe_task']
                if not successful:continue
                tally['groups_with_safe_task_candidate']+=1
                minimum=min(r['scored_cost'] for r in available)
                winners=[r for r in available if abs(r['scored_cost']-minimum)<=1e-6]
                has_success=any(r['category']=='safe_task' for r in winners)
                has_failure=any(r['category']=='safe_task_failed' for r in winners)
                if has_success and has_failure:tally['mixed_task_tie']+=1
                elif has_success:tally['least_cost_has_safe_task']+=1
                else:tally['least_cost_has_only_task_failures']+=1
                if label=='all':
                    best_task=min(successful,key=lambda r:r['scored_cost'])
                    details.append(dict(context=g[0]['context'],alpha=g[0]['alpha'],event_id=g[0]['event_id'],
                        force_N=g[0]['force_N'],least_cost=minimum,least_cost_roles=[r['role'] for r in winners],
                        least_cost_categories=[r['category'] for r in winners],best_task_role=best_task['role'],
                        best_task_cost=best_task['scored_cost'],best_task_extra_cost=best_task['scored_cost']-minimum))
            summaries[label]=tally
        paired={}
        for target,base in [('trained_plain','parent_plain'),('trained_support','parent_support'),('parent_support','parent_plain'),('trained_plain','parent_support')]:
            t=dict(known_both_survived=0,lower_cost=0,lower_cost_task_loss=0,task_loss_cases=[])
            for group in groups.values():
                byrole={r['role']:r for r in group};a,b=byrole[target],byrole[base]
                if not (safe(a) and safe(b)):continue
                t['known_both_survived']+=1
                if a['scored_cost']<b['scored_cost']-1e-6:
                    t['lower_cost']+=1
                    if a['category']=='safe_task_failed' and b['category']=='safe_task':
                        t['lower_cost_task_loss']+=1;t['task_loss_cases'].append(a['case_id'])
            paired[target+'_vs_'+base]=t
        failures=[r for r in rows if r['category']=='safe_task_failed']
        assert all(r['complete_task_observation'] and r['tail_observed_ticks']==round(band['hold_s']/dt) for r in failures)
        limits={'path':('tail_path_max_m',band['final_y_error_m']),
                'heading':('tail_heading_error_max_rad',band['final_path_heading_error_rad']),
                'speed':('tail_speed_error_max_m_s',band['final_speed_error_m_s'])}
        late={}
        for name,group in [('all',failures),('nominal',[r for r in failures if not r['force_N']]),('disturbed',[r for r in failures if r['force_N']])]:
            item=dict(cases=len(group),goal_not_crossed=sum(not r['goal_crossed_while_task_valid'] for r in group))
            for key,(field,limit) in limits.items():
                values=np.asarray([r[field] for r in group],float);assert np.isfinite(values).all()
                item[key]=dict(original_band=limit,violations=int(np.sum(values>limit)),
                    minimum=float(values.min()) if len(values) else None,median=float(np.median(values)) if len(values) else None,
                    p90=float(np.quantile(values,.9)) if len(values) else None,maximum=float(values.max()) if len(values) else None)
            late[name]=item
        result=dict(source=str(SOURCE),data_role='posthoc old fixed200 DEV; not current three-arm training or final300',
            source_files_sha256={str(p):digest(p) for p in [directory/'results.json',directory/'summary.json',directory/'status.json',SOURCE/'left/common_environment_config.json',Path(__file__)]},
            source_level='Authenticated derived result tables; raw trajectories were audited upstream, not re-executed or individually rehashed in this audit',
            role_counts=original['role_counts'],ordering=summaries,ordering_details=details,
            paired_surviving_cost=paired,failed_surviving_tail=late,numerical_tie_tolerance=1e-6,
            physics_steps=0,optimizer_updates=0,
            limitations='Finite existing library, same-event originalalpha only, excludes unknown groups from ordering; no policy optimality or reward-causality claim; strict task labels unchanged')
        atomic_json(OUT/'results.json',result)
        update(phase='completed',checked_cases=200,stage='derived_ordering_and_continuous_tail_summary')
        print(json.dumps(dict(ordering=summaries,paired=paired,tails=late),ensure_ascii=False,indent=2))
    except BaseException as exc:
        (OUT/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


if __name__=='__main__':main()
