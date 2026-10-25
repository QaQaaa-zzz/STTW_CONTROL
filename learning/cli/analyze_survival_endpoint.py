"""Offline physical/task/preference reporting after fixed terminal collection."""
import argparse,json,os,time
from pathlib import Path
from sttw_control.preference_best import atomic_json,digest


def analyze(root):
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='completed' or not status['endpoint_authenticated']:
        raise ValueError('authenticated completed endpoint collection required')
    out=root/'task_analysis';out.mkdir(exist_ok=False)
    state=dict(phase='running',pid=os.getpid(),stage='read_saved_cases',physics_steps=0,optimizer_steps=0,completed_cases=0)
    def update(**kw):
        state.update(kw,checked_unix=time.time());atomic_json(out/'status.json',state)
    update()
    from prepare_short_recovery_campaign import register
    register(out/'ACTIVE_RUN.json','STTW_CONTROL 固定终点原任务与防摔联合分析',out/'status.json')
    try:
        import numpy as np
        from sttw_control.survival_endpoint_task_metrics import task_metrics,zero_relative_cases
        cfg=json.loads((root/'common_environment_config.json').read_text());plan=json.loads((root/'plan.json').read_text())
        rows=json.loads((root/'results.json').read_text());index=json.loads((root/'case_index.json').read_text())
        if len(rows)!=plan['protocol_plan']['declared_actual_episodes'] or len(index)!=len(rows):raise ValueError('incomplete physical case matrix')
        byid={r['case_id']:r for r in index}
        if len(byid)!=len(index) or set(byid)!={r['case_id'] for r in rows}:raise ValueError('mismatched actual case identities')
        sources={str(root/p):digest(root/p) for p in ['results.json','case_index.json','common_environment_config.json','authenticated_models.json']}
        atomic_json(out/'plan.json',dict(project='STTW_CONTROL',purpose=__doc__,collection=str(root),
            code_sha256={str(p.resolve()):digest(p) for p in [Path(__file__),Path('learning/src/sttw_control/survival_endpoint_task_metrics.py')]},
            bands=cfg['stage_B_baseline'],data_role='DEV',physics_steps=0,optimizer_steps=0,
            scope='Original task bands and continuous observed-prefix errors; no new adoption threshold or checkpoint selection.'))
        augmented=[];goals={}
        for row in rows:
            item=byid[row['case_id']];path=root/item['file']
            if digest(path)!=item['sha256']:raise ValueError('changed factual raw case')
            sources[str(path)]=item['sha256']
            reference=root/item['reference']
            if str(reference) not in goals:
                sources[str(reference)]=digest(reference)
                with np.load(reference) as z:goals[str(reference)]=float(z['goal'])
            with np.load(path) as z:physical={k:z[k] for k in z.files if not k.startswith('decision_')}
            result=task_metrics(physical,speed=row['speed'],goal_s=goals[str(reference)],config=cfg)
            augmented.append(dict(row,**result))
            if len(augmented)%24==0:update(completed_cases=len(augmented))
        groups={}
        for row in augmented:groups.setdefault((row['panel'],row['role'],row['alpha']),[]).append(row)
        summary=[]
        for key,group in groups.items():
            average=lambda field:float(np.mean([r[field] for r in group if r[field] is not None])) if any(r[field] is not None for r in group) else None
            summary.append(dict(panel=key[0],role=key[1],alpha=key[2],actual_episodes=len(group),
                no_contact=sum(r['outcome']=='survived' for r in group),physical_unknown=sum(r['outcome'] not in ('survived','contact') for r in group),
                task_recovered=sum(r['task_recovered'] is True for r in group),task_unknown=sum(r['task_recovered'] is None for r in group),
                performance_success=sum(r['performance_success'] is True for r in group),
                domain_exits=sum(r['domain_exit'] for r in group),mean_hold_s=average('hold_ticks')*cfg['timing']['lower_dt_s'],
                mean_task_observed_s=average('task_observed_s'),mean_task_path_rmse_m=average('task_path_rmse_m'),
                mean_task_speed_rmse_m_s=average('task_speed_rmse_m_s')))
        changes=zero_relative_cases(augmented)
        atomic_json(out/'task_results.json',augmented);atomic_json(out/'grouped_results.json',summary);atomic_json(out/'change_vs_own_zero.json',changes)
        lines=['# Fixed endpoint physical and original-task outcomes','',
            '[Actual original/physical XY and per-case diagnostics](../figures/INDEX.md)','',
            'DEV results. Original bands and final hold are diagnostic labels, not new acceptance requirements.',
            'No contact, task recovery, and working-roll performance are separate outcomes. Error metrics cover only observed task-valid prefixes; shorter failed observations are not better tracking.',
            'Zero-middle rows are actual unique deterministic episodes. They are not replicated over alpha/noise or pooled with stochastic policy runs.',
            'Four action draws are not four independent training seeds. Original raw results and authenticated model identities remain unchanged.','',
            '|Panel|Role|Alpha|Actual n|No contact|Task recovered|Task unknown|Working performance|Domain exits|Mean hold s|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for s in summary:lines.append('|'+ '|'.join(str(s[k]) for k in ['panel','role','alpha','actual_episodes','no_contact','task_recovered','task_unknown','performance_success','domain_exits'])+f'|{s["mean_hold_s"]:.3f}|')
        lines+=['','Continuous path/speed errors, actual observation lengths and final-tail coverage: grouped_results.json and task_results.json.',
            'Own zero-force changes are in change_vs_own_zero.json. An absolute policy gap alone is not called disturbance-specific robustness.',
            'The linked figure index is a separate artifact; absent/unreviewed figures mean complete visual delivery is still pending.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        atomic_json(out/'receipt.json',dict(source_sha256=sources,actual_episode_count=len(augmented),physics_steps=0,optimizer_steps=0))
        update(phase='completed',stage='original_task_and_physical_outcomes_reported',completed_cases=len(augmented))
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);analyze(p.parse_args().run.resolve())
