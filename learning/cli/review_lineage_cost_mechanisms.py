"""All120 saved traces: cost components and retrospective common-clock masks.

No changed reward, controller, threshold selection, physical rollout or fit.
Masks formed from both observed endpoints are descriptive, not interventions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sttw_control.path_command_reward import configured_cost_keys
from sttw_control.preference_trace_rescore import rescore_trace_alpha


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def authority_masks(reference,nominal,config):
    """At-bound telemetry is NOT proof clipping was requested or active."""
    c=config['residual_interface']; eps=1e-6
    at_current=(np.isclose(reference[:,0],c['speed_reference_bounds_m_s'][0],atol=eps,rtol=0)|
        np.isclose(reference[:,0],c['speed_reference_bounds_m_s'][1],atol=eps,rtol=0)|
        np.isclose(abs(reference[:,1]),c['steer_reference_bound_rad'],atol=eps,rtol=0))
    # Conservative common historical reference box, not native controller replay.
    within_old=(reference[:,0]>=1.5-eps)&(reference[:,0]<=3.+eps)&(abs(reference[:,1])<=.35+eps)&(abs(nominal[:,1])<=.3+eps)
    return at_current,within_old


def common_masks(a,b,first_times,config):
    if not np.array_equal(a['physical_tick'],b['physical_tick']):raise ValueError('matched physical clocks required')
    active=a['active_tick'].astype(bool)&b['active_tick'].astype(bool)
    times=[t for t in first_times if t is not None]
    cutoff=round(min(times)/.005)-1 if times else len(active)-1
    pre=active&(a['physical_tick']<=cutoff);post=active&~pre
    ca,old_a=authority_masks(a['reference'],a['nominal_reference'],config)
    cb,old_b=authority_masks(b['reference'],b['nominal_reference'],config)
    legal=active&a['actor_action_valid'].astype(bool)&b['actor_action_valid'].astype(bool)
    chi_a=a['chi']>1e-7;chi_b=b['chi']>1e-7
    phase=dict(full=active,through_first_goal_entry=pre,after_first_goal_entry=post,
        both_chi_active=active&chi_a&chi_b,both_chi_zero=active&~chi_a&~chi_b,
        mixed_chi=active&(chi_a!=chi_b),
        prefix_both_legal_no_current_bound=pre&legal&~ca&~cb,
        prefix_both_legal_in_old_reference_box=pre&legal&old_a&old_b,
        prefix_both_legal_no_current_bound_both_chi_active=pre&legal&~ca&~cb&chi_a&chi_b)
    if not np.array_equal(pre.astype(int)+post.astype(int),active.astype(int)):raise ValueError('goal split must partition clock')
    if not np.array_equal(phase['both_chi_active'].astype(int)+phase['both_chi_zero'].astype(int)+phase['mixed_chi'].astype(int),active.astype(int)):raise ValueError('chi split must partition clock')
    return phase,dict(cutoff_physical_tick=int(cutoff),first_goal_times_s=first_times,
        current_bound_ticks=[int((x&active).sum()) for x in [ca,cb]],
        outside_old_box_ticks=[int((~x&active).sum()) for x in [old_a,old_b]])


def review(root,analysis_name):
    root=Path(root).resolve();analysis=root/analysis_name
    if json.loads((analysis/'status.json').read_text())['phase']!='completed':raise ValueError('complete audited analysis required')
    rows=json.loads((analysis/'results.json').read_text());cfg=json.loads((root/'common_environment_config.json').read_text())
    plan=json.loads((root/'plan.json').read_text());keys=list(configured_cost_keys(cfg));scale=cfg['path_reward']['scale']*cfg['timing']['lower_dt_s']
    with (analysis/'fixed_trace_objective_queries.csv').open() as f:qq=list(csv.DictReader(f))
    q={(r['case_id'],float(r['objective_alpha'])):float(r['scored_cost']) for r in qq}
    if len(rows)!=120 or len(q)!=600:raise ValueError('all120 traces and600 queries required')
    bykey={(r['role'],r['context'],r['alpha']):r for r in rows};raws={};authority=[]
    fields=['active_tick','physical_tick','effective_costs','chi','actual_forward_speed','requested_speed',
        'path_cross_track','path_heading_error','contact_failure','reference','nominal_reference','actor_action_valid']
    for row in rows:
        if row['force_N']!=0 or row['physical_outcome']!='survived':raise ValueError('fixed all-surviving nominal cohort required; do not silently exclude failures')
        path=Path(row['raw_file'])
        if digest(path)!=row['raw_sha256']:raise ValueError('source raw changed')
        with np.load(path) as z:data={k:z[k] for k in fields}
        if not data['active_tick'].all() or len(data['active_tick'])!=2000:raise ValueError('whole ten-second common clock required')
        scored={}
        for user in [0.,1.]:
            s=rescore_trace_alpha(data,alpha=user,config=cfg)
            np.testing.assert_allclose(s['scored_cost'],q[(row['case_id'],user)],rtol=0,atol=1e-6)
            scored[user]=s['effective_costs']*scale
        raws[row['case_id']]=(data,scored)
    chi_proof=[]
    controller=json.loads((root/'source_declaration.json').read_text())['source']['controller']
    for context,spec in plan['contexts'].items():
        with np.load(spec['path']) as path:peak_k=float(np.max(np.abs(path['curvature'])))
        v=max(spec['speed_m_s'],controller['minimum_speed'])
        m2=-(controller['mass']*v*v*controller['cg_height']-controller['mass']*controller['cg_forward']*controller['trail']*controller['gravity'])*np.cos(controller['caster'])/controller['wheelbase']
        m2=(-1 if m2<0 else 1)*max(abs(m2),.1)
        q=(-controller['mass']*controller['gravity']*controller['cg_height'])/m2
        phi=abs(np.arctan(controller['wheelbase']*peak_k/np.cos(controller['caster']))/q)
        chi_proof.append(dict(context=context,requested_speed_m_s=spec['speed_m_s'],peak_curvature_inv_m=peak_k,
            maximum_feedforward_roll_coordinate_rad=float(phi),chi_start_rad=.18,
            maximum_model_chi=float(np.clip((phi-.18)/.12,0,1))))
    chi_max=max(float(data['chi'].max()) for data,_ in raws.values())
    chi_nonzero=sum(int(np.count_nonzero(data['chi'])) for data,_ in raws.values())
    detail=[];states=[]
    for role in plan['evaluation_roles']:
        for context in plan['contexts']:
            pair=[bykey[(role,context,a)] for a in [0.,1.]]
            da,sa=raws[pair[0]['case_id']];db,sb=raws[pair[1]['case_id']]
            masks,state=common_masks(da,db,[r['first_arrival_s'] for r in pair],cfg)
            states.append(dict(role=role,context=context,**state))
            for user in [0.,1.]:
                totals={}
                for stage,mask in masks.items():
                    ac=sa[user][mask].sum(0);bc=sb[user][mask].sum(0);delta=ac-bc
                    d=dict(role=role,context=context,user_alpha=user,stage=stage,paired_ticks=int(mask.sum()),
                        executed0_total=float(ac.sum()),executed1_total=float(bc.sum()),delta_total=float(delta.sum()),
                        delta_preference=float(delta[:3].sum()),delta_other=float(delta[3:].sum()),
                        delta_components=dict(zip(keys,delta.tolist())))
                    np.testing.assert_allclose(d['delta_total'],d['delta_preference']+d['delta_other'],atol=1e-10)
                    detail.append(d);totals[stage]=delta
                np.testing.assert_allclose(totals['full'],totals['through_first_goal_entry']+totals['after_first_goal_entry'],atol=1e-10)
                np.testing.assert_allclose(totals['full'],totals['both_chi_active']+totals['both_chi_zero']+totals['mixed_chi'],atol=1e-10)
    summary=[]
    for role in plan['evaluation_roles']:
        for user in [0.,1.]:
            for stage in masks:
                rr=[r for r in detail if r['role']==role and r['user_alpha']==user and r['stage']==stage]
                if len(rr)!=12:raise ValueError('all12 matched contexts must remain')
                summary.append(dict(role=role,user_alpha=user,stage=stage,contexts=12,
                    nonempty_contexts=sum(r['paired_ticks']>0 for r in rr),
                    mean_paired_s=float(np.mean([r['paired_ticks']*.005 for r in rr])),
                    mean_delta_total=float(np.mean([r['delta_total'] for r in rr])),
                    mean_delta_preference=float(np.mean([r['delta_preference'] for r in rr])),
                    mean_delta_other=float(np.mean([r['delta_other'] for r in rr])),
                    mean_component_delta={k:float(np.mean([r['delta_components'][k] for r in rr])) for k in keys}))
    out=analysis/'lineage_cost_mechanisms_v2';out.mkdir(exist_ok=False)
    result=dict(summary=summary,details=detail,reference_telemetry=states,trajectories=120,reconstructed_queries=240,
        chi_maximum_in_all120_raw=chi_max,chi_nonzero_samples=chi_nonzero,chi_total_samples=240000,source_demand_proof=chi_proof,
        new_physics_steps=0,optimizer_updates=0,source_results_sha256=digest(analysis/'results.json'),
        source_queries_sha256=digest(analysis/'fixed_trace_objective_queries.csv'),script_sha256=digest(__file__),
        sign='Positive delta = executed alpha0 has higher cost than executed alpha1 under SAME user objective.',
        scope='Retrospective descriptive decomposition of fixed whole ten-second traces. Same paired sample masks for both executions; stages do not change the primary full cost. Bound telemetry is not inferred clipping; old-box masks do not replay a native historical controller. Overlapping diagnostic masks are not additive. No causal identification, policy selection, new reward, fitting or independent TEST.')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# 固定120轨迹的分量—阶段—权限诊断','',result['sign'],'',result['scope'],'',
        '所有120条均整窗存活，固定12上下文/五族/两执行端点，无筛选；240项原user0/1完整成本重建吻合原查询。以两个执行端点首次任一到达目标的时刻（含该步）作为共同前缀，余下为共同后缀；未到达者不剔除。χ以原值>1e-7区分非零。参考“触边”是实际数值在边界，不冒称原实现的clip事件。旧共同框采用速度1.5–3.0、参考steer±.35、nominal steer±.3，仅作后验掩码，不是原生重放。','',
        '|模型|用户α|阶段|平均配对秒|总成本差|偏好三通道差|其余成本差|',
        '|---|---:|---|---:|---:|---:|---:|']
    show=['full','through_first_goal_entry','after_first_goal_entry','prefix_both_legal_no_current_bound','prefix_both_legal_no_current_bound_both_chi_active']
    for r in summary:
        if r['stage'] in show:
            lines.append(f"|{r['role']}|{r['user_alpha']:g}|{r['stage']}|{r['mean_paired_s']:.3f}|{r['mean_delta_total']:.4f}|{r['mean_delta_preference']:.4f}|{r['mean_delta_other']:.4f}|")
    lines+=['','完整χ互斥分区、每个成本分量和每情境值见JSON/CSV；诊断掩码来自两条实际轨迹，含后处理选择，不能解释为关掉限幅/提前终止后的控制收益。不设置新通过门槛，不改原α或用分阶段成本替代完整任务评分。原120世界XY图在../lineage_nominal_review/all_xy_lineage_overview.png。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--analysis-name',default='analysis_corrected_roles_v1');a=p.parse_args();review(a.run,a.analysis_name)
