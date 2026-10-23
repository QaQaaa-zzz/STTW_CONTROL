"""Read-only CPU verification of saved campaign traces; no physics imports/calls."""
import ast, csv, hashlib, json
from pathlib import Path
import numpy as np

REPORT = Path(__file__).resolve().parent
# Canonical source is recorded in the report; publication does not relocate it.
CAMPAIGN = Path(json.loads((REPORT/'identities.json').read_text())['source_manifest']).parent
REPO = CAMPAIGN.parents[1]
source = REPO / 'learning/src/sttw_control/path_command_reporting.py'
parsed = ast.parse(source.read_text())
function_names = {'_interval_context', 'reconstruct_costs', 'reward_series'}
body = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names]
namespace = {'np': np}
exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
reward_series = namespace['reward_series']
cfg = json.loads((CAMPAIGN/'training/manifest.json').read_text())['config']
dt = cfg['timing']['lower_dt_s']; horizon = cfg['training_future_phase_C']['episode_s']
selections = {f'best_stage_{stage}': json.loads((CAMPAIGN/f'training/best_stage_{stage}.json').read_text()) for stage in (1,2,3)}
selections['last400'] = {'update':400, 'stage':2, 'validation':str(CAMPAIGN/'training/validation/update_0400_stage3')}
rows=[]; summaries={}; mismatches=[]; reward_diffs=[]; cumulative_diffs=[]; path_checks=0; path_check_failures=[]; endpoint_pairs=[]
for name, selection in selections.items():
    folder = Path(selection['validation']); metrics=json.loads((folder/'metrics.json').read_text()); local=[]
    for saved in metrics['cases']:
        alpha=saved['alpha']; case=saved['case']; key='alpha_'+format(alpha,'g').replace('.','p')
        file=folder/key/(case+'.npz')
        with np.load(file, allow_pickle=False) as archive: data={k:archive[k] for k in archive.files}
        # Exact original scenario equality across all alpha traces and stage-local B0.
        with np.load(folder/key/(case+'_path.npz'),allow_pickle=False) as archive: path={k:archive[k] for k in archive.files}
        for comparison in (folder/'alpha_0'/(case+'_path.npz'), CAMPAIGN/f'training/stage_{selection["stage"]+1}_baseline'/(case+'_path.npz')):
            with np.load(comparison,allow_pickle=False) as archive: same=set(path)==set(archive.files) and all(np.array_equal(path[k], archive[k]) for k in path)
            path_checks+=1
            if not same: path_check_failures.append([str(file),str(comparison)])
        n=len(data['time']); ev=data['actual_forward_speed']-data['v_user']; ey=data['path_cross_track']; eh=data['path_heading_error']; roll=data['peak_roll']
        tail=data['time']+dt>horizon-cfg['stage_B_baseline']['hold_s']+1e-5
        turn=abs(data['path_curvature'])>1e-4; mask=turn if turn.any() else np.ones(n,bool)
        rms=lambda values: float(np.sqrt(np.mean(values[mask]**2)))
        observed=dict(physical=bool(data['physical_failure'].any()), domain=bool(data.get('tracking_domain_failure',data['domain_exit']).any()),goal=bool('straight' in case or np.any((data['goal_section_signed_distance']>=0)&(data['path_progress']>=data['goal_progress']))),hold=bool(n==round(horizon/dt) and tail.sum()==round(cfg['stage_B_baseline']['hold_s']/dt) and np.all(abs(ev[tail])<=.1) and np.all(abs(ey[tail])<=.1) and np.all(abs(eh[tail])<=.05)),working=bool(np.any(roll>.302)), peak_roll=float(roll.max()),roll_exposure_s=float(np.sum(roll>.302)*dt),path_rmse=rms(ey),speed_rmse=rms(ev),samples=n)
        for field, value in observed.items():
            if value!=saved[field]: mismatches.append(dict(panel=name,alpha=alpha,case=case,field=field,recomputed=value,saved=saved[field]))
        reward,components=reward_series(data,alpha,cfg)
        logged=np.asarray(data['scored_tick_reward'],float)
        delta=float(np.max(abs(reward-logged))); cumulative_delta=float(np.max(abs(np.cumsum(reward)-np.cumsum(logged))))
        reward_diffs.append(delta); cumulative_diffs.append(cumulative_delta)
        if not np.allclose(reward,logged,rtol=2e-4,atol=2e-6): mismatches.append(dict(panel=name,alpha=alpha,case=case,field='scored_tick_reward',max_abs_error=delta))
        finite=all(np.isfinite(data[k]).all() for k in ['actual_xy','actual_forward_speed','path_cross_track','path_heading_error','phi','scored_tick_reward'])
        if not finite:mismatches.append(dict(panel=name,alpha=alpha,case=case,field='nonfinite_raw_measurements'))
        row=dict(panel=name,update=selection['update'],stage=selection['stage']+1,alpha=alpha,case=case,**observed,turn_samples=int(mask.sum()),observed_seconds=float(n*dt),hold_tail_samples=int(tail.sum()),hold_speed_failed=bool(np.any(abs(ev[tail])>.1)),hold_path_failed=bool(np.any(abs(ey[tail])>.1)),hold_heading_failed=bool(np.any(abs(eh[tail])>.05)),hold_max_abs_speed_error=float(abs(ev[tail]).max()),hold_max_abs_path_error=float(abs(ey[tail]).max()),hold_max_abs_heading_error=float(abs(eh[tail]).max()),whole_path_rmse=float(np.sqrt(np.mean(ey**2))),whole_speed_rmse=float(np.sqrt(np.mean(ev**2))),mean_actual_speed=float(data['actual_forward_speed'].mean()),turn_mean_actual_speed=float(data['actual_forward_speed'][mask].mean()),reward_sum=float(reward.sum()),logged_reward_sum=float(logged.sum()),reward_mean=float(reward.mean()),reward_max_abs_reconstruction_error=delta,cumulative_reward_abs_reconstruction_error=cumulative_delta,xy_final_x=float(data['actual_xy'][-1,0]),xy_final_y=float(data['actual_xy'][-1,1]),trace_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        rows.append(row);local.append(row)
    common=not any(x['physical'] or x['domain'] or not x['goal'] or not x['hold'] or x['working'] for x in local)
    primary=all(x['path_rmse']<=.1 if x['alpha']==0 else x['speed_rmse']<=.05 if x['alpha']==1 else True for x in local if 'straight' not in x['case'])
    for field,val in [('common_qualified',common),('endpoint_primary_qualified',primary)]:
        if val!=metrics[field]:mismatches.append(dict(panel=name,field=field,recomputed=val,saved=metrics[field]))
    summaries[name]=dict(update=selection['update'],stage=selection['stage']+1,cases=len(local),common_qualified=common,endpoint_primary_qualified=primary,physical_failures=sum(x['physical'] for x in local),domain_failures=sum(x['domain'] for x in local),goal_failures=sum(not x['goal'] for x in local),hold_failures=sum(not x['hold'] for x in local),working_failures=sum(x['working'] for x in local),working_roll_exposure_s=sum(x['roll_exposure_s'] for x in local),peak_roll_rad=max(x['peak_roll'] for x in local),hold_failure_causes={k:sum(x[k] for x in local) for k in ['hold_speed_failed','hold_path_failed','hold_heading_failed']},endpoint_primary_failures=[dict(alpha=x['alpha'],case=x['case'],path_rmse=x['path_rmse'],speed_rmse=x['speed_rmse']) for x in local if 'straight' not in x['case'] and ((x['alpha']==0 and x['path_rmse']>.1) or (x['alpha']==1 and x['speed_rmse']>.05))])
    for case in sorted({x['case'] for x in local}):
        zero=next(x for x in local if x['alpha']==0 and x['case']==case);one=next(x for x in local if x['alpha']==1 and x['case']==case)
        endpoint_pairs.append(dict(panel=name,case=case,alpha0_path_rmse=zero['path_rmse'],alpha1_path_rmse=one['path_rmse'],alpha0_speed_rmse=zero['speed_rmse'],alpha1_speed_rmse=one['speed_rmse'],alpha1_speed_rmse_lower=one['speed_rmse']<zero['speed_rmse'],alpha0_path_rmse_lower=zero['path_rmse']<one['path_rmse'],alpha1_mean_speed_minus_alpha0=one['turn_mean_actual_speed']-zero['turn_mean_actual_speed'],scope='actual occupied turn ticks for curves; whole trace for straight; descriptive paired scenario, no matched trajectory-time weighting; alpha1 is not required to worsen path'))
paired=[]
for before in [r for r in rows if r['panel']=='best_stage_3']:
    after=next(r for r in rows if r['panel']=='last400' and r['alpha']==before['alpha'] and r['case']==before['case'])
    key='alpha_'+format(before['alpha'],'g').replace('.','p');case=before['case']
    files=[Path(selections[n]['validation'])/key/(case+'_path.npz') for n in ['best_stage_3','last400']]
    with np.load(files[0],allow_pickle=False) as p0,np.load(files[1],allow_pickle=False) as p1:
        same=set(p0.files)==set(p1.files) and all(np.array_equal(p0[k],p1[k]) for k in p0.files)
    path_checks+=1
    if not same:path_check_failures.append([str(x) for x in files])
    paired.append(dict(alpha=before['alpha'],case=case,identical_stage_scenario=same,best325_hold=before['hold'],last400_hold=after['hold'],best325_working=before['working'],last400_working=after['working'],best325_path_rmse=before['path_rmse'],last400_path_rmse=after['path_rmse'],best325_speed_rmse=before['speed_rmse'],last400_speed_rmse=after['speed_rmse'],best325_total_reward=before['reward_sum'],last400_total_reward=after['reward_sum'],last_minus_best_reward=after['reward_sum']-before['reward_sum']))
scenario_geometry={}
for name,selection in selections.items():
    folder=Path(selection['validation']); details=[]
    for file in sorted((folder/'alpha_0').glob('*_path.npz')):
        with np.load(file,allow_pickle=False) as path:
            curvature=float(np.max(np.abs(path['curvature'])))
            details.append(dict(case=file.stem.removesuffix('_path'),radius_m=None if curvature==0 else 1/curvature,max_abs_curvature_per_m=curvature))
    scenario_geometry[name]=details
reward_comparison_by_alpha={}
for alpha in [0.,.25,.5,.75,1.]:
    selection=[x for x in paired if x['alpha']==alpha]
    reward_comparison_by_alpha[str(alpha)]=dict(cases=len(selection),mean_return_best325=float(np.mean([x['best325_total_reward'] for x in selection])),mean_return_last400=float(np.mean([x['last400_total_reward'] for x in selection])),last400_return_improved_cases=sum(x['last_minus_best_reward']>0 for x in selection))
result=dict(scenario_geometry=scenario_geometry,plot_semantics=dict(ALL_XY='Exact alias of last400 XY_TURN_EXIT; turn/exit zoom, not a full-campaign overview',solid_rewards='Each alpha observed trace reconstructed under its own alpha reward',dashed_rewards='Same stage-local zero-upper local300 B0 trace independently rescored under each alpha; dashed color is not explicitly paired with solid color',B0='Zero upper residual plus frozen local300; not distilled Actor or update0 policy',case_name_warning='R2 in case filenames is a fixed label; actual turn radii are stage1 2.2m, stage2 2.0m, stage3 1.8m'),same_alpha_reward_comparison=reward_comparison_by_alpha,scope='saved NPZ CPU audit only; no simulator, optimizer, model loading, model adoption or source/config changes',reward_reconstruction_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),method='AST-extracted NumPy-only interval/cost/reward reconstruction from report source; compare every raw-derived step to logged scored_tick_reward; independently reconstruct saved curriculum metrics',config_reward_version=cfg['path_reward']['reward_version'],selected_trace_count=len(rows),selected_saved_tick_count=sum(x['samples'] for x in rows),raw_metrics_and_reward_mismatches=mismatches,path_equality_checks=path_checks,path_equality_failures=path_check_failures,max_step_reward_abs_difference=max(reward_diffs),max_cumulative_reward_abs_difference=max(cumulative_diffs),reward_tolerance=dict(rtol=2e-4,atol=2e-6),stage_summaries=summaries,endpoint_pairs=endpoint_pairs,same_stage_best325_last400=paired,caveats=['Reward return across alpha is not a common physical-quality ranking because weights differ.','Reward reconstruction reuses frozen report equations; agreement checks saved reward consistency, not an independent physics validation.','Best325 versus last400 comparisons use identical stage3 paths and speeds only; cross-stage scores cannot establish PPO improvement.','DEV is reused for selection and curriculum advancement, not a held-out test.','No requirement that alpha1 worsen path; report speed direction and path retention separately.'])
(REPORT/'TRAJECTORY_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
with (REPORT/'TRAJECTORY_METRICS.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:result[k] for k in ['selected_trace_count','selected_saved_tick_count','raw_metrics_and_reward_mismatches','path_equality_checks','path_equality_failures','max_step_reward_abs_difference','max_cumulative_reward_abs_difference','stage_summaries']},indent=2))
