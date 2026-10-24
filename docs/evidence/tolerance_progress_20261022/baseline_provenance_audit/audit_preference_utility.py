"""Cross-score all factual nominal trajectories; no policy or physics updates."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-analysis',type=Path,default=Path('runs/zero_middle_nominal_full_analysis_v1/analysis.json'))
    p.add_argument('--extra-reference-run',type=Path)
    p.add_argument('--save-curves',action='store_true')
    p.add_argument('--zero-run',type=Path,default=Path('runs/zero_middle_nominal_full_v1'));a=p.parse_args()
    original=a.source_analysis.resolve();data=json.loads(original.read_text())
    for file,sha in data['source_sha256'].items():
        if digest(file)!=sha:raise ValueError('Original nominal source changed '+file)
    run=a.zero_run.resolve();plan=json.loads((run/'plan.json').read_text())
    declaration=Path(plan['source_declaration']);decl=json.loads(declaration.read_text());cfg=decl['config'];prior=Path(plan['original_H75_evaluation'])
    assert cfg['path_reward']['reward_version']=='path_persistent_preference_v3'
    from sttw_control.path_command_training import load_environment
    from sttw_control.direct_command_policy import q_ratio
    from sttw_control.path_command_reward import costs,configured_cost_keys
    from sttw_control.geometric_path import build_path
    import jax,jax.numpy as j
    env,initial=load_environment(cfg,decl['source'],'terminal_contract_v2');cc=env.cc;keys=configured_cost_keys(cfg)
    pose=np.asarray(env.physics.helpers.pose(initial.data))
    indices=[keys.index(k) for k in ['speed','path','primary']];scale=cfg['path_reward']['scale']*cfg['timing']['lower_dt_s']
    receipts={str(x.resolve()):digest(x) for x in [original,declaration,Path(__file__)]}
    for name in ['path_command_reward','path_command_env','direct_command_policy','controller']:
        file=Path('learning/src/sttw_control')/(name+'.py');receipts[str(file.resolve())]=digest(file)
    def read(file):
        receipts[str(file.resolve())]=digest(file)
        with np.load(file) as z:return {k:z[k] for k in z.files}
    @jax.jit
    def alpha_cost(alpha,chi,ev,ey,eh,peak,rate,speed):
        def one(c,v,y,h,p,r,s):
            result=costs(alpha=alpha,chi=c,ev=v,ey=y,eh=h,peak_roll=p,roll_rate=r,speed=s,offsets=j.zeros(2),offset_rate=j.zeros(2),config=cfg)
            return j.stack([result['effective'][k] for k in ['speed','path','primary']])
        return jax.vmap(one)(chi,ev,ey,eh,peak,rate,speed)
    alphas=[0.,.25,.5,.75,1.];rows=[];max_component_error=0.;max_return_error=0.;pareto=[];curves=[]
    extra=None
    if a.extra_reference_run:
        extra=json.loads((a.extra_reference_run/'results.json').read_text())
        receipts[str((a.extra_reference_run/'results.json').resolve())]=digest(a.extra_reference_run/'results.json')
        if len(extra['cases'])!=4:raise ValueError('Incomplete external comparator')
    for zero in data['zero_middle']:
        route,speed=zero['route'],zero['speed'];zrow=zero['labels']
        path=read(prior/(route+'_path.npz'))
        rebuilt=build_path('left90_R2' if route.startswith('left') else 'right90_R2',pose,cfg)
        for key in ['s','xy','goal']:
            np.testing.assert_allclose(path[key],np.asarray(getattr(rebuilt,key)),atol=1e-6,rtol=0)
        path['curvature']=np.asarray(rebuilt.curvature)
        members=[('zero_middle',0.,read(run/'evaluation'/(zrow['name']+'.npz')),zero)]
        for beta in alphas:
            record=next(x for x in data['comparisons'] if x['route']==route and x['speed']==speed and x['alpha']==beta)
            trace=read(prior/(record['H75_labels']['name']+'.npz'));members.append((f'H75_a{beta:g}',beta,trace,record['H75']))
        if extra:
            from analyze_nominal_control import metrics
            record=next(x for x in extra['cases'] if x['route']==route and x['speed']==speed)
            trace=read(a.extra_reference_run/'evaluation'/(record['name']+'.npz'))
            members.append(('reference_filter',0.,trace,metrics(trace,speed,float(rebuilt.turn_end))))
        for name,beta,trace,mm in members:
            if not len(trace['time']) or not np.isfinite(trace['scored_tick_reward']).all():
                raise ValueError('Empty/nonfinite factual trajectory; preserve source failure')
            curvature=j.interp(j.asarray(trace['path_progress']),j.asarray(path['s']),j.asarray(path['curvature']))
            chi=j.clip((j.abs(j.arctan(cc.wheelbase*curvature/j.cos(cc.caster))/q_ratio(j.asarray(speed),cc))-.18)/.12,0,1)
            if 'effective_costs' in trace:effective=trace['effective_costs']
            else:effective=np.stack([trace['effective_cost_'+key] for key in keys],axis=-1)
            args=[j.asarray(x) for x in [trace['actual_forward_speed']-np.float32(speed),trace['path_cross_track'],trace['path_heading_error'],trace['peak_roll'],trace['phi_dot'],trace['actual_forward_speed']]]
            factual=np.asarray(alpha_cost(j.asarray(beta),chi,*args))
            error=float(np.max(np.abs(factual-effective[:,indices])));max_component_error=max(max_component_error,error)
            if error>1e-3:raise ValueError('Cannot reconstruct factual alpha-dependent costs '+str(error))
            returns=[];component_returns=[];curve_rewards=[];curve_components=[]
            for alpha in alphas:
                alternative=np.asarray(alpha_cost(j.asarray(alpha),chi,*args))
                reward=trace['scored_tick_reward']-scale*(alternative.sum(1)-effective[:,indices].sum(1))
                returns.append(float(reward.astype('float64').sum()))
                all_costs=effective.copy();all_costs[:,indices]=alternative
                component_returns.append({key:float(-scale*all_costs[:,i].astype('float64').sum()) for i,key in enumerate(keys)})
                if a.save_curves:curve_rewards.append(reward);curve_components.append(-scale*all_costs)
            factual_error=abs(returns[alphas.index(beta)]-float(trace['scored_tick_reward'].astype('float64').sum()));max_return_error=max(max_return_error,factual_error)
            if factual_error>1e-3:raise ValueError('Return reconstruction failed')
            rows.append(dict(route=route,speed=speed,policy=name,behavior_alpha=beta,utility_alphas=alphas,returns=returns,component_returns=component_returns,observed_ticks=len(trace['time']),complete_nominal_horizon=len(trace['time'])==2000,physical_failure=bool(trace['physical_failure'].any()),domain_exit=bool(trace['domain_exit'].any()),metrics={k:mm[k] for k in ['turn_speed_rmse_m_s','turn_path_rmse_m','peak_roll_rad']}))
            if a.save_curves:curves.append((f'{route}_v{speed:g}_{name}',dict(time=trace['time'],reward=np.asarray(curve_rewards),signed_components=np.asarray(curve_components),component_names=np.asarray(keys),alphas=np.asarray(alphas))))
        for name,_,_,mm in members:
            vector=np.array([mm[k] for k in ['turn_speed_rmse_m_s','turn_path_rmse_m','peak_roll_rad']])
            dominators=[]
            for other,_,_,om in members:
                ov=np.array([om[k] for k in ['turn_speed_rmse_m_s','turn_path_rmse_m','peak_roll_rad']])
                if np.all(ov<=vector) and np.any(ov<vector):dominators.append(other)
            pareto.append(dict(route=route,speed=speed,policy=name,dominators=dominators))
    comparisons=[];coverage=[];max_linearity_error=0.
    for context in data['zero_middle']:
        rs=[x for x in rows if x['route']==context['route'] and x['speed']==context['speed']]
        # V3 mixes separately capped endpoint channels affinely in requested
        # alpha for a fixed factual trajectory. This is utility analysis only.
        crossings=[0.,1.]
        for r in rs:
            err=max(abs(value-((1-alpha)*r['returns'][0]+alpha*r['returns'][-1])) for alpha,value in zip(alphas,r['returns']))
            max_linearity_error=max(max_linearity_error,err)
            for other in rs:
                slope=(r['returns'][-1]-r['returns'][0])-(other['returns'][-1]-other['returns'][0])
                if abs(slope)>1e-10:
                    x=(other['returns'][0]-r['returns'][0])/slope
                    if 0<x<1:crossings.append(float(x))
        intervals=[];points=sorted(set(crossings))
        for low,high in zip(points[:-1],points[1:]):
            middle=(low+high)/2
            best=max(rs,key=lambda r:(1-middle)*r['returns'][0]+middle*r['returns'][-1])
            if intervals and intervals[-1]['policy']==best['policy']:intervals[-1]['alpha_hi']=high
            else:intervals.append(dict(alpha_lo=low,alpha_hi=high,policy=best['policy']))
        coverage.append(dict(route=context['route'],speed=context['speed'],factual_utility_envelope=intervals))
        for i,alpha in enumerate(alphas):
            diagonal=next(x for x in rs if x['policy']==f'H75_a{alpha:g}');simple=[x for x in rs if x['policy'] in ['zero_middle','H75_a1','reference_filter']]
            best=max(rs,key=lambda x:x['returns'][i]);best_simple=max(simple,key=lambda x:x['returns'][i])
            comparisons.append(dict(route=context['route'],speed=context['speed'],requested_alpha=alpha,adaptive_return=diagonal['returns'][i],best_simple_policy=best_simple['policy'],best_simple_return=best_simple['returns'][i],delta=diagonal['returns'][i]-best_simple['returns'][i],best_factual_policy=best['policy'],best_factual_return=best['returns'][i],regret_to_factual_best=best['returns'][i]-diagonal['returns'][i],delta_components={key:diagonal['component_returns'][i][key]-best_simple['component_returns'][i][key] for key in keys},versus_all_fixed={r['policy']:diagonal['returns'][i]-r['returns'][i] for r in rs}))
    if max_linearity_error>1e-3:raise ValueError('Affine utility assumption failed')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.save_curves:
        (out/'curves').mkdir()
        for tag,values in curves:np.savez_compressed(out/'curves'/(tag+'.npz'),**values)
    atomic_json(out/'analysis.json',dict(trajectories=rows,comparisons=comparisons,pareto=pareto,coverage=coverage,max_linearity_error=max_linearity_error,max_component_error=max_component_error,max_return_error=max_return_error,input_sha256=receipts,new_physics_ticks=0,optimizer_updates=0,scope='Descriptive cross-scoring on factual nominal DEV trajectories. Row-wise best is retrospective utility comparator, not a deployed selector. Exact Pareto membership is numerical, not statistical or practical significance.'))
    lines=['# 偏好功能的实际价值：名义任务轨迹交叉评分','',f'{len(rows)}条既有真实轨迹×五个评分偏好，0新物理/训练。同α成本重建后，只替换V3中依赖α的speed/path/primary项；修正幅度/变化率及其余成本保留真实日志，不把固定α1重评分冒充在线换α。',f"完整观察10s的轨迹{sum(r['complete_nominal_horizon'] for r in rows)}/{len(rows)}；提前终止的真实长度/触地/域退出逐条保存，原记录终止惩罚保留，不补齐轨迹。",f'分量最大重建误差{max_component_error:g}，累计return最大误差{max_return_error:g}。','',
           '|路线/速度|用户α|随α策略减简单策略中较佳者return|较佳简单策略|全部既有轨迹中最高分策略|','|---|---:|---:|---|---|']
    for r in comparisons:lines.append(f"|{r['route']}/{r['speed']}|{r['requested_alpha']}|{r['delta']:.5f}|{r['best_simple_policy']}|{r['best_factual_policy']}|")
    lines+=['','简单策略包含zero、固定α1及可选reference_filter；逐工况挑较佳者是事后参照，不是可实现oracle算法。完整return矩阵与速度/路径/roll三指标、全部精确Pareto支配关系保留analysis.json。分组奖励不能替代任务或不摔倒标签，也不把毫米级非支配性解释为实用优势。','',
            f"20个用户α/上下文中，随α策略高于上述简单策略较佳者的数目：{sum(x['delta']>0 for x in comparisons)}；相等{sum(x['delta']==0 for x in comparisons)}。不作跨αreturn平均排名。"]
    lines+=['','## 与全部固定行为的比较','',f'固定轨迹的V3效用对用户α为仿射；五点评分的最大线性残差{max_linearity_error:g}。以下仅是在已有固定轨迹中计算效用上包络，不插值真实运动，不是在线选择器。']
    for group in coverage:
        lines.append(f"- {group['route']} / {group['speed']}: "+'; '.join(f"α∈({r['alpha_lo']:.5f},{r['alpha_hi']:.5f}) → {r['policy']}" for r in group['factual_utility_envelope']))
    lines+=['','每行对全部固定α/zero的return差、距本行最优的regret及有符号成本差均已保存；交点是效用的数值相等点，非容扰阈值或置信边界。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(max_component_error=max_component_error,max_return_error=max_return_error,positive=sum(x['delta']>0 for x in comparisons),equal=sum(x['delta']==0 for x in comparisons))))


if __name__=='__main__':main()
