"""All-alpha full-task comparison; re-score the constant policy at matching alpha."""
import argparse
import json
from pathlib import Path

import numpy as np

from sttw_control.preference_best import atomic_json, digest


def archive(path, receipts):
    receipts[str(path.resolve())] = digest(path)
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def metrics(trace, speed, turn_end):
    t = trace['time']
    turn = (trace['path_progress'] >= 3.) & (trace['path_progress'] <= turn_end)
    rms = lambda x: float(np.sqrt(np.mean(x*x))) if len(x) else None
    return dict(observed_ticks=len(t), observed_s=float(t[-1]+.005),
                full_speed_rmse_m_s=rms(trace['actual_forward_speed']-speed),
                full_path_rmse_m=rms(trace['path_cross_track']),
                turn_speed_rmse_m_s=rms(trace['actual_forward_speed'][turn]-speed),
                turn_path_rmse_m=rms(trace['path_cross_track'][turn]),
                peak_roll_rad=float(trace['peak_roll'].max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    status = json.loads((run/'status.json').read_text())
    if status['phase'] != 'completed' or status['completed_cases'] != 4:
        raise ValueError('Wait for all four full-task traces')
    plan = json.loads((run/'plan.json').read_text())
    prior = Path(plan['original_H75_evaluation'])
    prior_status = json.loads((prior/'status.json').read_text())
    if prior_status['phase'] != 'completed' or prior_status['completed_cases'] != 20:
        raise ValueError('Existing H75 panel incomplete')
    declaration = json.loads(Path(plan['source_declaration']).read_text())
    cfg = declaration['config']
    if cfg['path_reward'] != plan['config']['path_reward'] or declaration['source'] != plan['source']:
        raise ValueError('Reward or physical source differs')
    receipts = {str(p.resolve()):digest(p) for p in [run/'plan.json', run/'results.json', prior/'manifest.json', prior/'results.json']}
    old_rows = json.loads((prior/'results.json').read_text())['cases']
    new_rows = json.loads((run/'results.json').read_text())['cases']
    import jax
    import jax.numpy as j
    from sttw_control.path_command_reward import costs, configured_cost_keys
    keys = configured_cost_keys(cfg)
    scale = cfg['path_reward']['scale']*.005
    @jax.jit
    def rescore(alpha, chi, ev, ey, eh, peak, rate, speed):
        def one(c, v, y, h, p, r, s):
            result = costs(alpha=alpha, chi=c, ev=v, ey=y, eh=h, peak_roll=p,
                           roll_rate=r, speed=s, offsets=j.zeros(2), offset_rate=j.zeros(2), config=cfg)
            return j.stack([result['effective'][key] for key in keys]), result['reward']
        return jax.vmap(one)(chi, ev, ey, eh, peak, rate, speed)
    out.mkdir(parents=True, exist_ok=False)
    (out/'cases').mkdir()
    (out/'overviews').mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    alphas = [0., .25, .5, .75, 1.]
    colors = ['#0072B2','#009E73','#CC79A7','#E69F00','#D55E00']
    comparisons, baseline_metrics, overview_links, detail_links = [], [], [], []
    max_reconstruction_error = 0.
    for new_row in new_rows:
        speed = new_row['actual_requested_speed_m_s']
        route = 'left125_R2p2' if new_row['builder'].startswith('left') else 'right125_R2p2'
        baseline = archive(run/'evaluation'/(new_row['name']+'.npz'), receipts)
        path = archive(run/'evaluation'/(new_row['name']+'_path.npz'), receipts)
        old_path = archive(prior/(route+'_path.npz'), receipts)
        np.testing.assert_allclose(path['xy'], old_path['xy'], atol=1e-6, rtol=0)
        for field in ['latent_z', 'offsets', 'target_offset', 'filtered_offset']:
            np.testing.assert_array_equal(baseline[field], 0.)
        if baseline['policy_fault'].any() or baseline['lower_fault'].any():
            raise ValueError('Engineering failure cannot be a task loss')
        base_metrics = metrics(baseline, speed, float(path['turn_end']))
        baseline_metrics.append(dict(route=route, speed=speed, labels=new_row, **base_metrics))
        overview, overview_axes = plt.subplots(4, 1, figsize=(10, 11), layout='constrained')
        overview_axes[0].plot(*baseline['actual_xy'].T, 'k--', label='zero middle')
        for ax, field in zip(overview_axes[1:], ['actual_forward_speed','path_cross_track','phi']):
            ax.plot(baseline['time'], baseline[field], 'k--', label='zero middle')
        curves = {}
        # Logged constant policy used alpha=0; terminal remainder is preserved
        # separately rather than fabricated during alternative scoring.
        logged_effective = np.stack([baseline['effective_cost_'+key] for key in keys], axis=-1)
        remainder = baseline['scored_tick_reward'] + scale*logged_effective.sum(1)
        arguments = [baseline['chi'], baseline['actual_forward_speed']-np.float32(speed),
                     baseline['path_cross_track'], baseline['path_heading_error'], baseline['peak_roll'],
                     baseline['phi_dot'], baseline['actual_forward_speed']]
        for alpha, color in zip(alphas, colors):
            effective, reward = jax.device_get(rescore(j.asarray(alpha), *[j.asarray(x) for x in arguments]))
            reward = reward+remainder
            if alpha == 0.:
                max_reconstruction_error = max(max_reconstruction_error, float(np.max(np.abs(reward-baseline['scored_tick_reward']))))
            row = next(r for r in old_rows if r['route']==route and r['speed']==speed and r['alpha']==alpha and r['force_N']==0)
            trace = archive(prior/(row['name']+'.npz'), receipts)
            mm = metrics(trace, speed, float(path['turn_end']))
            step_curve = lambda t, r: (t[np.arange(0,len(t),4)], np.add.reduceat(r,np.arange(0,len(t),4),axis=0))
            bt, br = step_curve(baseline['time'], reward)
            ht, hr = step_curve(trace['time'], trace['scored_tick_reward'])
            _, bc = step_curve(baseline['time'], -scale*effective)
            _, hc = step_curve(trace['time'], -scale*trace['effective_costs'])
            reward_delta = float(trace['scored_tick_reward'].sum()-reward.sum())
            comparisons.append(dict(route=route, speed=speed, alpha=alpha, H75=mm, zero_middle=base_metrics,
                                    H75_labels=row, zero_middle_labels=new_row,
                                    matched_alpha_return_H75_minus_zero=reward_delta,
                                    zero_rescored_return=float(reward.sum()), H75_logged_return=float(trace['scored_tick_reward'].sum())))
            overview_axes[0].plot(*trace['actual_xy'].T, color=color, label=f'H75 a={alpha:g}')
            for ax,field in zip(overview_axes[1:],['actual_forward_speed','path_cross_track','phi']):
                ax.plot(trace['time'],trace[field],color=color,label=f'H75 a={alpha:g}')
            fig, axes = plt.subplots(9, 1, figsize=(11, 17), layout='constrained')
            for model, data, col, policy_t, step_reward, signed in [('zero middle',baseline,'#555555',bt,br,bc),('H75',trace,color,ht,hr,hc)]:
                complete = len(data['time']) == 2000
                contact = data['physical_failure'].any()
                ending = 'x' if contact else 'o' if complete else 's'
                axes[0].plot(*data['actual_xy'].T,color=col,label=model)
                axes[0].scatter(*data['actual_xy'][-1],color=col,marker=ending)
                for ax,field in zip(axes[1:6],['actual_forward_speed','path_cross_track','path_heading_error','phi','actual_delta']):
                    ax.plot(data['time'],data[field],color=col,label=model)
                axes[1].plot(data['time'],data['governed'][:,0],color=col,ls='--',alpha=.5)
                axes[1].plot(data['time'],data['wheel_speed_proxy'],color=col,ls=':',alpha=.5)
                axes[5].plot(data['time'],data['governed'][:,1],color=col,ls='--',alpha=.5)
                axes[6].plot(policy_t,step_reward,color=col,label=model)
                for ax,key,style in [(axes[7],'speed','-'),(axes[7],'path','--'),(axes[7],'primary',':'),(axes[8],'roll','-'),(axes[8],'working_roll','--')]:
                    ax.plot(policy_t,signed[:,keys.index(key)],color=col,ls=style,label=model+' '+key)
                name=model.replace(' ','_')
                for field in ['time','actual_xy','actual_forward_speed','path_cross_track','path_heading_error','phi','actual_delta','governed','wheel_speed_proxy']:
                    curves[f'a{alpha:g}_{name}_{field}']=data[field]
                curves[f'a{alpha:g}_{name}_policy_time']=policy_t
                curves[f'a{alpha:g}_{name}_step_reward']=step_reward
                curves[f'a{alpha:g}_{name}_signed_components']=signed
            axes[0].plot(*path['xy'][path['s']<=speed*10+1].T,'k:',lw=1,label='original route')
            axes[0].set(xlabel='world X [m]',ylabel='world Y [m]',aspect='equal')
            axes[1].axhline(speed,color='gray',ls=':')
            for ax,ylabel in zip(axes[1:],['speed/ref/proxy [m/s]','signed path [m]','heading [rad]','roll [rad]','steer/ref [rad]','reward /50Hz step','signed task components','signed roll components']):
                ax.set(xlabel='original task time [s]',ylabel=ylabel)
            for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=6)
            tag=f'{route}_v{speed:g}_a{alpha:g}'
            fig.suptitle(tag+' | same-alpha V3 scoring; no external pulse')
            fig.savefig(out/'cases'/(tag+'.png'),dpi=85);plt.close(fig);detail_links.append(tag)
        overview_axes[0].plot(*path['xy'][path['s']<=speed*10+1].T,'k:',label='original route')
        overview_axes[0].set(xlabel='world X [m]',ylabel='world Y [m]',aspect='equal')
        overview_axes[1].axhline(speed,color='gray',ls=':')
        for ax,ylabel in zip(overview_axes[1:],['speed [m/s]','signed path [m]','roll [rad]']):ax.set(xlabel='original task time [s]',ylabel=ylabel)
        for ax in overview_axes:ax.grid(alpha=.2);ax.legend(fontsize=6)
        tag=f'{route}_v{speed:g}'
        overview.suptitle(tag+' | all five alpha, fixed zero-middle baseline')
        overview.savefig(out/'overviews'/(tag+'.png'),dpi=100);plt.close(overview);overview_links.append(tag)
        np.savez_compressed(out/'cases'/(tag+'_all_signed_components.npz'),component_names=np.asarray(keys),**curves)
    assert len(comparisons)==20 and len(baseline_metrics)==4
    atomic_json(out/'analysis.json',dict(comparisons=comparisons, zero_middle=baseline_metrics,
                new_nominal_lower_ticks=status['active_lower_ticks'],new_nominal_cases=4,reused_H75_cases=20,
                max_zero_alpha0_reward_reconstruction_error=max_reconstruction_error,
                source_sha256=receipts, analysis_source_sha256=digest(__file__),
                scope='Fixed deterministic DEV, no independent training repeats; matched alpha scoring is a factual trajectory re-evaluation, not new rollouts.'))
    lines=['# 完整10s：零中层与H75五α名义对照','',
           '四条新增零中层真实轨迹，对照20条已有H75真实轨迹。先看原始XY与真实终点，再看同α的逐50Hz奖励/有符号分量；不同α的加权reward不作排名。','',
           f"新增{status['active_lower_ticks']}lower ticks，0优化；零中层同一物理轨迹在五α下离线重评，不算20新回合。α0原日志reward逐tick重建最大差{max_reconstruction_error:g}。",'',
           '|路线|速度|α|H75弯道速度RMSE|零中层弯道速度RMSE|H75弯道路径RMSE|零中层弯道路径RMSE|同α H75−零 return|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        h,b=r['H75'],r['zero_middle']
        lines.append(f"|{r['route']}|{r['speed']:g}|{r['alpha']:g}|{h['turn_speed_rmse_m_s']:.5f}|{b['turn_speed_rmse_m_s']:.5f}|{h['turn_path_rmse_m']:.5f}|{b['turn_path_rmse_m']:.5f}|{r['matched_alpha_return_H75_minus_zero']:.5f}|")
    lines += ['', '任务、工作侧倾、接触/域退出与观察长度见analysis.json，不以reward代替任务或防摔。虚线为修改参考、点线为轮速代理，均不替代真实速度或原始参考。', '', '## 四组全五α概览', '']
    lines += [f'- [{x}](overviews/{x}.png)' for x in overview_links]
    lines += ['', '## 全部20个同α配对诊断', '']+[f'- [{x}](cases/{x}.png)' for x in detail_links]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=len(comparisons),new_cases=4,new_ticks=status['active_lower_ticks'],figures=24,reconstruction_error=max_reconstruction_error)))


if __name__ == '__main__':
    main()
