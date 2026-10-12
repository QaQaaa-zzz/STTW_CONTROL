"""Offline diagnostic gates and complete case figures; imports no simulator."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/path_safety_repair_20261012/diagnostic'
sys.path.insert(0, str(ROOT / 'learning/src'))
from sttw_control.path_command_reporting import reward_series, reconstruct_costs
from sttw_control.path_command_selection import summarize_task
from sttw_control.preference_best import atomic_json


def read(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main():
    preparation = json.loads((RUN / 'PREPARATION.json').read_text())
    accounting = json.loads((RUN / 'ACCOUNTING.json').read_text())
    request = preparation['request']
    prefix_audit = json.loads((OUT / 'prefix_audit.json').read_text()) if (OUT / 'prefix_audit.json').exists() else {}
    # Preserve execution errors. Full retained cases may be plotted descriptively,
    # but a failed strict prefix cannot become a valid causal/launch gate.
    if accounting['state'] != 'completed':
        for row in request['cases']:
            if not (RUN / row['id'] / 'source.json').is_file():
                raise ValueError('Incomplete failed execution: do not infer missing-case gates')
        if not prefix_audit:
            raise ValueError('Failed execution requires independent numerical prefix audit')
    manifests = {key: json.loads(Path(value).read_text()) for key, value in request['model_manifests'].items()}
    configs = {key: value['config'] for key, value in manifests.items()}
    names = list(dict.fromkeys(row['case'] for row in request['cases']))
    originals = {key: {name: read(Path(folder) / (name + '.npz')) for name in names}
                 for key, folder in request['cached_panels'].items()}
    metric_panels = {}
    for key in ('parent1', 'u200'):
        cfg = configs[key]
        folder = Path(request['cached_panels'][key])
        routes = cfg['stage_C_evaluation']['route_ids']
        speeds = [cfg['stage_C_evaluation']['low_speed_m_s'], cfg['stage_C_evaluation']['conflict_speed_m_s']]
        metric_panels[key] = {f'{route}_{speed:.1f}': read(folder / f'{route}_{speed:.1f}.npz')
                             for route in routes for speed in speeds}
    result = dict(schema='path_safety_repair_diagnostic_analysis_v1', accounting=accounting,
                  training_updates=0, observed_cases={}, pair_gates={}, reward_reconstruction_max_error=0.,
                  execution_protocol_pass=accounting['state']=='completed', prefix_audit=prefix_audit,
                  status='diagnostic intervention only; not trained policy adoption or overlimit recovery')
    grouped = {name: {} for name in names}
    row_metrics = {}
    for row in request['cases']:
        name, model = row['case'], row['model']
        d = read(RUN / row['id'] / (name + '.npz'))
        panel = dict(metric_panels[model]); panel[name] = d
        # Other cases supply the existing complete-panel interface; only this case
        # is extracted. No hybrid six-case success claim is made.
        metric = summarize_task(panel, 1, configs[model])['cases'][name]
        ey = np.abs(d['path_cross_track']); tail = d['time'] + .005 > 19.
        metric.update(max_path_m=float(ey.max()), path_exceed_seconds=float(np.sum(ey > .4)*.005),
                      last1s_max_path=float(ey[tail].max()) if tail.any() else None,
                      last1s_max_speed=float(np.abs(d['actual_forward_speed'][tail]-d['v_user'][tail]).max()) if tail.any() else None,
                      last1s_max_heading=float(np.abs(d['path_heading_error'][tail]).max()) if tail.any() else None,
                      qualified_case=bool(not metric['physical_failure'] and not metric['tracking_domain_failure']
                                          and not metric['working_failure'] and not metric['corridor_failure']
                                          and not metric['primary_band_failure'] and metric['goal'] and metric['final_hold']))
        metric['parent_max_path_m'] = float(np.abs(originals[model][name]['path_cross_track']).max())
        metric['path_peak_reduced'] = metric['max_path_m'] < metric['parent_max_path_m']
        metric['strict_prefix_pass'] = None if model != 'u200' else prefix_audit.get(name, {}).get('prefix_pass', False)
        metric['evidence_role'] = 'valid fixed-Actor probe' if model != 'u200' or metric['strict_prefix_pass'] else 'descriptive only; declared prefix check failed'
        starts = d['interval_start_tick']
        _, indices = np.unique(starts, return_index=True)
        metric['target_clips_policy_intervals'] = int(d['diagnostic_target_clip_count'][indices].sum())
        reward, _ = reward_series(d, 1, configs[model])
        error = float(np.abs(reward-d['scored_tick_reward']).max())
        np.testing.assert_allclose(reward, d['scored_tick_reward'], atol=2e-6, rtol=2e-4)
        result['reward_reconstruction_max_error'] = max(result['reward_reconstruction_max_error'], error)
        result['observed_cases'][row['id']] = metric
        row_metrics[row['id']] = metric
        grouped[name][row['id']] = (d, configs[model], 1)
    for beta, backoff in ((.75, 0), (.5, 0), (.5, .05)):
        selected = [row for row in request['cases'] if row['model'] == 'parent1'
                    and row['transform']['beta'] == beta and row['transform']['speed_backoff'] == backoff]
        assert len(selected) == 2
        ms = [row_metrics[row['id']] for row in selected]
        result['pair_gates'][f'beta{beta}_backoff{backoff}'] = dict(
            case_ids=[row['id'] for row in selected],
            supports_turn_repair=all(not m['physical_failure'] and not m['tracking_domain_failure']
                                     and not m['working_failure'] and m['path_peak_reduced'] for m in ms),
            both_cases_fully_qualified=all(m['qualified_case'] for m in ms))
    fade = [row_metrics[row['id']] for row in request['cases'] if row['model'] == 'u200']
    result['supports_turn_repair'] = any(x['supports_turn_repair'] for x in result['pair_gates'].values())
    result['descriptive_latefade_holds'] = all(not m['physical_failure'] and not m['tracking_domain_failure']
                                              and not m['working_failure'] and m['final_hold'] for m in fade)
    result['supports_ordinary_bias_intervention'] = result['descriptive_latefade_holds'] and all(m['strict_prefix_pass'] for m in fade)
    result['allow_conditional_training'] = result['execution_protocol_pass'] and result['supports_turn_repair'] and result['supports_ordinary_bias_intervention']
    result['next_stage'] = 'B/C implementation and bounded validation' if result['allow_conditional_training'] else 'Stop automatic training under declared diagnostic gate'
    assert accounting['active_lower_ticks'] <= 32000
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'metrics.json', result)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    zoom_fig, zoom_axes = plt.subplots(2, 2, figsize=(14, 11))
    rows_md = []
    for name, overview, zoom in zip(names, axes.flat, zoom_axes.flat):
        path = read(Path(request['cached_panels']['B0']) / (name + '_path.npz'))
        datasets = {'B0 local300': (originals['B0'][name], configs['parent1'], 1),
                    'fixed alpha0': (originals['alpha0'][name], configs['parent1'], 0),
                    'Coverage alpha1': (originals['parent1'][name], configs['parent1'], 1)}
        if name.startswith('straight'):
            datasets['V1 alpha1 u200'] = (originals['u200'][name], configs['u200'], 1)
        for row_id, triple in grouped[name].items():
            row = next(row for row in request['cases'] if row['id']==row_id)
            spec = row['transform']
            label = ('late fade' if row['model']=='u200' else f"beta={spec['beta']}, backoff={spec['speed_backoff']}")
            if row['model']=='u200' and not row_metrics[row_id]['strict_prefix_pass']:label+=' (prefix FAIL)'
            datasets[label] = triple
        maximum = max(float(d['path_progress'].max()) for d, _, _ in datasets.values())
        pmask = path['s'] <= maximum + 1
        overview.plot(*path['xy'][pmask].T, 'k--', label='original fixed path')
        colors = ['.45', '#1f77b4', '#ff7f0e'] + (['#d62728','#8c564b'] if name.startswith('straight') else ['#9467bd','#2ca02c','#17becf'])
        zoom.plot(*path['xy'][pmask].T, 'k--', label='original fixed path')
        detail, da = plt.subplots(3, 3, figsize=(18, 13))
        chain, ca = plt.subplots(2, 3, figsize=(17, 9))
        component_fig, component_axes = plt.subplots(4, 3, figsize=(17, 14))
        cumulative_fig, cumulative_axes = plt.subplots(4, 3, figsize=(17, 14))
        for (label, (d, cfg, alpha)), color in zip(datasets.items(), colors):
            t = d['time'] + .005; steps = np.arange(1, len(t)+1)
            reward, components = reward_series(d, alpha, cfg)
            overview.plot(*d['actual_xy'].T, color=color, label=label, lw=1)
            overview.plot(*d['actual_xy'][-1], 'o', color=color, ms=3)
            zoom.plot(*d['actual_xy'].T, color=color, label=label, lw=1.3)
            mask = d['path_progress'] <= float(path['turn_end'])+3 if path['turn_end'] > 0 else d['path_progress'] < 35
            da[0, 0].plot(*d['actual_xy'][mask].T, color=color, label=label)
            series = (d['peak_roll'], d['actual_forward_speed']-d['v_user'], d['path_cross_track'], d['path_heading_error'])
            for ax, value in zip((da[0, 1], da[0, 2], da[1, 0], da[1, 1]), series):ax.plot(t, value, color=color, label=label)
            da[1, 2].plot(steps, reward, color=color, label=label)
            da[2, 0].plot(steps, reward.cumsum(), color=color, label=label)
            for key, linestyle in [('speed','-'), ('path','--')]:
                da[2, 1].plot(steps, components[key], color=color, ls=linestyle, label=label+' '+key)
            da[2, 2].plot(t, d['offsets'][:, 1], color=color, label=label)
            # B0 is rescored separately under each actual objective.
            if label != 'B0 local300':
                br, bparts = reward_series(originals['B0'][name], alpha, cfg)
                da[1, 2].plot(np.arange(1,len(br)+1), br, color=color, ls=':', lw=.7)
                da[2, 0].plot(np.arange(1,len(br)+1), br.cumsum(), color=color, ls=':', lw=.7)
            else:
                bparts = None
            for ax, cumax, (key, values) in zip(component_axes.flat, cumulative_axes.flat, components.items()):
                ax.plot(steps, values, color=color, label=label)
                cumax.plot(steps, values.cumsum(), color=color, label=label)
                ax.set(title=key, xlabel='Actual lower steps', ylabel='signed reward')
                cumax.set(title=key, xlabel='Actual lower steps', ylabel='cumulative signed reward')
                if bparts is not None:
                    bs = np.arange(1, len(bparts[key])+1)
                    ax.plot(bs, bparts[key], color=color, ls=':', lw=.7)
                    cumax.plot(bs, bparts[key].cumsum(), color=color, ls=':', lw=.7)
            ca[0, 0].plot(t, d['governed'][:, 0], color=color, ls='--', label=label+' corrected')
            ca[0, 0].plot(t, d['actual_forward_speed'], color=color, label=label+' measured')
            ca[0, 1].plot(t, d['nominal'][:, 1], color=color, ls=':', label=label+' nominal')
            ca[0, 1].plot(t, d['governed'][:, 1], color=color, ls='--', label=label+' corrected')
            ca[0, 1].plot(t, d['actual_delta'], color=color, label=label+' measured')
            for ax, value in zip((ca[0, 2], ca[1, 0], ca[1, 1], ca[1, 2]),
                                 (d['final_command'][:, 0], d['final_command'][:, 1], d['offsets'][:, 0],
                                  reconstruct_costs(d, alpha, cfg)['offset_rate'][:, 1])):
                ax.plot(t, value, color=color, label=label)
        da[0, 0].plot(*path['xy'][path['s'] <= (float(path['turn_end'])+3 if path['turn_end'] > 0 else 35)].T, 'k--', label='original path')
        da[0, 0].set_aspect('equal', adjustable='box')
        da[0, 1].axhline(.302, color='k', ls='--')
        for y in (-.4, .4):da[1, 0].axhline(y, color='k', ls='--')
        titles = [('Turn/prefix XY','World X (m)','World Y (m)'), ('Physical substep roll peak','Time (s)','rad'),
                  ('Original requested speed error','Time (s)','m/s'), ('Original signed path error','Time (s)','m'),
                  ('Original heading error','Time (s)','rad'), ('Per-step reward; dotted=matched B0','Actual lower steps','reward'),
                  ('Cumulative reward; own objectives','Actual lower steps','reward'), ('Signed speed/path reward components','Actual lower steps','reward'),
                  ('Applied steer correction','Time (s)','rad')]
        for ax, (title, xlabel, ylabel) in zip(da.flat, titles):
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=5)
        ca[0, 0].axhline(float(d['v_user'][0]), color='k', ls=':', label='original speed request')
        ctitles = [('Requested/corrected/measured speed','m/s'), ('Nominal/corrected/measured steer','rad'),
                   ('Final actuator front steer-rate command','rad/s'), ('Final rear wheel-axis-rate command','rad/s'),
                   ('Speed correction','m/s'), ('20ms-block steer correction rate','rad/s')]
        for ax, (title, ylabel) in zip(ca.flat, ctitles):
            ax.set(title=title,xlabel='Time (s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=4)
            if name.startswith('straight'):ax.axvspan(10,10.5,color='grey',alpha=.15)
        if name.startswith('straight'):
            for ax in (da[0,1],da[0,2],da[1,0],da[1,1],da[2,2]):ax.axvspan(10,10.5,color='grey',alpha=.15)
        overview.set(title=name,xlabel='World X (m)',ylabel='World Y (m)');overview.set_aspect('equal',adjustable='box');overview.legend(fontsize=6)
        zoom.set(title=name,xlabel='World X (m)',ylabel='World Y (m)');zoom.set_aspect('equal',adjustable='box');zoom.legend(fontsize=6);zoom.grid(alpha=.2)
        if path['turn_end']>0:
            reference = path['xy'][path['s']<=float(path['turn_end'])+3]
            zoom.set_xlim(float(reference[:,0].min())-.5,float(reference[:,0].max())+1.5)
            zoom.set_ylim(float(reference[:,1].min())-.5,float(reference[:,1].max())+.5)
        else:
            original = originals['u200'][name]
            x10 = float(original['actual_xy'][np.argmin(abs(original['time']-10)),0])
            zoom.set_xlim(x10-.5,x10+6.);zoom.set_ylim(-2.,2.)
            zoom.axhline(.4,color='k',ls=':',label='corridor');zoom.axhline(-.4,color='k',ls=':')
        warning = ' | strict prefix FAIL: descriptive' if name=='straight_2.6' else ''
        subtitle=name+' | full20s, fixed Actor, no external disturbance'+warning+'\nOwn alpha/reward version; intervention is not trained capability; originals and all10 costs in linked NPZ'
        detail.suptitle(subtitle);detail.tight_layout();detail.savefig(OUT/(name+'_diagnostic.png'),dpi=120);plt.close(detail)
        chain.suptitle(subtitle);chain.tight_layout();chain.savefig(OUT/(name+'_command_chain.png'),dpi=120);plt.close(chain)
        for figure, axs, suffix in ((component_fig, component_axes, 'reward_components'),
                                    (cumulative_fig, cumulative_axes, 'cumulative_components')):
            for ax in axs.flat:
                if not ax.has_data():ax.axis('off');continue
                ax.grid(alpha=.2);ax.legend(fontsize=4)
            figure.suptitle(subtitle+'\nDotted lines = same alpha/reward B0 rescoring; actual terminations retained')
            figure.tight_layout();figure.savefig(OUT/(name+'_'+suffix+'.png'),dpi=105);plt.close(figure)
        rows_md.append(f'[{name}: reward/error/detail]({name}_diagnostic.png) · [command chain]({name}_command_chain.png) · '
                       f'[all signed components]({name}_reward_components.png) · [cumulative components]({name}_cumulative_components.png)')
    fig.suptitle('Fixed-Actor causal diagnostic | all four declared cases, all eight interventions\nOriginal20s, no disturbance, frozenlocal300 B0 and fixed alpha0; markers=real ends')
    fig.tight_layout();fig.savefig(OUT/'ALL_XY.png',dpi=140);plt.close(fig)
    zoom_fig.suptitle('Fixed-Actor diagnostic: turn / 10s intervention detail, original world coordinates\nEqual axes; complete20s traces in ALL_XY; straight2.6 prefix check failed')
    zoom_fig.tight_layout();zoom_fig.savefig(OUT/'DETAIL_XY.png',dpi=140);plt.close(zoom_fig)
    lines=['# Fixed-Actor path repair diagnostic — 2026-10-12','', '![All four case local XY](DETAIL_XY.png)','',
           '[Complete20s actual/reference XY, all cases and methods](ALL_XY.png)','',
           '**Fixed Actor interventions only; no training update and no natural-overlimit recovery.**', '',
           f"Execution ACCOUNTING state: **{accounting['state']}**, retained unchanged. All eight physical traces exist; straight2.6 strict prefix failed at 2e-6. Its latefade outcome is descriptive, not accepted causal qualification.", '',
           f"Active lower ticks: {accounting['active_lower_ticks']}/32000. Training transitions: 0.", '',
           '| Intervention/case | Roll rad | Path peak m | Primary normalized error | Final hold | Physical case gates | Strict prefix |',
           '|---|---:|---:|---:|---|---|---|']
    for key,m in result['observed_cases'].items():
        prefix = 'N/A (intervention from reset)' if m['strict_prefix_pass'] is None else str(m['strict_prefix_pass'])
        lines.append(f"| {key} | {m['peak_roll']:.6f} | {m['max_path_m']:.6f} | {m['primary_normalized_rmse']:.3f} | {m['final_hold']} | {m['qualified_case']} | {prefix} |")
    lines+=['',f"Turn-repair support gate: **{result['supports_turn_repair']}**. Ordinary latefade support gate: **{result['supports_ordinary_bias_intervention']}**.",
            f"Conditional training decision: **{result['next_stage']}**.", '',*rows_md,'',
            '[Calculated per-case gates and pair decision](metrics.json) · [Offline source](analyze.py)', '',
            '[All-channel strict prefix audit](prefix_audit.json). Both new traces apply exactly the original Actor latent before10s; low-speed trace matches all original columns exactly. High-speed trace differs in measured motion before intervention despite identical batch shape and frozen source. Numerical compilation/rounding is a hypothesis, not an established root cause. Tolerance was not relaxed and no physical retry was performed.', '',
            '[Original NPZ, exact Actor/config/bank identities and execution accounting]('+str(RUN)+')', '',
            'Original six-case parent and paired200 evidence remain available in [prior report]('+str(ROOT.parent/'path-safety-budget-200/docs/evidence/path_safety_preference_v1/paired200_analysis/INDEX.md')+').', '',
            'All cases use original speed/path requests. .05chi speed backoff may fail original speed-primary gate; it does not change the gate. No incomplete bins, shifted references or zero-padded failed tails are accepted. One historical DEV seed, no holdout claim.']
    (OUT/'INDEX.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:result[key] for key in ('supports_turn_repair','supports_ordinary_bias_intervention','allow_conditional_training','next_stage','reward_reconstruction_max_error')},indent=2))


if __name__=='__main__':main()
