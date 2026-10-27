"""Explain native reward advantages before/after arrival without new physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax.numpy as j
from sttw_control.preference_best import atomic_json,digest
from sttw_control.preference_trace_rescore import rescore_trace_alpha
from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import q_ratio
from sttw_control.geometric_path import build_path
from sttw_control.path_command_reward import configured_cost_keys


def review(root):
    p=json.loads((root/'plan.json').read_text());cfg=p['config'];cc=ControllerConfig(**p['initial_source']['controller'])
    source=root/'analysis/review.json';records=json.loads(source.read_text())['records'];out=root/'task_scope'
    if len(records)!=p['cached_cases']+8:raise ValueError('complete declared panel needed')
    keys=list(configured_cost_keys(cfg));scale=cfg['path_reward']['scale']*.005;rows=[]
    for r in records:
        file=Path(r['raw_file'])
        if digest(file)!=r['raw_sha256']:raise ValueError('raw changed')
        with np.load(file) as z:raw={k:z[k] for k in z.files}
        c=next(c for c in p['cases'] if c['route']==r['route']);path=build_path(c['builder'],np.zeros(3),cfg)
        curvature=np.interp(raw['path_progress'],np.asarray(path.s),np.asarray(path.curvature)).astype(np.float32)
        ff=np.arctan(np.float32(cc.wheelbase)*curvature/np.cos(np.float32(cc.caster)))
        chi=np.clip((abs(ff/np.asarray(q_ratio(j.float32(r['speed']),cc)))-.18)/.12,0,1)
        data=dict(raw,requested_speed=raw['v_user'],chi=chi,contact_failure=raw['physical_failure']|raw['domain_exit'])
        score=rescore_trace_alpha(data,alpha=r['alpha'],config=cfg)
        np.testing.assert_allclose(score['scored_cost'],r['user_objective_costs'][str(r['alpha'])],rtol=3e-5,atol=3e-5)
        n=len(raw['time']);arrival=r['first_arrival_s'];cut=int(round(arrival/.005)) if arrival is not None else n
        # Integrate all original components using their documented names, then
        # retain the difference to scored reward as terminal replacement delta.
        # It is NOT all an extra penalty: replacement discards that tick's rate.
        costs=np.asarray(raw['effective_costs'],dtype=np.float64)
        full={k:float(costs[:,i].sum()*scale) for i,k in enumerate(keys)}
        prefix={k:float(costs[:cut,i].sum()*scale) for i,k in enumerate(keys)}
        row=dict(family=r['family'],route=r['route'],speed=r['speed'],alpha=r['alpha'],
            arrived=r['goal1m_before_native_stop'],first_arrival_s=arrival,prefix_s=cut*.005,
            prefix_is_completed_task=arrival is not None,full_original_cost=r['user_objective_costs'][str(r['alpha'])],
            full_component_integrals=full,prefix_component_integrals=prefix,
            terminal_replacement_delta=float(score['scored_cost']-sum(full.values())),
            prefix_component_sum=float(sum(prefix.values())),
            prefix_peak_roll_rad=float(raw['peak_roll'][:cut].max()),
            prefix_path_rmse_m=float(np.sqrt(np.mean(raw['path_cross_track'][:cut]**2))),
            prefix_speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed'][:cut]-r['speed'])**2))),
            after_arrival_observed_s=(n-cut)*.005 if arrival is not None else None)
        rows.append(row)
    comparisons=[]
    families={r['family'] for r in rows}
    pairings=[(f,'zero') for f in families if f!='zero']
    if 'equilibrium' in families:pairings += [('H75','equilibrium'),('current','equilibrium')]
    for alpha in (0.,1.):
        for family,control in pairings:
            details=[]
            for r in rows:
                if r['family']!=family or r['alpha']!=alpha:continue
                b=next(b for b in rows if (b['family'],b['route'],b['speed'],b['alpha'])==(control,r['route'],r['speed'],alpha))
                if not (r['arrived'] and b['arrived']):continue
                details.append(dict(route=r['route'],speed=r['speed'],
                    arrival_time_delta_s=r['first_arrival_s']-b['first_arrival_s'],
                    prefix_component_delta={k:r['prefix_component_integrals'][k]-b['prefix_component_integrals'][k] for k in keys},
                    prefix_path_rmse_delta_m=r['prefix_path_rmse_m']-b['prefix_path_rmse_m'],
                    prefix_speed_rmse_delta_m_s=r['prefix_speed_rmse_m_s']-b['prefix_speed_rmse_m_s']))
            comparisons.append(dict(model=family,control=control,alpha=alpha,both_arrived=len(details),details=details))
    out.mkdir(exist_ok=False)
    atomic_json(out/'review.json',dict(records=rows,comparisons=comparisons,
        source_review_sha256=digest(source),analysis_source_sha256=digest(__file__),physics_steps=0,
        scope='Original OWN-user component rates times .005*scale; prefix includes first goal entry. Arrival-free rows are observed failures, not cheaper completed tasks. Prefix duration differs by policy. Domain stop is not contact; whole-window native reward is not goal-only task utility. No new metric weights selected.'))
    lines=['# Original reward and goal-arrival scope','',
        '|Model|Route|v|Alpha|Arrived|First arrival s|Prefix path RMSE m|Prefix speed RMSE m/s|Prefix peak roll rad|',
        '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["family"]}|{r["route"]}|{r["speed"]}|{r["alpha"]:g}|{r["arrived"]}|{r["first_arrival_s"]}|{r["prefix_path_rmse_m"]:.4f}|{r["prefix_speed_rmse_m_s"]:.4f}|{r["prefix_peak_roll_rad"]:.4f}|')
    lines += ['', 'A lower original V3 total can be dominated by working-roll costs. This is neither proof of fewer actual falls nor automatic improvement of the goal-only task.',
        'Report every original component and terminal replacement separately; do not tune a new weighted scalar until the desired task tradeoff is fixed.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);review(p.parse_args().root.resolve())
