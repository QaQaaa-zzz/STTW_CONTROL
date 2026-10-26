"""One-step alpha-table sensitivity on saved states, not alternate rollouts."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def audit(root,output):
    root=Path(root).resolve();output=Path(output)
    if json.loads((root/'analysis/status.json').read_text())['phase']!='completed':
        raise ValueError('completed raw audit required')
    plan=json.loads((root/'plan.json').read_text())
    cfg=json.loads((root/'common_environment_config.json').read_text())
    rows=json.loads((root/'analysis/results.json').read_text())
    with (root/'physical/initial_carry.pkl').open('rb') as f:initial=pickle.load(f).state
    profiles={};profile_hashes={}
    for key,item in plan['nominal_profiles'].items():
        file=item['file']
        if file not in profiles:
            digest=hashlib.sha256(Path(file).read_bytes()).hexdigest()
            if digest!=item['sha256']:raise ValueError('nominal profile changed')
            profile_hashes[file]=digest
            with np.load(file,allow_pickle=False) as z:profiles[file]={k:z[k] for k in z.files}
    interface=cfg['residual_interface'];dt=cfg['timing']['lower_dt_s']
    scale=interface['steer_scale_rad'];rate=interface['correction_steer_rate_rad_s']
    decay=np.exp(-dt/interface['filter_tau_s'][1]);bound=interface['steer_reference_bound_rad']
    cases=[]
    for row in rows:
        if row['role']!='feedback_nominal':continue
        with np.load(row['raw_file'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        ticks=np.flatnonzero(raw['active_tick']&raw['actor_action_valid']&(raw['physical_tick']%4==0))
        lane=row['case_index']
        filtered=np.r_[float(initial.filtered[lane,1]),raw['filtered_offset'][:,1]][ticks]
        offsets=np.r_[float(initial.offsets[lane,1]),raw['correction_offset'][:,1]][ticks]
        nominal=raw['nominal_reference'][ticks,1]
        def publish(target):
            f=decay*filtered+(1-decay)*target
            inc=np.clip(f-offsets,-rate*dt,rate*dt)
            return np.clip(nominal+offsets+inc,-bound,bound)
        native=scale*np.tanh(raw['original_actor_latent'][ticks,1])
        progress=raw['nominal_progress_m'][ticks]
        ey=raw['feedback_path_error_m'][ticks];eh=raw['feedback_heading_error_rad'][ticks]
        results={}
        for label,alpha in [('matched',row['alpha']),('shared_alpha05',.5)]:
            table=profiles[plan['nominal_profiles'][f"{row['context']}/a{alpha:g}"]['file']]
            covered=(progress>=table['progress'][0])&(progress<=table['progress'][-1])
            ney=np.interp(progress,table['progress'],table['path_error'])
            neh=np.interp(progress,table['progress'],table['heading_error'])
            deh=np.arctan2(np.sin(eh-neh),np.cos(eh-neh))
            request=np.where(covered,row['feedback_gain_rad']*np.tanh(deh/.3-(ey-ney)/.5),0.)
            target=np.clip(native+request,-scale*(1-1e-6),scale*(1-1e-6))
            results[label]=dict(request=request,target=target,reference=publish(target),covered=covered)
        # Reconstruct actual first-tick reference through the REAL filter/slew
        # and bounds before interpreting a hypothetical common-table difference.
        expected=results['matched']['reference']
        np.testing.assert_allclose(expected,raw['reference'][ticks,1],atol=3e-6,rtol=2e-5)
        target_delta=results['shared_alpha05']['target']-results['matched']['target']
        reference_delta=results['shared_alpha05']['reference']-expected
        stats=lambda a:dict(max_abs=float(abs(a).max()) if len(a) else None,
            rms=float(np.sqrt(np.mean(a*a))) if len(a) else None,
            abs_sum=float(abs(a).sum()),above_1e4=int((abs(a)>1e-4).sum()))
        cases.append(dict(case_id=row['case_id'],alpha=row['alpha'],bank_index=row.get('bank_index'),
            context=row['context'],force_N=row['force_N'],legal_decisions=len(ticks),
            target_delta_rad=stats(target_delta),one_tick_reference_delta_rad=stats(reference_delta),
            coverage_changes=int(np.sum(results['matched']['covered']!=results['shared_alpha05']['covered'])),
            actual_reference_reconstruction_max_error_rad=float(abs(expected-raw['reference'][ticks,1]).max()) if len(ticks) else None))
    strata={}
    for alpha in sorted({r['alpha'] for r in cases}):
        rr=[r for r in cases if r['alpha']==alpha];n=sum(r['legal_decisions'] for r in rr)
        strata[str(alpha)]=dict(cases=len(rr),legal_decisions=n,
            max_one_tick_reference_change_rad=max(r['one_tick_reference_delta_rad']['max_abs'] or 0 for r in rr),
            mean_abs_one_tick_reference_change_rad=sum(r['one_tick_reference_delta_rad']['abs_sum'] for r in rr)/n if n else None,
            decisions_above_1e4_rad=sum(r['one_tick_reference_delta_rad']['above_1e4'] for r in rr),
            coverage_changes=sum(r['coverage_changes'] for r in rr))
    result=dict(cases=cases,by_alpha=strata,profile_hashes=profile_hashes,
        results_sha256=hashlib.sha256((root/'analysis/results.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_steps=0,optimizer_updates=0,
        scope='Fixed actual Actor alpha and saved actual states; hypothetical replacement by prechosen alpha.5 zero profile for ONE next5ms reference update only. Actual filter/slew/bounds reconstruction checked. No alternate future state/rollout, success prediction, deployed selection or causal outcome claim.',
        threshold='1e-4rad is a descriptive reference-resolution counter, not a success/adoption gate. Raw maxima and means reported.')
    if output.exists():raise ValueError('keep existing audit immutable')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return strata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.run,args.output),indent=2))
