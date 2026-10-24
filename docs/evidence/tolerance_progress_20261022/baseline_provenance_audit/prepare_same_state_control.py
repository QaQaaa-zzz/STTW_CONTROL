"""Authenticate existing alpha0 physical states; change only future preference/latent."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

from sttw_control.preference_best import atomic_json, digest
from sttw_control.balance_probe import classify_contact_balance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, out = Path.cwd(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources, cases, reused, lineage = {}, [], [], []
    original_plan = None
    for side, name in [('left','alpha_physical_boundary_v1'),('right','alpha_physical_boundary_mirror_v1')]:
        root = base/'runs'/name
        plan = json.loads((root/'prepare_plan.json').read_text())
        original_plan = original_plan or plan
        # Wrapper scripts have since gained constant-policy support. Reuse is
        # pinned to the unchanged actual policy/physics execution components.
        for path, sha in plan['source_sha256'].items():
            if '/learning/' in path or path==plan['source_declaration'] or path==plan['model']['actor']:
                if digest(path) != sha:raise ValueError('Original runtime identity changed '+path)
                sources[path] = sha
        rows = json.loads((root/'results.json').read_text())['cases']
        prefixes = json.loads((root/'prefix_results.json').read_text())['cases']
        for prefix in [x for x in prefixes if x['alpha']==0.]:
            src = Path(prefix['snapshot'])
            if digest(src) != prefix['snapshot_sha256']:raise ValueError('Original snapshot changed')
            sources[str(src)] = digest(src)
            with src.open('rb') as stream:state = pickle.load(stream)
            if float(state.alpha)!=0. or not np.asarray(state.history.mask).all():
                raise ValueError('Invalid real alpha0 prefix')
            changed = state.replace(alpha=np.ones_like(state.alpha))
            restored = changed.replace(alpha=state.alpha)
            before, after = jax.tree.leaves(state), jax.tree.leaves(restored)
            assert len(before)==len(after)
            for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
            target = out/(prefix['state_id']+'_request_alpha1.pkl')
            with target.open('wb') as stream:pickle.dump(changed,stream)
            sources[str(target)] = digest(target)
            lineage.append(dict(state_id=prefix['state_id'], source=str(src), source_sha256=digest(src),
                                switched_snapshot=str(target), switched_sha256=digest(target),
                                original_alpha=0., future_alpha=1., changed_field='alpha only',
                                unchanged_leaf_count=len(before)-1, genuine_prefix=prefix))
            for force in [0.,-80.,80.]:
                baseline = next(x for x in rows if x['state_id']==prefix['state_id'] and x['force_N']==force)
                trace = root/(baseline['name']+'.npz')
                with np.load(trace,allow_pickle=False) as z:
                    data={k:z[k] for k in z.files}
                actual=classify_contact_balance(data,horizon_s=4.)
                if actual['balance_label']!=baseline['balance_label']:
                    raise ValueError('Cached physical label mismatch')
                sources[str(trace)] = digest(trace)
                reused.append(dict(**baseline,trace=str(trace),trace_sha256=digest(trace),source_run=str(root)))
                for mode,alpha,snapshot in [('switch_alpha1',1.,target),('zero_latent',0.,src)]:
                    cases.append(dict(state_id=prefix['state_id'],policy=mode,source_alpha=0.,alpha=alpha,
                                      route=prefix['route'],requested_speed=prefix['speed'],fraction=prefix['fraction'],
                                      snapshot=str(snapshot),original_snapshot=str(src),gate_mode=0,force_N=force))
    assert len(lineage)==8 and len(reused)==24 and len(cases)==48
    actor=original_plan['model']['actor']
    zero=base/'runs/zero_middle_physical_left_v1/zero_middle_actor.pkl'
    zero_identity=json.loads((zero.parent/'preexecution_verification.json').read_text())['zero_actor_sha256']
    if digest(zero)!=zero_identity:raise ValueError('Authenticated constant actor changed')
    with zero.open('rb') as stream:zero_parameters=pickle.load(stream)
    for field in ['kernel','bias']:
        np.testing.assert_array_equal(zero_parameters['params']['Dense_3'][field],0.)
    for path in [Path(__file__).resolve(),Path(__file__).with_name('SAME_STATE_CONTROL_DESIGN.md').resolve(),base/'learning/cli/probe_sensor_gate_domain_hold.py',Path(actor),zero]:
        sources[str(path)]=digest(path)
    atomic_json(out/'snapshot_lineage.json',dict(cases=lineage,semantics='genuine alpha0 physical/history state; only current preference request may change'))
    atomic_json(out/'baseline_reuse.json',dict(cases=reused,new_lower_ticks=0))
    plan=dict(project='STTW_CONTROL',purpose='One same-state post-capture control-authority check; no switching-time sweep',
              source_declaration=original_plan['source_declaration'],path_evidence_directory=original_plan['path_evidence_directory'],
              actors=dict(switch_alpha1=dict(path=actor,sha256=digest(actor)),zero_latent=dict(path=str(zero),sha256=digest(zero))),
              cases=cases,source_sha256=sources,sensor_contract=dict(rear_wheel_radius_proxy_m=.1),
              pulse_duration_s=.1,observation_s=4.,maximum_lower_ticks=38400,training_steps=0,
              reused_baseline_cases=24,declared_new_cases=48,force_direction='same original capture heading across all three controls',
              termination='contact_only + identical domain hold; not full task success',
              stop_rule='48 new branches once; no retries, time/force/filter sweep or training extension')
    atomic_json(out/'plan.json',plan)
    atomic_json(out/'status.json',dict(phase='prepared',completed_cases=0,declared_cases=48,active_lower_ticks=0,training_steps=0))
    sys.path.insert(0,str(base/'learning/cli'))
    from prepare_short_recovery_campaign import register
    register(out/'ACTIVE_RUN.json','STTW_CONTROL 同一状态后续偏好控制权限48分支',out/'status.json')
    print(out,'48 new branches;24 reused;38400 maximum lower ticks;0 prefix/optimizer steps')


if __name__ == '__main__':
    main()
