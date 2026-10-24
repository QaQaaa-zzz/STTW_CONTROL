"""Prepare A3 authority diagnostic using authentic existing lower histories."""
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
from sttw_control.path_command_policy import correction_tick
from sttw_control.preference_best import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    base, root = Path.cwd(), Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = base/'runs/transient_balance_prediction_v1'
    old = json.loads((source/'scan_plan.json').read_text())
    for file, sha in old['source_sha256'].items():
        if digest(file) != sha:
            raise ValueError('Historical execution identity changed: '+file)
    declaration = json.loads(Path(old['source_declaration']).read_text())
    cfg, dt, ticks = declaration['config'], .005, 800
    prefix = [r for r in json.loads((source/'prefix_results.json').read_text())['cases']
              if r['capture_s'] == .15]
    assert len(prefix) == 6 and all(r['prefix_valid'] for r in prefix)
    interface = cfg['residual_interface']

    @jax.jit
    def profile(nominal, target):
        scale = j.where(target >= 0, interface['speed_positive_scale_m_s'],
                        interface['speed_negative_scale_m_s'])
        latent = j.array([j.arctanh(target/scale), 0.])
        def step(carry, _):
            filtered, offset, ref, _ = correction_tick(*carry, latent, nominal, cfg)
            return (filtered, offset), ref
        return jax.lax.scan(step, (j.zeros(2), j.zeros(2)), None, length=ticks)[1]

    sys.path.insert(0, str(base/'learning/cli'))
    from prepare_short_recovery_campaign import register
    jobs = []
    for delay in [0., .25, .60]:
        folder = root/f'delay_{delay:g}'
        folder.mkdir()
        cases, profiles, paths = [], [], []
        sha = {str(Path(f).resolve()): digest(f) for f in [
            old['source_declaration'], __file__, Path(__file__).with_name('DESIGN.md'),
            Path(__file__).with_name('execute_authority.py'),
            base/'learning/cli/probe_balance_boundary.py',
            base/'learning/src/sttw_control/balance_probe.py',
            base/'learning/src/sttw_control/path_command_policy.py']}
        for row in prefix:
            nominal = np.asarray(row['governed_reference'], dtype=np.float32)
            snapshot = source/(row['state_id']+'_lower_only_snapshot.pkl')
            sha[str(snapshot)] = digest(snapshot)
            for mode, target in [('hold', 0.), ('accel', .24), ('brake', -.24)]:
                if delay == 0 and mode == 'hold':
                    continue
                values = np.asarray(profile(j.asarray(nominal), j.asarray(target)))
                assert values.shape == (ticks, 2) and np.isfinite(values).all()
                np.testing.assert_allclose(values[:, 1], nominal[1], atol=1e-7)
                delta = np.diff(np.vstack([nominal, values]), axis=0)[:, 0]
                assert delta.max() <= interface['correction_speed_up_m_s2']*dt+1e-6
                assert delta.min() >= -interface['correction_speed_down_m_s2']*dt-1e-6
                assert values[:, 0].min() >= interface['speed_reference_bounds_m_s'][0]-1e-6
                assert values[:, 0].max() <= interface['speed_reference_bounds_m_s'][1]+1e-6
                for force in [0., -24., 24., -48., 48.]:
                    cases.append(dict(state_id=row['state_id'], speed_ref_m_s=row['speed_ref_m_s'],
                                      prime_force_N=row['prime_force_N'], capture_s=.15,
                                      mode=mode, target_speed_offset_m_s=target,
                                      second_delay_s=delay, force_N=force, reference=nominal.tolist()))
                    profiles.append(values)
                    paths.append(str(snapshot))
        np.save(folder/'reference_profiles.npy', np.asarray(profiles))
        sha[str(folder/'reference_profiles.npy')] = digest(folder/'reference_profiles.npy')
        status = folder/'status.json'
        plan = dict(project='STTW_CONTROL', purpose='A3 post-prime reference authority before repeated pulse',
                    source_declaration=old['source_declaration'], snapshot=paths[0],
                    case_snapshots=paths, case_groups=[r['state_id'] for r in cases], cases=cases,
                    source_sha256=sha, forces_N=[r['force_N'] for r in cases],
                    references_m_s_rad=[r['reference'] for r in cases],
                    reference_profiles_file=str(folder/'reference_profiles.npy'),
                    pulse_delay_s=delay, pulse_duration_s=.1, observation_s=4.,
                    maximum_lower_ticks=len(cases)*800, training_steps=0,
                    termination='contact_only_v1', wait_for=str(source/'status.json'),
                    force_direction='world vector from capture heading, same for all interventions',
                    stop_rule='Fixed grid once; engineering error stops campaign, no retry')
        atomic_json(folder/'plan.json', plan)
        atomic_json(status, dict(phase='planned', completed_cases=0, declared_cases=len(cases)))
        register(folder/'ACTIVE_RUN.json', 'STTW_CONTROL A3重复扰动参考权限 '+folder.name, status)
        jobs.append(dict(name=folder.name, plan=str(folder/'plan.json'), cases=len(cases)))
    reuse = []
    old_rows = json.loads((source/'results.json').read_text())['cases']
    for case, result, snapshot in zip(old['cases'], old_rows, old['case_snapshots']):
        if case['capture_s'] == .15:
            file = source/(result['name']+'.npz')
            reuse.append(dict(case=case, result=result, snapshot=snapshot,
                              snapshot_sha256=digest(snapshot), raw_file=str(file), raw_sha256=digest(file)))
    assert len(reuse) == 30 and sum(job['cases'] for job in jobs) == 240
    atomic_json(root/'declaration.json', dict(project='STTW_CONTROL', jobs=jobs,
        reused_immediate_hold=reuse, new_cases=240, reused_cases=30,
        maximum_lower_ticks=192000, training_steps=0,
        design=str(Path(__file__).with_name('DESIGN.md').resolve()),
        design_sha256=digest(Path(__file__).with_name('DESIGN.md')),
        source_plan_sha256=digest(source/'scan_plan.json')))
    atomic_json(root/'status.json', dict(phase='prepared', declared_cases=240, completed_cases=0,
                                        active_lower_ticks=0, training_steps=0))
    register(root/'ACTIVE_RUN.json', 'STTW_CONTROL A3重复扰动参考权限总控', root/'status.json')
    print(root, '240 new +30 reused; max192000 lower ticks; no training')


if __name__ == '__main__':
    main()
