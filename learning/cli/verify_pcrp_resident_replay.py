"""Replay E1a warm/w00 three times in one Isaac process and compare to its cold archive."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from zipfile import ZipFile

from pcrp_replay_gate import REPO, read, write

LAB = '/home/qy/miniconda3/envs/env_isaaclab/bin/python'
STEPS = ('stage', 'policy_steps', 'physics_steps', 'failures', 'valid_steps',
         'push_events', 'post_fork_failures', 'fork_policy_step')


def normalized_contract(folder, expected_batch):
    contract = read(folder / 'contract.json')
    # Process revision and storage paths change by design; compare all physics,
    # initial values, lane ordering, controls and stochastic readbacks.
    contract.pop('code_revision')
    if 'batch_source' in contract:
        split = contract['batch_source']
        if Path(split.pop('output')).resolve() != expected_batch.resolve():
            raise ValueError(f'Wrong source batch for {folder}')
    return contract


def compare_folder(expected, actual, old_batch, new_batch, *, archive):
    source = normalized_contract(expected, old_batch)
    observed = normalized_contract(actual, new_batch)
    contract_diff = [k for k in sorted(source.keys() | observed.keys())
                     if k not in source or k not in observed or source[k] != observed[k]]
    old = read(expected / 'status.json')
    new = read(actual / 'status.json')
    if (old['stage'], old['policy_steps']) != ('complete', 500):
        raise ValueError(f'Incomplete reference for {expected}')
    status_diff = [k for k in STEPS if old[k] != new[k]]
    row = dict(source=str(expected), output=str(actual), contract_diff=contract_diff,
               status_diff=status_diff, valid_steps=new['valid_steps'], failures=new['failures'])
    member_diff = []
    if archive:
        with ZipFile(expected / 'trajectory.npz') as ref, ZipFile(actual / 'trajectory.npz') as replay:
            names = sorted(ref.namelist())
            if len(names) != 52 or len(set(names)) != 52:
                raise ValueError(f'Expected 52 unique reference NPZ members in {expected}')
            observed_names = replay.namelist()
            if len(set(observed_names)) != len(observed_names):
                raise ValueError(f'Duplicate replay member in {actual}')
            member_diff = sorted(set(names).symmetric_difference(observed_names))
            for name in sorted(set(names).intersection(observed_names)):
                if ref.read(name) != replay.read(name):
                    member_diff.append(name)
        row.update(members=52, member_diff=member_diff)
    row['passed'] = not (contract_diff or status_diff or member_diff)
    return row


def main(run, output):
    source = run / 'e1/E1a/warm/w00'
    cases = read(source / 'cases.json')
    if len(cases) != 3 or not all(c['alphas'] == [0] and not any(c['zero_middle']) for c in cases):
        raise ValueError('Expected all three cold warm/w00 alpha-0 H75 slots')
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for i in range(3):
        block = output / f'replay_{i + 1}'
        block.mkdir()
        paths = [dict(c, output=str(block / Path(c['output']).name)) for c in cases]
        write(block / 'cases.json', paths)
        jobs.append(dict(config=str(source / 'settings.json'), terrain=cases[0]['terrain_path'],
                         case_manifest=str(block / 'cases.json'), output=str(block / 'batch')))
    write(output / 'jobs.json', jobs)
    command = ['bash', str(REPO / 'learning/scripts/gpu_queue.sh'), LAB, '-u',
               str(REPO / 'learning/cli/evaluate_isaac_gpu.py'), '--backend', 'lab',
               '--bundle', read(run / 'step0/protocol.json')['bundle'], '--seconds', '10',
               '--resident-jobs', str(output / 'jobs.json')]
    report = dict(stage='queued_or_running', passed=False, reference=str(source), command=command,
                  cold_timing=read(source / 'timing.json'), batches=[])
    report['cold_isaac_sample_progress'] = [json.loads(line) for line in
        (source / 'simulation.log').read_text().splitlines() if re.match(r'^\{"step":\s*\d+', line)]
    write(output / 'status.json', report)
    started = time.monotonic()
    env = dict(os.environ, PYTHONPATH=str(REPO / 'learning/src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    with (output / 'simulation.log').open('x') as log:
        result = subprocess.run(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    report.update(returncode=result.returncode, wall_s_including_queue=time.monotonic()-started)
    pids = []
    try:
        for i, job in enumerate(jobs):
            batch = Path(job['output'])
            entry = dict(index=i, output=str(batch), lanes=[], passed=False)
            report['batches'].append(entry)
            if not (batch / 'status.json').is_file():
                entry['state'] = 'missing_complete_batch'
                continue
            entry['batch'] = compare_folder(source / 'batch', batch, source / 'batch', batch, archive=False)
            for case in cases:
                original = Path(case['output'])
                entry['lanes'].append(compare_folder(original, batch.parent / original.name,
                    source / 'batch', batch, archive=True))
            timing = read(batch / 'resident_timing.json')
            entry['timing'] = timing
            pids.append(timing['process_id'])
            entry['passed'] = (entry['batch']['passed'] and all(r['passed'] for r in entry['lanes'])
                               and timing['candidate'] == i and Path(timing['scene_source']) == Path(jobs[0]['output']))
        report['one_process'] = len(pids) == 3 and len(set(pids)) == 1
        report['passed'] = result.returncode == 0 and report['one_process'] and all(b['passed'] for b in report['batches'])
        report['stage'] = 'complete' if report['passed'] else 'failed'
    except Exception as exc:
        report.update(stage='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(output / 'summary.json', report)
        write(output / 'status.json', report)
    print(f"{report['stage']}: {output / 'status.json'}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.run.resolve(), args.output.resolve()))
