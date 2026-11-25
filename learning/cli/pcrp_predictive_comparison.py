"""Freeze, run, and score the matched 40-slot model-based reference-governor study."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from pcrp_review_batches import LAB, load, write

REPO = Path(__file__).resolve().parents[2]
ALPHAS = (0., .5, 1.)
GOVERNOR = dict(preview_steps=15, speed_response_s=.2)
BRANCHES = [(label, a) for label in ("h75", "predictive_governor") for a in ALPHAS] + [("zero", 0.)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    source = root.parent / 'pcrp_curvature_governor_20261121'
    sources = load(source / 'source_cases.json')
    original = load(source / 'cases.json')
    if len(sources) != 40 or list(sources) != [Path(c['output']).name for c in original]:
        raise ValueError("The forty registered archive slots and physical-case order differ")
    previous = load(source / 'batch/contract.json')
    bundle = Path(previous['bundle'])
    identity = load(bundle / 'identity.json')['files']
    for key, expected in identity.items():
        if digest(bundle / key) != expected or previous['identity']['files'][key] != expected:
            raise ValueError(f"Changed evaluation bundle member {key}")
    base = load(source / 'settings.json')
    del base['traditional_rule']
    base['role'] = 'Matched forty-slot reference-coordination DEV comparison'
    base['plant_options']['replay_clone_origins'] = previous['lab_physics']['terrain_layout']['clone_origins']
    base['evaluation_role'] = base['role']
    root.mkdir(parents=True, exist_ok=False)
    rows, jobs, inputs = [], [], {}
    for label, a in BRANCHES:
        branch = f'{label}_a{a:g}'
        folder = root / branch
        folder.mkdir()
        settings = dict(base)
        if label == 'predictive_governor':
            settings['predictive_reference_governor'] = GOVERNOR
        cases = []
        for slot, (src, old) in enumerate(zip(sources, original)):
            arch = Path(sources[src])
            historical = load(arch / 'contract.json')
            cfg = historical['settings']
            if cfg['seed'] != old['seed'] or cfg['randomization'] != old['randomization'] or historical['terrain'][0] != load(Path(old['terrain_path'])/'identity.json'):
                raise ValueError(f"Historical physical group or terrain differs at {src}")
            condition, group = src.rsplit('_seed', 1)
            raw = folder / src
            cases.append(dict(old, output=str(raw), alphas=[a], zero_middle=[label == 'zero'], case_labels=[branch]))
            # Within this run each arm is matched to the freshly executed H75 slot.
            rows.append(dict(condition=condition, group=int(group), slot=slot, cohort=label, method=f'alpha_{a:g}',
                             alpha=a, lane=0, raw=str(raw), historical=str(arch),
                             h75_reference=str(root / f'h75_a{a:g}' / src), h75_lane=0))
        write(folder / 'settings.json', settings)
        write(folder / 'cases.json', cases)
        for filename in ('settings.json', 'cases.json'):
            inputs[str((folder / filename).relative_to(root))] = digest(folder / filename)
        jobs.append(dict(config=str(folder/'settings.json'), case_manifest=str(folder/'cases.json'),
                         terrain=cases[0]['terrain_path'], output=str(folder/'batch')))
    write(root/'jobs.json', jobs)
    with (root/'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ('jobs.json', 'episodes.csv'):
        inputs[key] = digest(root/key)
    write(root/'protocol.json', dict(bundle=str(bundle), files=identity, lower=base['lower_sha256'],
          original_manifest=str(source/'source_cases.json'), original_manifest_sha256=digest(source/'source_cases.json'),
          inputs=inputs, model=GOVERNOR, candidates=25, slots=40, episodes=280, training_updates=0,
          deck='One resident scene, seven independent 10 s runs in the same forty physical slots. Original curvature-cap batch world origins. H75 and model governor each use alpha 0, .5, 1. Zero uses alpha 0 and is rescored at other preferences.',
          role='Previously examined developmental groups. All seven trajectories per slot measured anew.',
          clocks_s=[.0002, .005, .02], speed_m_s=2.6, model_horizon_s=.30,
          inputs_at_decision='Current measured roll/roll rate and steering, simulated forward speed, current route projection, known centerline and reference-shaper state. All real lanes retain ECBC/ESO plus Local300.',
          score='Same-slot goal, full window, hold, A/B/C; paired wins/losses for each alpha, all termination signals, speed and cross-track RMSE over common completed ticks of all seven arms. No task or parameter selection.',
          metrics='A/B/C require full 10 s and final hold, peak |ey| <= .30/.60/.40 m and speed RMSE <= .20/.10/.15 m/s. Goal section crossing reported separately.',
          stop='Seven jobs with 500 upper ticks each, 280 trajectories maximum, or first failure. No retraining, no tuning, no automatic reruns.'))
    write(root/'status.json', dict(stage='prepared', episodes=280))


def notice(root, tag, message):
    receipt = root/(tag+'_notice.json')
    result = subprocess.run([sys.executable, 'learning/scripts/notify_sttw_job.py', '--message', message,
                             '--receipt', str(receipt)], cwd=REPO)
    if result.returncode:
        write(receipt, dict(status='failed', exit_code=result.returncode))


def run(root):
    plan = load(root/'protocol.json')
    for name, expected in plan['inputs'].items():
        if digest(root/name) != expected:
            raise ValueError(f"Changed frozen manifest {name}")
    for name, expected in plan['files'].items():
        if digest(Path(plan['bundle'])/name) != expected:
            raise ValueError(f"Changed frozen bundle {name}")
    jobs = load(root/'jobs.json')
    if any(Path(j['output']).exists() for j in jobs):
        raise ValueError('This physics run already has a batch output. Use analyze for completed work.')
    cmd = [LAB, '-u', 'learning/cli/evaluate_isaac_gpu.py', '--backend', 'lab', '--bundle', plan['bundle'],
           '--resident-jobs', str(root/'jobs.json'), '--seconds', '10']
    files = ('learning/src/sttw_control/predictive_reference_governor.py',
             'learning/src/sttw_control/isaac_path_rollout.py', 'learning/cli/evaluate_isaac_gpu.py',
             'learning/cli/pcrp_predictive_comparison.py')
    write(root/'code_identity.json', dict(head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source_sha256={name:digest(REPO/name) for name in files},
        command=cmd, initial_state_sha256=digest(REPO/load(Path(jobs[0]['config']))['initial_state'])))
    write(root/'status.json', dict(stage='running', completed=0, slots=40))
    notice(root, 'start_0', f'PCRP 双通道预测调节器配对开始，第1/7批40槽。{root}')
    completed = 0
    env = dict(os.environ, PYTHONPATH=str(REPO/'learning/src')+os.pathsep+os.getenv('PYTHONPATH', ''))
    with (root/'simulation.log').open('a') as log, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors='replace') as proc:
        for line in proc.stdout:
            log.write(line)
            if line.startswith('{"resident_job_complete"'):
                log.flush()
                marker = json.loads(line)
                if marker['index'] != completed or marker['resident_job_complete'] != jobs[completed]['output']:
                    raise ValueError(f"Unexpected resident result {marker}")
                status = load(Path(jobs[completed]['output'])/'status.json')
                if status['stage'] != 'complete' or status['policy_steps'] != 500:
                    raise ValueError(f"Resident job {completed} failed its 10 s batch contract")
                completed += 1
                write(root/'status.json', dict(stage='running', completed=40*completed, slots=40))
                notice(root, f'done_{completed}', f'PCRP 预测调节器配对第{completed}/7批40槽轨迹已保存 {jobs[completed-1]["output"]}')
                if completed < len(jobs):
                    notice(root, f'start_{completed}', f'PCRP 预测调节器配对第{completed+1}/7批40槽开始。')
        if proc.wait() or completed != len(jobs):
            raise RuntimeError(f"Resident evaluation exited {proc.returncode} after {completed} of seven jobs")
    write(root/'status.json', dict(stage='physics_complete', episodes=280))
    analyze(root)
    notice(root, 'report', f'PCRP 预测调节器与H75配对报告及逐组XY {root}/analysis/README.md')


def analyze(root):
    from pcrp_frontier_comparison import (common_window, csv_rows, csv_write, diagnostics, load_contract, read_traces)
    import torch
    torch.set_num_threads(1)
    plan = load(root/'protocol.json')
    specs = csv_rows(root/'episodes.csv')
    if digest(root/'episodes.csv') != plan['inputs']['episodes.csv']:
        raise ValueError('Trajectory manifest differs from registration')
    jobs = load(root/'jobs.json')
    scene = jobs[0]['output']
    for index, job in enumerate(jobs):
        batch = Path(job['output'])
        timing = load(batch/'resident_timing.json')
        status = load(batch/'status.json')
        if (timing['scene_source'] != scene or timing['candidate'] != index or status['stage'] != 'complete'
                or status['policy_steps'] != 500 or status['physics_steps'] != 2_000_000):
            raise ValueError(f'Incomplete or different resident scene at {batch}')
    output = root/'analysis'
    traces_dir = output/'traces'
    panels = output/'panels'
    panels.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(exist_ok=True)
    traces = read_traces(specs, traces_dir)
    groups = defaultdict(list)
    records = {r['raw']:r for r in specs}
    for d in traces:
        r = d['row']
        spec = records[r['raw']]
        slot = int(spec['slot'])
        source = Path(r['raw'])
        contract = load_contract(source)
        reference = load_contract(Path(spec['h75_reference']))
        group = (r['condition'], r['group'])
        if (contract['batch_source']['lane_start'] != slot or contract['origin'] != reference['origin'] or
                contract['identity']['files'] != plan['files'] or contract['settings']['lower_sha256'] != plan['lower']):
            raise ValueError(f"Different physical slot, origin or controller in {source}")
        r['slot'] = slot
        is_model = r['family'] == 'Predictive governor'
        if contract['settings'].get('predictive_reference_governor') != (GOVERNOR if is_model else None):
            raise ValueError(f'Unexpected governor configuration for {source}')
        if is_model:
            with np.load(source/'trajectory.npz') as z:
                live = z['valid'][:,0].astype(bool)
                counts = z['governor_feasible_count'][:,0][live]
                choices = z['governor_candidate'][:,0][live]
                margins = z['governor_margin_rad'][:,0][live]
            if (not np.all((0 <= counts) & (counts <= 25)) or not np.all((0 <= choices) & (choices < 25))
                or not np.isfinite(margins).all() or not np.all((margins >= 0) == (counts > 0))):
                raise ValueError(f'Invalid governor decision log at {source}')
            r.update(decisions=len(counts), no_feasible=int((counts==0).sum()), zero_chosen=int((choices==0).sum()),
                     minimum_forecast_margin_rad=float(margins.min()))
        groups[group].append(d)
    if len(groups) != 40:
        raise ValueError('Missing physical case')
    outcomes = defaultdict(list)
    pairs, gallery = [], []
    for group, items in sorted(groups.items()):
        if len(items) != 7 or {(d['row']['cohort'], d['row']['alpha']) for d in items} != set(BRANCHES):
            raise ValueError(f'Missing matched method in {group}')
        common_window(items)
        links = diagnostics(group, items, panels, ('H75', 'Predictive governor'))
        gallery.append(f'- {group[0]}, group {group[1]} ' + ' · '.join(
            f'[{label}](panels/{link})' for label, link in zip(('XY', 'H75', 'predictive governor'), links)))
        lookup = {(d['row']['cohort'], d['row']['alpha']):d['row'] for d in items}
        for d in items:
            r = d['row']
            outcomes[r['condition'], r['family'], r['alpha']].append(r)
        for a in ALPHAS:
            learned, model = lookup['h75', a], lookup['predictive_governor', a]
            pair = dict(condition=group[0], group=group[1], slot=model['slot'], alpha=a, common_s=model['common_s'])
            for name in ('full','hold','goal_crossed','A','B','C','common_path_rmse_m','common_speed_rmse_m_s'):
                pair['h75_'+name], pair['governor_'+name] = learned[name], model[name]
                pair['governor_minus_h75_'+name] = model[name] - learned[name]
            pairs.append(pair)
    csv_write(output/'episodes.csv', [d['row'] for d in traces])
    csv_write(output/'paired.csv', pairs)
    lines = ['# Paired reference-governor comparison', '',
             '[First recorded XY panel](panels/'+sorted(panels.glob('*_all_XY.png'))[0].name+') · [All episodes](episodes.csv) · [120 model–policy pairs](paired.csv)', '',
             'The controller scores 25 reference pairs at each 50 Hz decision using a 0.30 s nominal rollout. Each row uses the same previously examined groups and an independent reset in the same physical slots. RMSE values average group RMSE on the common fully observed lower ticks of all seven arms. Counts use the actual terminal time of each trial. The three task gates also require full 10 s and final hold.', '',
             '| Terrain | Method | α | n | Full | Goal | Hold | A | B | C | Cross-track RMSE (m) | Speed RMSE (m/s) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (condition, family, a), group in sorted(outcomes.items()):
        ns = [sum(r[k] for r in group) for k in ('full','goal_crossed','hold','A','B','C')]
        rmse = [np.mean([r[k] for r in group]) for k in ('common_path_rmse_m','common_speed_rmse_m_s')]
        lines.append('| ' + ' | '.join(map(str, (condition, family, a, len(group), *ns, *(f'{v:.4f}' for v in rmse)))) + ' |')
    lines.extend(['', '## Same-slot gains and losses', '',
                  '| α | Matched groups | Full gain / loss | Hold gain / loss | A gain / loss | B gain / loss | C gain / loss |',
                  '|---:|---:|---:|---:|---:|---:|---:|'])
    for a in ALPHAS:
        subset = [p for p in pairs if p['alpha'] == a]
        deltas = [Counter(p['governor_minus_h75_'+k] for p in subset) for k in ('full','hold','A','B','C')]
        lines.append(f'| {a:g} | {len(subset)} | ' + ' | '.join(f'{c[1]} / {c[-1]}' for c in deltas) + ' |')
    lines.extend(['', '## Observed trajectories and rewards', '', *gallery, ''])
    (output/'README.md').write_text('\n'.join(lines))
    write(root/'status.json', dict(stage='complete', episodes=280, paired=120, groups=40, report=str(output/'README.md')))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=['prepare','run','analyze'])
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    root = args.output.resolve()
    os.chdir(REPO)
    if args.action == 'run':
        try:
            run(root)
        except Exception as error:
            write(root/'status.json', dict(stage='error', message=str(error)))
            notice(root, 'error', f'PCRP 预测调节器配对异常 {error}，请查阅 {root}')
            raise
    else:
        (prepare if args.action == 'prepare' else analyze)(root)
