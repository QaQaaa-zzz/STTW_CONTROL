"""Re-run the 36 archived PCRP lanes twice with the original world translations."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[2]
LABELS = ['fixed0', 'fixed0.25', 'fixed0.5', 'fixed0.75', 'fixed1', 'Zero']
KEYS = [f'{layout}_v2.6_seed{seed}'
        for layout in ('flat_original', 'position_s4.2', 'position_s4.5')
        for seed in (401, 402)]
FLAGS = ('domain_exit', 'roll_limit_failure', 'contact_failure', 'policy_fault', 'numerical_failure')
RTOL, ATOL_M = 1e-5, 1e-8


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, position):
    stage = root / 'step0'
    stage.mkdir(parents=True, exist_ok=True)
    source_cases = {Path(c['output']).name: c for c in read(position / 'batch_000.json')}
    cases = [source_cases[key] for key in KEYS]
    contracts = [read(Path(c['output']) / 'contract.json') for c in cases]
    cfg = read(position / f'{KEYS[0]}.json')
    original_batch = read(Path(contracts[0]['batch_source']['output']) / 'contract.json')
    # Keep the original process RNG seed as well as each fixed physical-group readback.
    cfg['seed'] = original_batch['settings']['seed']
    cfg['plant_options']['replay_clone_origins'] = [
        origin for c in contracts for origin in c['lab_physics']['terrain_layout']['clone_origins']]
    bundle = Path(contracts[0]['bundle'])
    files = contracts[0]['identity']['files']
    if not all(sha(bundle / name) == digest for name, digest in files.items()):
        raise ValueError('H75 bundle does not match the archived actor, config and route')
    if sha(REPO / cfg['lower_actor']) != cfg['lower_sha256']:
        raise ValueError('Local300 checkpoint differs from the archive')
    for case, contract in zip(cases, contracts):
        if (contract['identity']['files'] != files or case['case_labels'] != LABELS
                or contract['settings']['randomization'] != case['randomization']
                or contract['settings']['lower_sha256'] != cfg['lower_sha256']
                or contract['settings']['user_speed_m_s'] != 2.6
                or [contract[k] for k in ('physics_dt','lower_dt','policy_dt')] != [.0002,.005,.02]):
            raise ValueError(f"Unmatched source contract at {case['output']}")
    write(stage / 'settings.json', cfg)
    for repeat in (1, 2):
        write(stage / f'cases_{repeat}.json',
              [dict(case, output=str(stage / f'repeat_{repeat}' / key))
               for case, key in zip(cases, KEYS)])
    write(stage / 'protocol.json', dict(
        archive=str(position), original_cases=[case['output'] for case in cases],
        original_batch=str(Path(contracts[0]['batch_source']['output'])), bundle=str(bundle),
        identities=dict(**files, local300=cfg['lower_sha256'],
                        initial_state=sha(REPO / cfg['initial_state']),
                        controller=sha(REPO / cfg['controller_config']), asset=sha(REPO / cfg['asset'])),
        xy_rtol=RTOL, xy_atol_m=ATOL_M,
        terminal='Same integer 5 kHz tick and active-step failure flags',
        passed='Every 200 Hz observed XY coordinate allclose and identical terminal tick/type in all 108 pairs',
        repeat='Two fresh processes of 36 lanes, archive-matched world translations and order within each case',
        batch_context='Original 672 lanes, repeats 36 lanes', budget=72, role='Previously examined DEV',
        stop='On any mismatch pause E1 and E2. No automatic change of design.'))
    return stage


def load_case(folder):
    contract = read(folder / 'contract.json')
    if read(folder / 'status.json')['policy_steps'] != 500:
        raise ValueError(f'Incomplete 10 s batch at {folder}')
    with np.load(folder / 'trajectory.npz') as data:
        trace = data['lower_trace'].reshape(-1, 6, 11)
        failed = data['failure'].any(axis=0)
        flags = {key: data[key] for key in FLAGS}
    substeps = round(contract['lower_dt'] / contract['physics_dt'])
    trajectories = []
    for lane in range(6):
        active = np.flatnonzero(trace[:, lane, 6])
        ticks = active * substeps + trace[active, lane, 7].astype(int)
        reason = '+'.join(k for k, a in flags.items() if a[active, lane].any())
        terminal = reason or ('unresolved' if failed[lane] else 'full_window')
        trajectories.append((trace[active, lane, :2], ticks, terminal))
    return contract, trajectories


def compare_case(a, b):
    xa, ta, end_a = a
    xb, tb, end_b = b
    common = min(len(ta), len(tb))
    delta = np.abs(xa[:common] - xb[:common])
    close = np.isclose(xa[:common], xb[:common], atol=ATOL_M, rtol=RTOL)
    mismatch = np.flatnonzero(~close.all(axis=1))
    time_match = np.array_equal(ta, tb)
    xy_match = len(xa) == len(xb) and bool(close.all())
    return dict(passed=bool(time_match and end_a == end_b and xy_match),
                times_equal=time_match, xy_close=xy_match,
                xy_bitwise=xa.shape==xb.shape and xa.tobytes()==xb.tobytes(),
                terminal_equal=end_a==end_b, tick_a=int(ta[-1]), tick_b=int(tb[-1]),
                terminal_a=end_a, terminal_b=end_b, common_samples=common,
                max_abs_coordinate_error_m=float(delta.max()),
                max_xy_distance_m=float(np.linalg.norm(xa[:common]-xb[:common],axis=1).max()),
                first_nonclose_tick=int(ta[mismatch[0]]) if len(mismatch) else None)


def analyze(stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clock = time.monotonic()
    protocol = read(stage / 'protocol.json')
    with np.load(Path(protocol['bundle']) / 'route.npz') as route:
        reference = route['xy']
    rows = []
    plots = stage / 'figures'
    plots.mkdir(exist_ok=True)
    for key, source in zip(KEYS, protocol['original_cases']):
        inputs = [load_case(folder) for folder in (Path(source), stage/'repeat_1'/key, stage/'repeat_2'/key)]
        archived = inputs[0][0]
        for repeat, _ in inputs[1:]:
            for field in ('identity', 'terrain', 'origin', 'randomization_readback', 'delay_steps',
                          'physics_dt', 'lower_dt', 'policy_dt', 'backend'):
                if archived[field] != repeat[field]:
                    raise ValueError(f'{key}: archive mismatch for {field}')
            if any(archived['settings'][k]!=repeat['settings'][k] for k in (
                    'asset', 'lower_actor', 'lower_sha256', 'controller_config', 'initial_state',
                    'alphas', 'zero_middle', 'case_labels', 'user_speed_m_s', 'randomization', 'terrains')):
                raise ValueError(f'{key}: frozen configuration mismatch')
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
        for i, (label, ax) in enumerate(zip(LABELS, axes.flat)):
            lanes = [trajectories[i] for _, trajectories in inputs]
            for first, second, name in ((1,2,'repeat_1_vs_2'), (1,0,'repeat_1_vs_archive'), (2,0,'repeat_2_vs_archive')):
                rows.append(dict(case=key, lane=label, comparison=name, **compare_case(lanes[first], lanes[second])))
            extent = np.concatenate([x for x, _, _ in lanes])
            # Show the portion of the original path up to the furthest observed endpoint.
            nearest = [np.argmin(((reference-x[-1])**2).sum(1)) for x,_,_ in lanes]
            ax.plot(*reference[:max(nearest)+2].T, color='0.45', ls=':', label='Original route')
            for (xy, _, term), color, ls, name in zip(
                    lanes, ('black','tab:blue','tab:orange'), ('-','--',':'), ('Archive','Repeat 1','Repeat 2')):
                ax.plot(*xy.T, color=color, ls=ls, lw=1.3, label=name)
                ax.scatter(*xy[-1], c=color, marker='x' if term!='full_window' else 'o', s=13)
            ax.set(xlabel='World x (m)', ylabel='World y (m)', title=label, aspect='equal')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=4)
        fig.suptitle(f'{key} | frozen H75 / Local300 | x = early termination', y=1.05)
        fig.savefig(plots/f'{key}.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    with (stage/'comparisons.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    totals = {name: {metric:sum(bool(row[metric]) for row in rows if row['comparison']==name)
                     for metric in ('passed','times_equal','xy_close','xy_bitwise','terminal_equal')}
              for name in dict.fromkeys(row['comparison'] for row in rows)}
    passed = all(row['passed'] for row in rows)
    lines = ['# PCRP 确定性回放', '', '## 实际与参考 XY', ''] + [
        f'- [{key}](figures/{key}.png)' for key in KEYS]
    lines += ['', '## 逐轨迹核对', '', '|比较|XY 与终点全部相符|XY 容差内|终止时刻相同|终止类型相同|',
              '|---|---:|---:|---:|---:|']
    for name, counts in totals.items():
        lines.append('|'+name+'|'+ '|'.join(f"{counts[k]}/36" for k in ('passed','xy_close','times_equal','terminal_equal'))+'|')
    lines += ['', '每条轨迹从首个有效下层控制步到实际终点逐坐标比较。坐标容差为 1e-8 m + 1e-5 × 对照坐标绝对值，终止按物理 tick 和类型精确匹配。两次新回放各 36 个并行环境，按原 672 环境档案恢复对应 lane 的世界坐标。',
              '', '门槛通过。' if passed else '门槛未通过，暂停 E1 和 E2，待决定重复设计。', '',
              '[108 项比较与各自的终止类型、时刻、首个超差点和最大误差](comparisons.csv) · [参数与身份](protocol.json)']
    (stage/'README.md').write_text('\n'.join(lines)+'\n')
    status=read(stage/'status.json')
    status.update(stage='complete', passed=passed, gate='passed' if passed else 'mismatch',
                  completed_rollouts=72, totals=totals, analysis_plot_s=time.monotonic()-clock)
    write(stage/'status.json',status)
    return passed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args()
    root=args.run.resolve()
    if args.analyze_only:
        analyze(root/'step0')
        return
    stage=prepare(root,root.parent/'h75_contact_oat'/'position')
    if args.prepare_only:
        return
    protocol=read(stage/'protocol.json')
    env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    status=dict(stage='running',passed=None,repeats=[],completed_rollouts=0)
    write(stage/'status.json',status)
    for repeat in (1,2):
        manifest=stage/f'cases_{repeat}.json'
        directory=stage/f'repeat_{repeat}'
        cmd=['/home/qy/miniconda3/envs/env_isaaclab/bin/python','-u','learning/cli/evaluate_isaac_gpu.py',
             '--backend','lab','--bundle',protocol['bundle'],'--config',str(stage/'settings.json'),
             '--terrain',read(manifest)[0]['terrain_path'],'--output',str(directory/'batch'),
             '--case-manifest',str(manifest),'--seconds','10']
        started=time.monotonic()
        with (stage/f'repeat_{repeat}.log').open('w') as log:
            ret=subprocess.run(cmd,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:
            status.update(stage='error', failed_repeat=repeat, exit_code=ret.returncode)
            write(stage/'status.json',status)
            sys.exit(ret.returncode)
        batch=read(directory/'batch/status.json')
        if batch['stage']!='complete' or batch['policy_steps']!=500:
            raise ValueError(f'Incomplete repeat {repeat}')
        elapsed=time.monotonic()-started
        status['repeats'].append(dict(repeat=repeat,lanes=36,wall_s=elapsed,sampling_s=batch['elapsed_s'],
                                      init_and_shutdown_s=elapsed-batch['elapsed_with_output_s'],
                                      export_s=batch['elapsed_with_output_s']-batch['elapsed_s']))
        status['completed_rollouts']+=36
        write(stage/'status.json',status)
    analyze(stage)


if __name__ == '__main__':
    main()
