"""Score the fixed E2 geometry and initial-speed grids using the frozen 200 Hz traces."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from analyze_roll_braking_boundary import margin
from analyze_roll_contacts import roll_state
from pcrp_replay_gate import FLAGS
from sttw_control.isaac_path_reporting import static_case_metrics

CENTERS = (3.9, 4.2, 4.5)
ALPHAS = (0., .25, .5, .75, 1.)
SHIFTS = (-4, -2, -1, 0, 1, 2, 4)
SPEEDS = (-.02, -.01, 0., .01, .02)
B_NOM = np.radians(6146.05)
BOOTSTRAP_SEED = 401


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def entry_index(wheel_xy, route, terrain):
    # Exact clipped line-segment projection, matching place_isaac_terrain.py. Work up
    # to the first crossing in short blocks; projection uses the original world route.
    ids = np.flatnonzero(route['s'] <= float(route['turn_end']) + 3)
    start = route['xy'][ids[:-1]].astype(float)
    edge = np.diff(route['xy'][ids].astype(float), axis=0)
    lengths = np.linalg.norm(edge, axis=1)
    s, ds = route['s'][ids[:-1]], np.diff(route['s'][ids])
    previous_q = None
    for first in range(0, len(wheel_xy), 50):
        points = wheel_xy[first:first+50]
        delta = points[:, None, :] - start
        frac = np.clip((delta * edge).sum(-1) / lengths**2, 0, 1)
        offset = delta - frac[..., None] * edge
        k = np.sum(offset**2, axis=-1).argmin(axis=1)
        idx = np.arange(len(points))
        n = (edge[k, 0]*offset[idx, k, 1] - edge[k, 1]*offset[idx, k, 0]) / lengths[k]
        q = s[k] + ds[k]*frac[idx, k] - terrain['center_s_m'] - terrain['oblique_s_per_lateral_m']*n
        edge_q = -terrain['length_m']/2
        before = np.r_[q[0] if previous_q is None else previous_q, q[:-1]]
        hit = np.flatnonzero((before < edge_q) & (q >= edge_q) & (abs(n) <= terrain['half_width_m']))
        if len(hit):
            return first + int(hit[0])
        previous_q = q[-1]
    return None


def extract_case(requests, output, curve_name):
    source = Path(requests[0]['source'])
    contract = json.loads((source/'contract.json').read_text())
    status = json.loads((source/'status.json').read_text())
    if status['policy_steps'] != 500 or status['stage'] != 'complete' or status['push_events']:
        raise ValueError(f'Unexpected 10 s evaluation status or perturbation: {source}')
    with np.load(source/'trajectory.npz') as z:
        data = {k:z[k] for k in ('lower_trace','pose','progress','reward','failure',
                                  'body_state','measurement','contact_force',*FLAGS)}
    metrics = static_case_metrics(data, contract)
    trace = data['lower_trace'].reshape(-1, len(metrics), 11)
    settings = contract['settings']
    with np.load(requests[0]['route']) as z:
        route = {k:z[k] for k in ('xy','s','turn_end')}
    curves, rows = {}, []
    front, rear, chassis = [contract['bodies'].index(k) for k in ('frontwheel','rearwheel','chasis')]
    for spec in requests:
        i = spec['lane']
        if settings['alphas'][i] != spec['alpha'] or settings['zero_middle'][i] or settings['seed'] != spec['group']:
            raise ValueError(f'{source}: alpha, controller or group differs from lane manifest')
        terrain = contract['terrain'][i]
        center = spec['center_m']
        if center is None:
            if terrain['height_m'] != 0:
                raise ValueError(f'{source}: flat control has raised terrain')
        elif not np.isclose(terrain['center_s_m'], center+spec['ds_cm']/100, rtol=0, atol=1e-9):
            raise ValueError(f'{source}: wrong ridge position')
        ids = np.flatnonzero(trace[:,i,6])
        x = trace[ids,i]
        t = (ids * round(contract['lower_dt']/contract['physics_dt']) + x[:,7].astype(int)) * contract['physics_dt']
        body = data['body_state'][ids,i]
        phi, omega = roll_state(body[:,chassis])
        H = margin(phi, omega, .7, B_NOM)[2]
        m = metrics[i]
        terminal = '+'.join(k for k in FLAGS if data[k][ids,i].any()) or 'unresolved'
        row = dict(**spec, full=int(not m['failed']), hold=int(m['final_hold']),
                   terminal=terminal if m['failed'] else 'full_window', time_s=t[-1],
                   peak_roll_rad=float(x[:,4].max()), min_H_rad=float(H.min()),
                   entry_time_s=None, entry_v_m_s=None, entry_phi_rad=None, entry_omega_rad_s=None,
                   entry_delta_rad=None, entry_H_rad=None, curve_file=f'curves/{curve_name}.npz')
        wheel_xy = body[:,front,:2] - np.asarray(contract['origin'][i][:2])
        j = None if center is None else entry_index(wheel_xy, route, terrain)
        if j is not None:
            row.update(entry_time_s=t[j], entry_v_m_s=x[j,2].item(), entry_phi_rad=phi[j].item(),
                       entry_omega_rad_s=omega[j].item(), entry_delta_rad=data['measurement'][ids[j],i,2].item(),
                       entry_H_rad=H[j].item())
        forces = data['contact_force'][ids,i]
        curves.update({f'lane{i}_{name}':v for name, v in [
            ('time_s', t), ('H_rad', H), ('phi_rad', phi), ('omega_rad_s', omega),
            ('front_net_vertical_N', forces[:,front,2]), ('rear_net_vertical_N', forces[:,rear,2])]})
        rows.append(row)
    np.savez_compressed(output/'curves'/f'{curve_name}.npz', **curves)
    return rows


def load_grid(rows, arm, centers, groups, values):
    # Tensor ordering: nominal layout, physical group, perturbation, alpha.
    mapping = {}
    for r in rows:
        if r['experiment'] != arm:
            continue
        shift = r['ds_cm'] if arm == 'E2a' else r['dv_m_s']
        key = (r['center_m'],r['group'],shift,r['alpha'])
        if key in mapping:
            raise ValueError(f'Duplicate E2 grid cell {arm}, {key}')
        mapping[key] = r['full']
    expected = [(c,g,d,a) for c in centers for g in groups for d in values for a in ALPHAS]
    if mapping.keys() != set(expected):
        raise ValueError(f'{arm} grid mismatch: missing {set(expected)-mapping.keys()}, extra {mapping.keys()-set(expected)}')
    return np.array([mapping[k] for k in expected],bool).reshape(len(centers),len(groups),len(values),len(ALPHAS))


def flip_stats(grid, axis, groups, label):
    # Each chain holds one group and the other factors fixed. A mixed chain contains
    # both outcomes on the tested axis. Report uniform chains in the overall denominator.
    chains = np.moveaxis(grid, axis, -1)
    changes = chains[...,1:] != chains[...,:-1]
    mixed = np.any(changes, axis=-1, keepdims=True)
    # A second conditional view corresponds literally to cells mixed across groups
    # (e.g. excluding pairs whose BOTH endpoints are unanimous 0/8 or 8/8).
    mixed_node = chains.any(axis=1,keepdims=True) & ~chains.all(axis=1,keepdims=True)
    masks = dict(overall=np.ones_like(changes), mixed_chain=np.broadcast_to(mixed,changes.shape),
                 mixed_group_cell=np.broadcast_to(mixed_node[...,1:]|mixed_node[...,:-1],changes.shape))
    counts = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0,len(groups),size=(10000,len(groups)))
    for subset, mask in masks.items():
        num = (changes & mask).sum(axis=(0,2,3))
        den = mask.sum(axis=(0,2,3))
        sampled_den = den[samples].sum(axis=1)
        good = sampled_den > 0
        scores = num[samples[good]].sum(axis=1)/sampled_den[good]
        lo, hi = np.percentile(scores,[2.5,97.5]) if good.any() else [None,None]
        counts.append(dict(metric=label, subset=subset, flips=int(num.sum()), pairs=int(den.sum()),
                           fraction=float(num.sum()/den.sum()) if den.sum() else None, bootstrap_low=lo,
                           bootstrap_high=hi, resamples_with_pairs=int(good.sum()), physical_groups=','.join(map(str,groups))))
    return counts


def classify(fs, fv, fa_s, fa_v, reversal):
    if (fa_s == 0 and fs == 0) or (fa_v == 0 and fv == 0):
        if not (fs > 0 and fs >= .5*fa_s or fv > 0 and fv >= .5*fa_v):
            return 'degenerate_alpha_benchmark'
    if fs >= .5*fa_s or fv >= .5*fa_v:
        return 'phase_dominated'
    if fs <= .2*fa_s and fv <= .2*fa_v and reversal:
        return 'alpha_structure'
    return 'mixed'


def summarize(rows):
    groups8, groups4 = tuple(range(401,409)), tuple(range(401,405))
    geo = load_grid(rows,'E2a',CENTERS,groups8,SHIFTS)
    vel = load_grid(rows,'E2b',CENTERS,groups4,SPEEDS)
    flat = load_grid(rows,'E2c',(None,),groups4,SPEEDS)
    counts = []
    for grid,axis,groups,name in [(geo[:,:,3:4,:],3,groups8,'f_alpha'),(geo,2,groups8,'f_s'),
                                  (vel,2,groups4,'f_v'),(flat,2,groups4,'f_v_flat'),
                                  (geo[:,:4,3:4,:],3,groups4,'f_alpha_groups401_404'),
                                  (geo,3,groups8,'f_alpha_all_geometry'),(vel,3,groups4,'f_alpha_all_ridge_speeds')]:
        counts.extend(flip_stats(grid,axis,groups,name))
    rates = {r['metric']:r['fraction'] for r in counts if r['subset']=='overall'}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0,8,size=(10000,8))
    smooth = []
    for j,c in enumerate(CENTERS):
        for k,a in enumerate(ALPHAS):
            # Resample a physical group jointly across its seven shifts.
            by_group = geo[j,:,:,k].sum(axis=1)
            lo, hi = np.percentile(by_group[draws].sum(axis=1)/56,[2.5,97.5])
            smooth.append(dict(center_m=c,alpha=a,full=int(by_group.sum()),n=56,
                               fraction=float(by_group.sum()/56),bootstrap_low=float(lo),bootstrap_high=float(hi)))
    sr = {(r['center_m'],r['alpha']):r['full'] for r in smooth}
    reversal = sr[3.9,0] > sr[3.9,1] and sr[4.5,0] < sr[4.5,1]
    decisions = {name:classify(rates['f_s'],rates['f_v'],rates[alpha_s],rates[alpha_v],reversal)
                 for name,alpha_s,alpha_v in [('nominal_centers','f_alpha','f_alpha'),
                     ('matched_groups','f_alpha','f_alpha_groups401_404'),
                     ('all_shifts','f_alpha_all_geometry','f_alpha_all_ridge_speeds')]}
    return counts, smooth, dict(smoothed_endpoint_reversal=reversal, decisions=decisions,
                                sensitivity_changes_verdict=len(set(decisions.values()))>1,
                                flat_f_v=rates['f_v_flat'],flat_control_needs_review=rates['f_v_flat']>0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    stage = args.run.resolve()/'e2'
    out = stage/'analysis'
    clock = time.monotonic()
    (out/'curves').mkdir(parents=True,exist_ok=True)
    specs, baseline = defaultdict(list), {}
    with (stage/'lanes.csv').open() as f:
        for r in csv.DictReader(f):
            row = dict(experiment=r['stage'], center_m=float(r['center_m']) if r['center_m'] else None,
                       group=int(r['seed']),ds_cm=int(r['ds_cm']),dv_m_s=float(r['dv_m_s']),alpha=float(r['alpha']),
                       source=r['output'],lane=int(r['lane']),route=r['route'],terrain=r['terrain'],data_origin=r['source'])
            specs[r['output']].append(row)
            if r['stage'] != 'E2a':
                key = (r['stage'],row['center_m'],row['group'],row['alpha'])
                baseline[key] = dict(row,dv_m_s=0.,source=r['base_output'],lane=ALPHAS.index(row['alpha']),
                                     data_origin='reference_archive')
    for row in baseline.values():
        specs[row['source']].append(row)
    rows = []
    for j,(source,requests) in enumerate(specs.items()):
        rows.extend(extract_case(requests,out,f'case_{j:04d}'))
    episodes = [r for r in rows if r['data_origin'] != 'reference_archive']
    write_csv(out/'episodes.csv',episodes)
    write_csv(out/'velocity_zero_references.csv',[r for r in rows if r['data_origin']=='reference_archive'])
    counts, smooth, summary = summarize(rows)
    write_csv(out/'flip_rates.csv',counts)
    write_csv(out/'smoothed_windows.csv',smooth)
    elapsed = time.monotonic()-clock
    from sttw_control.pcrp_decision_figures import write_e2_figures
    paths = write_e2_figures(episodes,out)
    summary.update(scoring_s=elapsed,plotting_s=time.monotonic()-clock-elapsed,
                   archive_rows=sum(r['data_origin']=='archive' for r in episodes),
                   planned_rows=len(episodes),bootstrap_samples=10000,bootstrap_seed=BOOTSTRAP_SEED)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# E2 接触相位敏感性', '', '## 实际和原始参考轨迹', '']
    lines.extend(f'- [{Path(path).name}]({Path(path).relative_to(out)})' for path in paths)
    lines += ['', '## 判决', '', f"名义中心基准为 **{summary['decisions']['nominal_centers']}**。"
              f" 平滑后端点反转为 {summary['smoothed_endpoint_reversal']}。组匹配和全扰动基准分别为 "
              f"{summary['decisions']['matched_groups']}、{summary['decisions']['all_shifts']}。", '',
              '平地初速扰动翻转比例为 '+f"{summary['flat_f_v']:.4f}。"+
              ('出现翻转，需结合确定性与实现核查。' if summary['flat_control_needs_review'] else '该网格观测到零翻转。'), '',
              '## 翻转率', '', '|量|子集|翻转数/相邻对数|比例|按物理组重采样的描述性 95% 区间|','|---|---|---:|---:|---:|']
    for r in counts:
        score = f"{r['fraction']:.3f}" if r['pairs'] else '—'
        interval = f"{r['bootstrap_low']:.3f}–{r['bootstrap_high']:.3f}" if r['pairs'] else '—'
        lines.append(f"|{r['metric']}|{r['subset']}|{r['flips']}/{r['pairs']}|{score}|{interval}|")
    lines += ['', '主判定采用三处名义中心、8 组、4 对相邻 α 的总体 f_alpha。几何和初速扰动按各自有序网格计算，初速零点复用同条件档案。mixed_chain 仅统计固定组的沿单因素非恒定序列；mixed_group_cell 仅统计一端或两端在物理组之间有分歧的相邻对。稀释检查两者与 overall 并列。条件子集在重采样前固定，按整组抽取 10,000 次；分母为零的抽样次数另计。', '',
              '入障定义取前轮轮心在原路线投影首次越过斜脊上游边界且位于有效宽度内的 200 Hz 实测样点。平地及未到达边界的轨迹保留空值。H 使用 φc=0.70 rad、Bnom=6146.05°/s² 与真实欧拉侧倾角速度，逐步保留时间、H 和前后轮竖向净力至实测终点。它是名义模型裕度。', '',
              '[逐轨迹入障值与原始文件](episodes.csv) · [初速零点来源](velocity_zero_references.csv) · [各中心 × α 的 56 次平滑完成率](smoothed_windows.csv) · [翻转数与区间](flip_rates.csv) · [分析耗时及判决](summary.json)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
