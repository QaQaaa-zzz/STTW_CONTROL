"""Replay frozen E1 winners on the explicitly confirmed E1c geometry protocol."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from pcrp_oracle_search import LABELS, LAYOUTS, LAB, REPO, archived_flat_passed, case_manifest, case_name, csv_write
from pcrp_phase_grid import GYM
from pcrp_replay_gate import compare_case, read, sha, write

SHIFTS = (-2, -1, 1, 2)
NONRIDGE = ('flat_original', 'gravel010')


def frozen_winner(row, times):
    lane = int(row['lane'])
    contract = read(Path(row['raw'])/'contract.json')
    settings = contract['settings']
    if settings['case_labels'][lane] != row['method']:
        raise ValueError(f"Best row and source label disagree at {row['raw']}")
    if int(row['generation']) == 0:
        if row['method'] not in LABELS:
            raise ValueError('Unknown generation-zero method')
        bias = np.zeros((len(times), 2))
    else:
        with np.load(row['knots_npz']) as knots:
            np.testing.assert_array_equal(knots['times_s'], times)
            bias = knots['offsets'][int(row['knots_lane'])]
    return settings['alphas'][lane], settings['zero_middle'][lane], contract['origin'][lane], bias


def confirmed_layouts(scope):
    # Missing keys cannot silently waive a layout. Null explicitly omits a layout
    # only under the user's confirmed scope; other definitions supply its meshes.
    if not isinstance(scope.get('user_confirmation'),str) or not scope['user_confirmation'].strip() or set(scope.get('nonridge',{})) != set(NONRIDGE):
        raise ValueError('E1c needs a confirmed choice for all four flat and four gravel cases.')
    for layout, terrains in scope['nonridge'].items():
        if terrains is not None and (set(terrains) != set(map(str, SHIFTS)) or
                                     any(not (Path(p)/'terrain.npz').is_file() for p in terrains.values())):
            raise ValueError(f'{layout} must explicitly specify its four confirmed terrain assets.')
    return {layout for layout in LAYOUTS if layout not in NONRIDGE or scope['nonridge'][layout] is not None}


def geometry(root, out, scope):
    rows = read(root/'e2/geometry.json')
    mapping = {(f"position_s{r['center_m']}", r['ds_cm']):Path(r['terrain']) for r in rows if r['ds_cm'] in SHIFTS}
    source = read(root.parent/'pcrp_p1_20261121/scene_4.275.json')
    for shift in SHIFTS:
        dest = out/'terrains'/f's4.275_ds{shift:+d}cm'
        scene = Path(str(dest)+'.json')
        scene.parent.mkdir(parents=True,exist_ok=True)
        write(scene,dict(source,center_s_m=round(4.275+shift/100,3)))
        subprocess.run([GYM/'bin/python','learning/cli/place_isaac_terrain.py','--terrain','bend_bump',
                        '--scene-config',str(scene),'--output',str(dest)],cwd=REPO,check=True,stdout=subprocess.DEVNULL,
                       env=dict(os.environ,LD_LIBRARY_PATH=str(GYM/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')))
        mapping['position_s4.275',shift] = dest
    for layout, variants in scope['nonridge'].items():
        if variants is not None:
            mapping.update({(layout,int(shift)):Path(folder).resolve() for shift,folder in variants.items()})
    return mapping


def majority_evidence(nominal, shifted):
    """Tabulate observed denominators and sharp bounds if non-flat shifted pairs are missing."""
    records, by_id = [], {}
    for row in shifted:
        if row.get('role')=='nominal_control':
            raise ValueError('Unchanged terrain controls cannot be included as shifted trials')
        key = (row['case'],int(row['shift_cm']))
        if key in by_id or key[1] not in SHIFTS:
            raise ValueError(f'Unexpected or duplicate shift {key}')
        by_id[key] = int(row['C'])
    expected = {case_name(layout,g) for layout in LAYOUTS for g in range(401,405)}
    if {r['case'] for r in nominal} != expected or len(nominal) != 24 or any(k[0] not in expected for k in by_id):
        raise ValueError('Expected 24 unique E1 cases and no out-of-scope shifted cases')
    for row in nominal:
        c = row['case']
        observed = [by_id.get((c,d)) for d in SHIFTS]
        if any(v is None for v in observed) and any(v is not None for v in observed):
            raise ValueError(f'Incomplete shift set for {c}')
        if row['layout'] not in NONRIDGE and None in observed:
            raise ValueError(f'Missing ridge shifts for {c}')
        passes = None if None in observed else observed[1]+observed[2]
        records.append(dict(case=c,layout=row['layout'],group=row['group'],nominal_C=int(row['C']),
                            C_minus2=observed[0],C_minus1=observed[1],C_plus1=observed[2],C_plus2=observed[3],
                            passed_at_one_cm=passes,blade_edge=None if passes is None else int(row['C'])==1 and passes==0))
    eligible = [r for r in records if r['layout']!='flat_original' and r['nominal_C']]
    measured = [r for r in eligible if r['passed_at_one_cm'] is not None]
    n, obs = len(eligible), len(measured)
    freq = Counter(r['passed_at_one_cm'] for r in measured)
    k = freq[1]+2*freq[2]
    minus, plus = (sum(r[key] for r in measured) for key in ('C_minus1','C_plus1'))
    unknown = n-obs
    metrics = {}
    for label, count, denominator, unobserved in (
            ('pooled_trials',k,2*n,2*unknown), ('both_shifts_per_case',freq[2],n,unknown),
            ('minus1',minus,n,unknown), ('plus1',plus,n,unknown)):
        metrics[label]=dict(passes=count,observed=denominator-unobserved,possible_additional=unobserved,
                            all_nominal_denominator=denominator,
                            majority_certain=2*count>denominator,
                            majority_possible=2*(count+unobserved)>denominator)
    # If each sign must separately retain a majority, take their logical conjunction.
    primary_readings = {
        'pooled':(metrics['pooled_trials']['majority_certain'],metrics['pooled_trials']['majority_possible']),
        'cases_both':(metrics['both_shifts_per_case']['majority_certain'],metrics['both_shifts_per_case']['majority_possible']),
        'each_sign':(all(metrics[s]['majority_certain'] for s in ('minus1','plus1')),
                     all(metrics[s]['majority_possible'] for s in ('minus1','plus1')))}
    if not all(r['nominal_C'] for r in records if r['layout']=='flat_original'):
        verdict='halt_flat_control'
    elif n<=1:
        verdict='route_A_limited_search'
    elif n<6 or not any(v[1] for v in primary_readings.values()):
        verdict='case_by_case'
    elif all(v[0] for v in primary_readings.values()):
        verdict='route_B'
    else:
        verdict='conditional_majority_or_missing_shifts'
    ridge = [r for r in records if r['layout'] not in NONRIDGE]
    ridge_counts = {str(shift):sum(r[key] for r in ridge) for shift,key in
                    zip(SHIFTS,('C_minus2','C_minus1','C_plus1','C_plus2'))}
    return records,dict(verdict=verdict,nonflat_nominal_pass=n,nonflat_nominal_total=20,
                        ridge_nominal_pass=sum(r['nominal_C'] for r in ridge),ridge_total=len(ridge),
                        ridge_shifted_passes=ridge_counts,
                        measured_eligible=obs,unmeasured_eligible=unknown,eligible_two_of_two=freq[2],
                        eligible_one_of_two=freq[1],eligible_zero_of_two=freq[0],
                        readings=metrics,majority_ranges=primary_readings)


def report(directory, records, decision, nominal, shifted, bundle, controls=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=directory/'figures'
    out.mkdir(exist_ok=True)
    with np.load(Path(bundle)/'route.npz') as route:
        reference=route['xy']
        route_s=route['s']
    lines=['# E1c 冻结解偏移复验','','## 实际 / 原始参考 XY','','|案例|名义 C|−2 cm|−1 cm|+1 cm|+2 cm|XY|','|---|---:|---:|---:|---:|---:|---|']
    for case,score in zip(records,nominal):
        key=case['case']
        motions=[('Nominal',Path(score['raw']),int(score['lane']),int(score['full']))]
        motions.extend((f"{v['shift_cm']:+d} cm",Path(v['output']),0,int(v['full'])) for v in shifted if v['case']==key)
        motions.extend((f"Nominal repeat {v['batch_shift_cm']:+d} batch",Path(v['output']),0,int(v['full']))
                       for v in controls if v['case']==key)
        fig,ax=plt.subplots(figsize=(6,4.4),layout='constrained')
        last=0
        for label,path,lane,full in motions:
            with np.load(path/'trajectory.npz') as z:
                trace=z['lower_trace'][:,:,lane].reshape(-1,11)
            xy=trace[trace[:,6].astype(bool),:2]
            last=max(last,int(np.argmin(((reference-xy[-1])**2).sum(axis=1))))
            (line,)=ax.plot(*xy.T,lw=1.1,label=label)
            ax.plot(*xy[-1],marker='o' if full else 'x',color=line.get_color(),ms=4)
        if case['layout'].startswith('position_s'):
            center=float(case['layout'].removeprefix('position_s'))
            ax.plot(*reference[np.abs(route_s-center)<=.4].T,color='sandybrown',lw=4,alpha=.6,label='Nominal ridge on route')
        ax.plot(*reference[:last+2].T,ls=':',lw=1.2,color='0.25',label='Original reference')
        ax.set(xlabel='World x (m)',ylabel='World y (m)',aspect='equal',title=f"{key} | {score['method']} | x = early endpoint")
        ax.legend(fontsize=8)
        fig.savefig(out/f'{key}.png',dpi=135)
        plt.close(fig)
        cells=[case[k] for k in ('nominal_C','C_minus2','C_minus1','C_plus1','C_plus2')]
        lines.append(f"|{key}|"+'|'.join('—' if v is None else str(v) for v in cells)+f"|[XY](figures/{key}.png)|")
    lines+=['','## 任务 C 与稳健性','',
            f"16 个单脊案例中名义通过 {decision['ridge_nominal_pass']}/{decision['ridge_total']}。"
            '四个偏移下的通过数分别为 '+ '、'.join(f"{d:+d} cm {decision['ridge_shifted_passes'][str(d)]}/16" for d in SHIFTS)+'。','',
            f"非平地名义通过 {decision['nonflat_nominal_pass']}/20。名义通过且两侧已评的案例中，"
            f"{decision['eligible_two_of_two']} 个两侧均通过，{decision['eligible_one_of_two']} 个单侧通过，"
            f"{decision['eligible_zero_of_two']} 个两侧均失败（刀锋解）。另有 {decision['unmeasured_eligible']} 个名义通过案例缺少偏移定义。",'',
            '|±1 cm 统计口径|通过数 / 实际分母|全体名义通过案例的分母|未测可增加通过数|','|---|---:|---:|---:|']
    for label,v in decision['readings'].items():
        lines.append(f"|{label}|{v['passes']}/{v['observed']}|{v['all_nominal_denominator']}|0–{v['possible_additional']}|")
    interpretation={'route_B':'达到控制器差距门槛，转入路线 B。','halt_flat_control':'平地阳性对照失败，先复查搜索。',
        'route_A_limited_search':'平地阳性对照通过，非平地至多 1/20 找到可行解。其余案例在本接口与本搜索下未找到可行解，采用路线 A。',
        'case_by_case':'按案例报告已找到的可行解，完整计数未满足路线 B，也未触发路线 A。',
        'conditional_majority_or_missing_shifts':'缺少的偏移数据或“过半”的统计口径会改变路线判断，保留条件结论。'}
    if controls:
        lines += ['', '## 平地和碎石的同槽名义复验', '',
                  '四次复验都保留原地形、赢家和本案例原槽位。图中每条曲线止于该次实际终点。表中的厘米数仅用于识别同批的单脊偏移。', '',
                  '|案例|同批单脊偏移 cm|名义 C / 实测 C|名义终点 / 实测终点 s|最大 XY 距离 m|V 实测减名义|复现核对|',
                  '|---|---:|---:|---:|---:|---:|---|']
        for v in controls:
            lines.append(f"|{v['case']}|{v['batch_shift_cm']:+d}|{v['nominal_C']} / {v['C']}|"
                         f"{v['nominal_time_s']:.4f} / {v['time_s']:.4f}|{v['max_xy_distance_m']:.6g}|"
                         f"{v['V_delta']:.6g}|{'通过' if v['matched'] else '有差异'}|")
        lines+=['','[32 条名义对照的原始文件、端点、C10、hold、C 和轨迹核对](nominal_controls.csv)']
    interpretation['nominal_replay_mismatch'] = '原地形同槽名义复验出现差异。脊偏移的原始计数和条件计算值列于 decision.json，路线判断待核对。'
    lines+=['',interpretation[decision['verdict']],'',
            '[逐轨迹分数与真实终点](scores.csv) · [冻结的偏置来源、控制器和地形](variants.csv) · [各案例的两侧实测数](case_summary.csv) · [判定及缺失界限](decision.json)']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')


def execute_legacy(root, scope_path):
    scope = read(scope_path)
    included = confirmed_layouts(scope)
    gate = read(root/'step0/status.json')
    if gate.get('stage')!='complete' or gate.get('passed') is not True:
        raise SystemExit('E1c requires a passed Step 0.')
    stages = [root/'e1'/s for s in ('E1a','E1b')]
    for stage in stages:
        status = read(stage/'status.json')
        if status.get('stage')!='complete' or status.get('generation')!=10 or not archived_flat_passed(stage,status):
            raise SystemExit(f'E1c requires complete search and positive flat control in {stage}.')
    nominal = []
    cases = {}
    for stage in stages:
        protocol = read(stage/'protocol.json')
        for c in protocol['cases']:
            cases[c['case']] = c
        with (stage/'best.csv').open() as f:
            for row in csv.DictReader(f):
                c = cases[row['case']]
                nominal.append(dict(row,group=c['group'],layout=c['layout'],C=int(row['C'])))
    times = read(stages[0]/'protocol.json')['times_s']
    if times!=read(stages[1]/'protocol.json')['times_s'] or len(cases)!=24 or len(nominal)!=24:
        raise ValueError('Incompatible E1a/b search contracts')
    directory = root/'e1/E1c'
    directory.mkdir(exist_ok=False)
    write(directory/'confirmed_protocol.json',scope)
    terrains=geometry(root,directory,scope)
    manifests, offsets, origins, variants = [],[],[],[]
    for winner in nominal:
        layout = winner['layout']
        if layout not in included:
            continue
        a,zero,origin,bias=frozen_winner(winner,times)
        c=cases[winner['case']]
        original_meta=read(Path(c['terrain'])/'identity.json')
        for shift in SHIFTS:
            path=terrains[layout,shift]
            meta=read(path/'identity.json')
            if layout not in NONRIDGE:
                if (not np.isclose(meta['center_s_m'],original_meta['center_s_m']+shift/100,rtol=0,atol=1e-9)
                        or any(meta[k]!=original_meta[k] for k in ('height_m','length_m','half_width_m',
                            'route_sha256','horizontal_scale_m','vertical_scale_m','oblique_s_per_lateral_m'))):
                    raise ValueError(f'Unmatched shifted ridge at {path}')
            raw=directory/f"{c['case']}_ds{shift:+d}cm"
            manifests.append(dict(seed=c['group'],randomization=c['randomization'],case_labels=[winner['method']],
                                  alphas=[a],zero_middle=[zero],zero_reference_channels=[[False,False]],
                                  user_speed_m_s=2.6,terrain_path=str(path),output=str(raw)))
            origins.append(origin)
            offsets.append(bias)
            variants.append(dict(case=c['case'],layout=layout,group=c['group'],shift_cm=shift,method=winner['method'],
                                 alpha=a,zero_middle=zero,origin=origin,terrain=str(path),terrain_sha256=meta['sha256'],
                                 source_winner=winner['raw'],source_lane=int(winner['lane']),
                                 source_knots=winner['knots_npz'],source_knots_lane=winner['knots_lane'],output=str(raw)))
    bundle=read(root/'step0/protocol.json')['bundle']
    cfg=read(root.parent/'h75_contact_oat/position/flat_original_v2.6_seed401.json')
    cfg['plant_options']['replay_clone_origins']=origins
    cfg['evaluation_role']='E1c DEV single replay of each frozen winner on confirmed shifted geometry'
    knots=directory/'knots.npz'
    np.savez(knots,times_s=times,offsets=offsets)
    cfg.update(oracle_knots=str(knots),oracle_knots_sha256=sha(knots))
    write(directory/'settings.json',cfg)
    write(directory/'cases.json',manifests)
    csv_write(directory/'variants.csv',variants)
    started=time.monotonic()
    env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    with (directory/'simulation.log').open('x') as log:
        subprocess.run(['/bin/bash',str(REPO/'learning/scripts/gpu_queue.sh'),LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--seconds','10',
                        '--bundle',bundle,'--config',directory/'settings.json','--case-manifest',directory/'cases.json',
                        '--terrain',manifests[0]['terrain_path'],'--output',directory/'batch'],cwd=REPO,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    status=read(directory/'batch/status.json')
    if status.get('stage')!='complete' or status.get('policy_steps')!=500:
        raise ValueError('Incomplete E1c batch')
    wall=time.monotonic()-started
    sys.path.insert(0,str(REPO/'docs/paper/english'))
    from score_decision_oracle import score_trajectories
    shifted=[]
    for v in variants:
        [score]=score_trajectories(Path(v['output']),[v['method']])
        shifted.append({**v,**score})
    records,decision=majority_evidence(nominal,shifted)
    csv_write(directory/'scores.csv',shifted)
    csv_write(directory/'case_summary.csv',records)
    write(directory/'decision.json',decision)
    scoring_done=time.monotonic()
    report(directory,records,decision,nominal,shifted,bundle)
    write(directory/'status.json',dict(stage='complete',completed=len(shifted),physics_wall_s=wall,
          sampling_s=status['elapsed_s'],scoring_s=scoring_done-started-wall,plotting_s=time.monotonic()-scoring_done,decision=decision['verdict']))


def paired_batches(protocol, winners, terrains, directory):
    stage, cases = protocol['stage'], protocol['cases']
    slots = 3 if stage=='E1a' else 21
    best = {r['case']:r for r in winners}
    if (len(winners)!=slots or [c['slot'] for c in cases]!=list(range(slots))
            or set(best)!={c['case'] for c in cases}):
        raise ValueError(f'Winner coverage or physical slot order differs from {stage}')
    schedules = []
    for c in cases:
        win = best[c['case']]
        source = read(Path(win['raw'])/'contract.json')
        split = source['batch_source']
        cfg = source['settings']
        if (cfg['seed']!=c['group'] or cfg['randomization']!=c['randomization']
                or int(win['lane'])!=0 or int(win['physical_slot'])!=c['slot']
                or (split['lane_start'], split['lane_stop'])!=(c['slot'],c['slot']+1)
                or source['terrain'][0]['sha256']!=c['terrain_sha256']):
            raise ValueError(f"Winner source has a different physical slot or terrain: {win['raw']}")
        alpha,zero,origin,bias = frozen_winner(win,protocol['times_s'])
        applied = Path(cfg['oracle_knots'])
        if sha(applied)!=cfg['oracle_knots_sha256'] or (int(win['generation'])>0 and
                (Path(win['knots_npz'])!=applied or int(win['knots_lane'])!=c['slot'])):
            raise ValueError(f"Archived winner knot identity differs: {win['raw']}")
        with np.load(applied) as knots:
            np.testing.assert_array_equal(knots['times_s'],protocol['times_s'])
            np.testing.assert_array_equal(knots['offsets'][c['slot']],bias)
        schedules.append((alpha,zero,origin,bias,str(applied)))
    batches = []
    for shift in SHIFTS:
        folder = directory/stage/f'ds{shift:+d}cm'
        variants, manifests, offsets = [], [], []
        for c,(alpha,zero,origin,bias,applied_knots) in zip(cases,schedules):
            win = best[c['case']]
            layout = c['layout']
            control = layout in NONRIDGE
            terrain = Path(c['terrain']) if control else terrains[layout,shift]
            meta = read(terrain/'identity.json')
            if not control:
                original = read(Path(c['terrain'])/'identity.json')
                if (not np.isclose(meta['center_s_m'],original['center_s_m']+shift/100,rtol=0,atol=1e-9)
                        or any(meta[k]!=original[k] for k in ('height_m','length_m','half_width_m','route_sha256',
                                    'horizontal_scale_m','vertical_scale_m','oblique_s_per_lateral_m'))):
                    raise ValueError(f'Wrong shifted ridge at {terrain}')
            m, = case_manifest(folder,[dict(c,terrain=str(terrain))],win['method'],[alpha])
            if m['zero_middle']!=[zero]:
                raise ValueError(f"Winner has inconsistent Zero mask: {win['raw']}")
            manifests.append(m)
            offsets.append(bias)
            variants.append(dict(case=c['case'],layout=layout,group=c['group'],source_stage=stage,
                                 role='nominal_control' if control else 'ridge_shift',shift_cm=None if control else shift,
                                 batch_shift_cm=shift,physical_slot=c['slot'],method=win['method'],alpha=alpha,
                                 zero_middle=zero,origin=origin,terrain=str(terrain),terrain_sha256=meta['sha256'],
                                 source_winner=win['raw'],source_lane=0,source_knots=applied_knots,
                                 source_knots_lane=c['slot'],output=m['output']))
        batches.append((folder, variants, manifests, np.stack(offsets)))
    return batches


def check_matched_runtime(folder, variants, reference):
    contract = read(folder/'batch/contract.json')
    settings = contract['settings']
    # A ridge shift replaces its mesh while preserving the source batch's native packing.
    actual = {k:contract[k] for k in ('backend','identity','bodies','origin','physics_dt','lower_dt','policy_dt',
                                      'randomization_readback','delay_steps')}
    actual.update(plant=settings['plant_options'],terrain_ids=settings['lane_terrain_ids'],
                  fixed_randomization_groups=settings['fixed_randomization_groups'],
                  terrain_layout=contract['lab_physics']['terrain_layout'])
    differences = [key for key,value in actual.items() if value!=reference[key]]
    if (differences or contract['lab_physics']['num_envs']!=len(variants)
            or settings['oracle_knots_sha256']!=sha(folder/'knots.npz')):
        raise ValueError(f'E1c batch differs from registered E1 packing in {folder}: {differences}')
    for v in variants:
        c = read(Path(v['output'])/'contract.json')
        slot = v['physical_slot']
        if (c['batch_source']!=dict(output=str(folder/'batch'),lane_start=slot,lane_stop=slot+1)
                or c['settings']['case_labels']!=[v['method']] or c['settings']['alphas']!=[v['alpha']]
                or c['settings']['zero_middle']!=[v['zero_middle']]
                or c['terrain'][0]['sha256']!=v['terrain_sha256'] or c['origin']!=[v['origin']]):
            raise ValueError(f"Winner assignment, terrain, or actual origin differs at {v['output']}")


def control_samples(path, terminal):
    contract = read(path/'contract.json')
    with np.load(path/'trajectory.npz') as z:
        trace = z['lower_trace'][:,:,0].reshape(-1,11)
    valid = np.flatnonzero(trace[:,6])
    ticks = valid*round(contract['lower_dt']/contract['physics_dt'])+trace[valid,7].astype(int)
    return trace[valid,:2], ticks, terminal


def control_comparison(winner, actual):
    source = control_samples(Path(winner['raw']),winner['terminal'])
    rerun = control_samples(Path(actual['output']),actual['terminal'])
    check = compare_case(rerun,source)
    same_labels = all(int(winner[k])==int(actual[k]) for k in ('full','hold','C','goal'))
    return dict(**actual,nominal_time_s=float(winner['time_s']),nominal_full=int(winner['full']),
                nominal_hold=int(winner['hold']),nominal_goal=int(winner['goal']),nominal_C=int(winner['C']),
                nominal_V=float(winner['V']),V_delta=float(actual['V'])-float(winner['V']),**check,
                matched=bool(check['passed'] and same_labels))


def execute_matched(root, scope):
    if (scope.get('nonridge_role')!='unchanged_nominal_controls' or scope['shifts_cm']!=list(SHIFTS)
            or scope['nonridge']!={k:None for k in NONRIDGE}
            or (scope['primary_shifted_lanes'],scope['nominal_control_lanes'],scope['physical_lanes'])!=(64,32,96)):
        raise ValueError('Expected the confirmed 64 shifted ridges and 32 unchanged flat/gravel controls')
    from pcrp_phase_grid import check_gate
    check_gate(root,matched=True)
    directory = root/'e1/E1c'
    stages, nominal = [], []
    for name in ('E1a','E1b'):
        stage = root/'e1'/name
        status, protocol = read(stage/'status.json'), read(stage/'protocol.json')
        if status.get('stage')!='complete' or status.get('generation')!=10 or not archived_flat_passed(stage,status):
            raise SystemExit(f'E1c requires 10 generations and positive flat controls in {name}')
        with (stage/'best.csv').open() as f:
            best = {r['case']:r for r in csv.DictReader(f)}
        cases = protocol['cases']
        if set(best)!={c['case'] for c in cases}:
            raise ValueError(f'Incomplete best.csv in {stage}')
        winners = [dict(best[c['case']],group=c['group'],layout=c['layout'],C=int(best[c['case']]['C'])) for c in cases]
        nominal.extend(winners)
        stages.append((stage,protocol,winners))
    times = stages[0][1]['times_s']
    if times!=stages[1][1]['times_s']:
        raise ValueError('E1a/b knot schedules differ')
    directory.mkdir(exist_ok=False)
    write(directory/'confirmed_protocol.json',scope)
    terrain_map = geometry(root,directory,scope)
    batches = [(stage,*b) for stage,protocol,winners in stages for b in paired_batches(protocol,winners,terrain_map,directory)]
    variants = [v for _,_,group,_,_ in batches for v in group]
    if Counter(v['role'] for v in variants)!={'ridge_shift':64,'nominal_control':32}:
        raise ValueError('Incomplete E1c geometry/control coverage')
    csv_write(directory/'variants.csv',variants)
    bundle = read(root/'step0/protocol.json')['bundle']
    winners = {r['case']:r for r in nominal}
    scored, controls = [], []
    sys.path.insert(0,str(REPO/'docs/paper/english'))
    from score_decision_oracle import score_trajectories
    status_path = directory/'status.json'
    status = dict(stage='running',completed=0,submitted=0,batches=[],physics_wall_s=0.,sampling_s=0.,scoring_s=0.)
    for stage,folder,group,manifests,offsets in batches:
        folder.mkdir(parents=True)
        # Copy the actual E1 source batch settings, including its native origin placement.
        cfg = read(Path(group[0]['source_winner']).parent/'settings.json')
        knots = folder/'knots.npz'
        np.savez(knots,times_s=times,offsets=offsets)
        cfg.update(oracle_knots=str(knots),oracle_knots_sha256=sha(knots))
        write(folder/'settings.json',cfg)
        write(folder/'cases.json',manifests)
        status.update(active=str(folder.relative_to(directory)),submitted=status['submitted']+len(group))
        write(status_path,status)
        cmd=['/bin/bash',str(REPO/'learning/scripts/gpu_queue.sh'),LAB,'-u','learning/cli/evaluate_isaac_gpu.py',
             '--backend','lab','--seconds','10','--bundle',bundle,'--config',str(folder/'settings.json'),
             '--case-manifest',str(folder/'cases.json'),'--terrain',manifests[0]['terrain_path'],'--output',str(folder/'batch')]
        before=time.monotonic()
        with (folder/'simulation.log').open('x') as log:
            subprocess.run(cmd,cwd=REPO,env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+
                           os.pathsep+os.environ.get('PYTHONPATH','')),stdout=log,stderr=subprocess.STDOUT,check=True)
        elapsed=time.monotonic()-before
        batch_status=read(folder/'batch/status.json')
        if batch_status.get('stage')!='complete' or batch_status.get('policy_steps')!=500:
            raise ValueError(f'Incomplete physical batch at {folder}')
        check_matched_runtime(folder,group,read(stage/'matched_runtime.json'))
        for v in group:
            score, = score_trajectories(Path(v['output']),[v['method']])
            row = dict(v,**{k:val for k,val in score.items() if k!='method'})
            scored.append(row)
            if v['role']=='nominal_control':
                controls.append(control_comparison(winners[v['case']],row))
        status['batches'].append(dict(stage=stage.name,shift_cm=group[0]['batch_shift_cm'],
                                      lanes=len(group),wall_including_queue_s=elapsed,sampling_s=batch_status['elapsed_s']))
        status.update(completed=len(scored),physics_wall_s=status['physics_wall_s']+elapsed,
                      sampling_s=status['sampling_s']+batch_status['elapsed_s'],
                      scoring_s=status['scoring_s']+time.monotonic()-before-elapsed)
        write(status_path,status)
    shifted = [r for r in scored if r['role']=='ridge_shift']
    records,decision=majority_evidence(nominal,shifted)
    decision.update(matched_nominal_controls=sum(r['matched'] for r in controls),total_nominal_controls=len(controls),
                    ridge_trials=len(shifted))
    if decision['matched_nominal_controls']!=len(controls):
        decision['conditional_verdict_ignoring_control_mismatch']=decision['verdict']
        decision['verdict']='nominal_replay_mismatch'
    csv_write(directory/'scores.csv',scored)
    csv_write(directory/'nominal_controls.csv',controls)
    csv_write(directory/'case_summary.csv',records)
    write(directory/'decision.json',decision)
    before=time.monotonic()
    report(directory,records,decision,nominal,shifted,bundle,controls)
    status.update(stage='complete',active=None,plotting_s=time.monotonic()-before,decision=decision['verdict'])
    write(status_path,status)


def execute(root, scope_path):
    scope = read(scope_path)
    confirmed_layouts(scope)
    if scope.get('design')=='same_slot_winner_replay':
        execute_matched(root,scope)
    else:
        execute_legacy(root,scope_path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--confirmed-protocol',type=Path,required=True,
                   help='Use e1c_scope.json for matched same-slot winner replay and unchanged flat/gravel controls.')
    a=p.parse_args()
    execute(a.run.resolve(),a.confirmed_protocol)
