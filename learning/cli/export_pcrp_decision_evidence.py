"""Export the completed PCRP decision report, compact traces and full archive index."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import unquote


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source, dest):
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f'Required evidence is absent or empty: {source}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def copy_names(root, out, directory, names):
    for name in names:
        relative = Path(directory)/name
        copy_file(root/relative, out/relative)


def trace_index(root, out, stage, case, winner):
    folder = Path('e1')/stage/'analysis'/case/'rollouts'
    rows = read_csv(root/folder/'index.csv')
    ids = {(int(r['generation']), int(r['candidate'])) for r in rows}
    expected = {(0,i) for i in range(6)} | {(g,i) for g in range(1,11) for i in range(32)}
    if len(rows)!=326 or len({r['raw'] for r in rows})!=326 or ids!=expected or winner not in ids:
        raise ValueError(f'Incomplete candidate coverage for {stage}/{case}')
    records, global_index = [], []
    for r in rows:
        gen, candidate = int(r['generation']), int(r['candidate'])
        include = gen==0 or (gen,candidate)==winner
        stem = f'g{gen:02d}_c{candidate:02d}'
        if r['npz']!=stem+'.npz' or r['csv']!=stem+'.csv' or r['case']!=case:
            raise ValueError(f'Wrong candidate or attachment name in {folder}/index.csv')
        record = dict(r)
        raw = Path(r['raw'])/'trajectory.npz'
        if not raw.is_absolute():
            raw = root/raw
        raw_size = raw.stat().st_size
        if raw_size==0:
            raise ValueError(f'Empty raw physics archive: {raw}')
        record.update(physics_npz_source=str(raw), physics_npz_bytes=raw_size,
                      included_in_package=int(include))
        for ext in ('npz','csv'):
            relative = folder/r[ext]
            source = root/relative
            size = source.stat().st_size
            if size<=0:
                raise ValueError(f'Empty candidate attachment: {source}')
            record[ext+'_source'] = str(source)
            record[ext+'_bytes'] = size
            record[ext] = r[ext] if include else ''
            if include:
                copy_file(source,out/relative)
        records.append(record)
        global_index.append(dict(stage=stage,case=case,generation=gen,candidate=candidate,method=r['method'],
                     included_in_package=int(include),
                     compact_npz=str(folder/r['npz']) if include else '',compact_csv=str(folder/r['csv']) if include else '',
                     npz_source=record['npz_source'],npz_bytes=record['npz_bytes'],
                     csv_source=record['csv_source'],csv_bytes=record['csv_bytes'],
                     physics_npz_source=str(raw),physics_npz_bytes=record['physics_npz_bytes'],lane=r['lane']))
    write_csv(out/folder/'index.csv',records)
    return global_index


def export_e1(root, out):
    all_cases, all_rollouts, winner_files = set(), [], []
    for stage, n in [('E1a',3),('E1b',21)]:
        folder = Path('e1')/stage
        analysis = folder/'analysis'
        status = read_json(root/folder/'status.json')
        if status['stage']!='complete' or status['completed_new']!=326*n or status['generation']!=10:
            raise ValueError(f'Expected a completed ten-generation search at {folder}')
        copy_names(root,out,folder,['status.json','protocol.json','matched_runtime.json','best.csv','candidates.csv','history.json'])
        copy_names(root,out,analysis,['README.md','cases.csv','timing.json'])
        cases = read_csv(root/analysis/'cases.csv')
        best = read_csv(root/folder/'best.csv')
        protocol_cases = {r['case'] for r in read_json(root/folder/'protocol.json')['cases']}
        if (len(cases)!=n or len(best)!=n or {r['case'] for r in cases}!=protocol_cases
                or {r['case'] for r in best}!=protocol_cases or all_cases & protocol_cases):
            raise ValueError(f'Case coverage does not match {folder}/protocol.json')
        all_cases.update(protocol_cases)
        winners = {r['case']:(int(r['generation']),int(r['candidate'])) for r in best}
        for r in best:
            source = Path(r['raw'])
            if not source.is_absolute():
                source = root/source
            knots = source.parent/'knots.npz'
            member = knots.relative_to(root)
            copy_file(knots, out/member)
            winner_files.append(dict(stage=stage,case=r['case'],method=r['method'],generation=r['generation'],
                                 candidate=r['candidate'],source_lane=r['lane'],physical_slot=r['physical_slot'],
                                 compact_knots=str(member),knots_lane=r['physical_slot'],source_knots=str(knots)))
        for row in cases:
            case = row['case']
            winner = (int(row['generation']),int(row['candidate']))
            if Path(case).name!=case or int(row['candidates'])!=326 or winner!=winners[case]:
                raise ValueError(f'Winner or candidate count disagrees for {case}')
            copy_names(root,out,analysis/case,['xy.png','frontier.png','timeline.png','reward.png','frontier.csv',
                                              'selected_traces.npz','entry_events.csv','reward.csv','reward_parts.npz'])
            all_rollouts.extend(trace_index(root,out,stage,case,winner))
        page = out/analysis/'README.md'
        page.write_text(page.read_text().replace('每条候选均有单独的压缩 NPZ','完整运行目录为每条候选保存单独的压缩 NPZ')+'\n紧凑包收录每例六条热启动和最终赢家的逐候选 NPZ、逐步奖励 CSV。'
                        '候选索引列出所有原件的绝对路径、字节数与包内收录标记。'
                        '[7,824 条完整来源索引](../../rollouts.csv)\n')
    combined = Path('e1/E1b/analysis/all_cases.csv')
    combined_rows = read_csv(root/combined)
    if len(combined_rows)!=24 or {r['case'] for r in combined_rows}!=all_cases:
        raise ValueError('Incomplete combined 24-case report')
    copy_file(root/combined,out/combined)
    write_csv(out/'e1/rollouts.csv', all_rollouts)
    write_csv(out/'e1/winner_sources.csv',winner_files)
    return all_cases, len(all_rollouts), sum(r['included_in_package'] for r in all_rollouts)


def export_e2(root, out):
    folder = Path('e2/analysis')
    copy_names(root,out,'e2',['status.json','protocol.json','verification.json','geometry.json','analysis_protocol.json',
                              'paired_lanes.csv','lanes.csv'])
    copy_names(root,out,folder,['README.md','summary.json','episodes.csv','velocity_zero_references.csv','actual_adjacent_pairs.csv',
                               'flip_rates.csv','smoothed_windows.csv'])
    primary = read_csv(root/folder/'episodes.csv')
    controls = read_csv(root/folder/'velocity_zero_references.csv')
    rows = primary + controls
    summary = read_json(root/folder/'summary.json')
    if (summary.get('matched_packing') is not True or len(primary)!=1160 or len(controls)!=80
            or len({(r['source'],r['lane']) for r in rows})!=1240 or summary['checked_primary_alpha_pairs']!=96
            or Counter(r['experiment'] for r in primary)!={'E2a':840,'E2b':240,'E2c':80}
            or Counter(r['experiment'] for r in controls)!={'E2b':60,'E2c':20}):
        raise ValueError('Incomplete matched E2 condition/control coverage')
    curves = {Path(r['curve_file']) for r in rows}
    if len(curves)!=1240 or any(p.parent!=Path('curves') or p.suffix!='.npz' for p in curves):
        raise ValueError('E2 curve paths do not cover the 1240 measured lanes')
    copy_names(root,out,folder,sorted(curves))
    figures = set()
    for r in rows:
        prefix = r['experiment']+('_s'+f"{float(r['center_m']):g}" if r['center_m'] else '_flat')
        figures.update([prefix+'_completion.png',f"{prefix}_g{r['group']}_xy.png"])
        if r['center_m']:
            figures.add(f"{prefix}_g{r['group']}_state.png")
    if len(figures)!=83:
        raise ValueError('Expected 40 XY grids, 36 ridge entry panels and seven completion maps')
    copy_names(root,out,folder/'figures',sorted(figures))
    return len(rows), len(figures)


def export_robustness(root, out, cases):
    folder = Path('e1/E1c')
    scores = read_csv(root/folder/'scores.csv')
    controls = read_csv(root/folder/'nominal_controls.csv')
    summary = read_csv(root/folder/'case_summary.csv')
    key = lambda r: (r['case'],int(r['batch_shift_cm']))
    if (len(scores)!=96 or {key(r) for r in scores}!={(c,d) for c in cases for d in (-2,-1,1,2)}
            or Counter(r['role'] for r in scores)!={'ridge_shift':64,'nominal_control':32}
            or len(controls)!=32 or {key(r) for r in controls}!={key(r) for r in scores if r['role']=='nominal_control'}
            or len(summary)!=24 or {r['case'] for r in summary}!=cases):
        raise ValueError('E1c requires 64 shifted ridge trials and 32 nominal control repeats in 24 cases')
    copy_names(root,out,folder,['README.md','status.json','confirmed_protocol.json','decision.json','variants.csv','scores.csv',
                               'nominal_controls.csv','case_summary.csv'])
    copy_names(root,out,folder/'figures',[c+'.png' for c in sorted(cases)])
    return len(scores)-len(controls),len(controls)


def offline_links(root, out, page):
    # Copy two cited historical XY panels. Absolute paths to external code and raw files
    # remain readable provenance even on a machine without the original mount.
    def replace(match):
        label, raw = match.groups()
        if raw.startswith(('#','https://','http://','mailto:')):
            return match.group(0)
        path, sep, anchor = unquote(raw.strip('<>')).partition('#')
        if not path:
            return match.group(0)
        target = Path(path)
        if target.is_absolute():
            if target.is_relative_to(root):
                compact = out/target.relative_to(root)
                if compact.is_file() or compact.is_dir():
                    return f'[{label}]({os.path.relpath(compact,page.parent)}{sep}{anchor})'
            if target.suffix=='.png' and target.is_file():
                compact = out/'support'/target.parent.name/target.name
                copy_file(target,compact)
                return f'[{label}]({os.path.relpath(compact,page.parent)})'
            return f'{label}（运行主机原件 `{raw}`）'
        local = (page.parent/target).resolve()
        if not local.is_relative_to(out) or not local.exists():
            raise ValueError(f'Broken offline report link in {page.relative_to(out)}: {raw}')
        return match.group(0)
    content = re.sub(r'\[([^\]\n]*)\]\(([^)\n]+)\)',replace,page.read_text())
    page.write_text(content)


def export(run, destination):
    root, destination = run.resolve(), destination.resolve()
    if destination==root or (destination.exists() and any(destination.iterdir())):
        raise ValueError('Export destination must be a new or empty directory')
    # Verify the final decision files first, before reading per-candidate attachments.
    final = Path('decision_report')
    decision = read_json(root/final/'summary.json')
    if (decision.get('stage')!='complete' or decision.get('main_allocation')!={'step0':72,'E2':1240,'E1a':978,'E1b':6846,'E1c':96}
            or decision['diagnostic_rollouts']!=672 or decision['training_updates']!=0):
        raise ValueError('Expected the final matched PCRP decision and its separate 672-lane diagnostic')
    if not (root/final/'README.md').is_file():
        raise FileNotFoundError(root/final/'README.md')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        out = Path(tmp)/'evidence'
        copy_names(root,out,final,['README.md','summary.json'])
        copy_names(root,out,'.',['registration.json','paired_authorization.json','e1c_scope.json','decision_experiments_plan.md'])
        copy_names(root,out,'step0',['README.md','status.json','protocol.json','comparisons.csv'])
        copy_names(root,out,'step0/packing_sensitivity',['archives.md','archives.csv','complete.md','complete.csv',
                                                        'first_policy_step.md','first_policy_step.csv'])
        copy_names(root,out,'replay672',['preflight.json','timing.json'])
        copy_names(root,out,'replay672/analysis',['README.md','summary.json','identity.json','lane_comparisons.csv',
                   'layout_alpha.csv','three_way_overlap.csv','early_summary.csv','early_signals.csv'])
        for folder, expected in [('step0/figures',6),('replay672/analysis/figures',112)]:
            panels = sorted((root/folder).glob('*.png'))
            if len(panels)!=expected:
                raise ValueError(f'{folder}: expected {expected} paired XY panels')
            copy_names(root,out,folder,[p.name for p in panels])
        e2_lanes, e2_figures = export_e2(root,out)
        cases, candidates, selected = export_e1(root,out)
        ridge, controls = export_robustness(root,out,cases)
        (out/'README.md').write_text(
            '# PCRP 判决实验完整报告与紧凑证据\n\n'
            '[综合判决与逐案例 XY](decision_report/README.md) · [24 案例可达前沿、时序与奖励](e1/E1b/analysis/README.md) · '
            '[E2 相位与初速](e2/analysis/README.md) · [E1c 脊位偏移与名义对照](e1/E1c/README.md)\n\n'
            '[第 0 步原始判定](step0/README.md) · [672 环境原顺序复验](replay672/analysis/README.md) · '
            '[原方案](decision_experiments_plan.md) · [配对授权](paired_authorization.json) · [E1c 范围](e1c_scope.json)\n\n'
            f'E1 共 {candidates:,} 条逐候选记录。每案六条热启动与最终赢家去重后，包内有 {selected} 对派生 NPZ 和逐步 CSV。'
            '全量索引按条记录派生 NPZ、逐步 CSV 及完整物理轨迹原件的绝对路径与字节数，含包内相对文件名。'
            '[全量索引](e1/rollouts.csv) · [24 个赢家的完整偏置结点与源批槽位](e1/winner_sources.csv)\n\n'
            f'E2 收录全部 {e2_lanes:,} 份曲线 NPZ 和 {e2_figures} 张图。'
            '实际批次及物理槽位可由各车道的来源记录与相邻比较表追溯。'
            '完整物理轨迹保存在运行主机。CSV/JSON 中绝对路径指向运行主机，本包内报告使用相对链接。\n\n'
            f'完整运行目录为 `{root}`。[包内文件清单](evidence_manifest.json)\n')
        manifest_path=out/'evidence_manifest.json'
        manifest_path.write_text('{}')
        decision_page=out/final/'README.md'
        decision_page.write_text(decision_page.read_text()+'\n[7,824 条逐候选 NPZ 和逐步 CSV 的原件路径、字节及包内标记](../e1/rollouts.csv)'
                                 ' · [24 个赢家的全时域偏置](../e1/winner_sources.csv)\n')
        for page in out.rglob('*.md'):
            offline_links(root,out,page)
        files = {p.relative_to(out).as_posix():p.stat().st_size for p in sorted(out.rglob('*')) if p.is_file() and p!=manifest_path}
        manifest = dict(stage='complete',source_run=str(root),e1_candidates=candidates,e1_included_candidates=selected,
                        e2_physical_lanes=e2_lanes,e2_figures=e2_figures,e1c_ridge_trials=ridge,e1c_controls=controls,
                        indexed_files_excluding_manifest=len(files),total_bytes_excluding_manifest=sum(files.values()),files=files)
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        os.replace(out,destination)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=export(args.run,args.out)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))
