"""Combine the completed matched PCRP decision experiments into one evidence index."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


E1_TEXT = {
    'route_B':'达到路线 B 的控制器差距与偏移稳健性门槛。',
    'route_A_limited_search':'触发路线 A；在本接口和本次搜索下，大多数非平地案例未找到通过 C 的解。',
    'case_by_case':'逐案例报告；整体结果未触发路线 A 或 B。',
    'conditional_majority_or_missing_shifts':'±1 cm 多数判定受统计口径或缺少的碎石偏移数据影响，路线保留为条件结论。',
    'nominal_replay_mismatch':'原地形同槽对照出现复验差异，E1 路线待核查。',
    'halt_flat_control':'平地阳性对照未通过，E1 路线待核查。'}
E2_TEXT = {
    'phase_dominated':'相位主导，窄 α 窗口应解释为接触相位敏感性。',
    'alpha_structure':'α 结构成立，平滑后的端点反转仍存在。',
    'mixed':'两类门槛均未触发，偏好与相位的影响按混合结果报告。',
    'degenerate_alpha_benchmark':'相邻 α 基准退化，无法依据比例作出明确判决。'}


def joint_readout(e1, e2, flat_review, sensitivities_disagree):
    statements = [E1_TEXT[e1]]
    if flat_review:
        statements.append(f'平地阴性对照出现翻转，E2 主指标分类为 {e2}；接触机理解释需先核查确定性和实现。')
        return statements
    statements.append(E2_TEXT[e2])
    if sensitivities_disagree:
        statements.append('名义中心与其他 α 基准给出的分类有差异，E2 解释限于已声明的主基准。')
    if e1=='route_B':
        statements.append('路线 B 的后续方向为地形预览与学习。预览量的实车可得性以及启发式修正所需的动作通道证据另行确认。')
    elif e1=='route_A_limited_search':
        statements.append('路线 A 将平滑弯道上的取舍作为主结果，崎岖路面报告本次搜索得到的可达前沿。')
    return statements


def summarize(root):
    out=root/'decision_report'
    e2=root/'e2'; e2a=e2/'analysis'; rc=root/'e1/E1c'
    counts={}
    timing={}
    flags=read(root/'step0/status.json')
    replay=read(root/'replay672/analysis/summary.json')
    rt=read(root/'replay672/timing.json')
    require(flags['stage']=='complete' and replay['passed'] and replay['matched_lanes']==672
            and rt['stage']=='physics_complete', 'Incomplete gate or original-packing replay')
    counts['step0']=flags['completed_rollouts']
    diagnostic=rt['lanes']
    timing['第0步']=dict(batches=2,lanes=counts['step0'],wall_s=sum(b['wall_s'] for b in flags['repeats']),
                       sampling_s=sum(b['sampling_s'] for b in flags['repeats']),analysis_s=flags['analysis_plot_s'])
    timing['原排列诊断（单列）']=dict(batches=1,lanes=diagnostic,wall_s=rt['wall_s'],sampling_s=rt['sampling_s'],
                               analysis_s=replay['analysis_total_s'],plotting_s=replay['plotting_s'])
    def link(file,label):
        require(file.is_file(),f'Missing report artifact: {file}')
        return f'[{label}]({os.path.relpath(file,out)})'

    summaries=[]
    winners={}
    for name,expected in [('E1a',3),('E1b',21)]:
        stage=root/'e1'/name
        state=read(stage/'status.json'); analysis=stage/'analysis'
        require(state['stage']=='complete' and state['generation']==10
                and state['completed_new']==state['completed_total']==326*expected, f'Incomplete {name}')
        counts[name]=state['completed_new']
        batches=read(analysis/'timing.json')
        require(len(batches['batches'])==326 and sum(b['lanes'] for b in batches['batches'])==counts[name],
                f'{name} timed physical batch count mismatch')
        timing[name]=dict(batches=326,lanes=counts[name],wall_s=sum(b['wall_s'] for b in batches['batches']),
                          sampling_s=sum(b['sampling_s'] for b in batches['batches']),
                          analysis_s=batches['analysis_and_plot_s'],
                          scoring_and_archive_s=sum(b['scoring_and_archive_s'] for b in batches['batches']))
        best={r['case']:r for r in table(stage/'best.csv')}
        cases=table(analysis/'cases.csv')
        require(len(best)==len(cases)==expected and set(best)=={r['case'] for r in cases},f'Unmatched {name} cases')
        for row in cases:
            key=row['case'];win=best[key]
            require(key not in winners and int(row['candidates'])==326 and row['raw']==win['raw']
                    and int(row['lane'])==int(win['lane']) and row['C']==win['C'], f'Winner mismatch {key}')
            winners[key]=win
            summaries.append(dict(case=key,stage=name,case_dir=analysis/key,win=win,metrics=row))
    control_status=read(rc/'status.json'); decision=read(rc/'decision.json')
    shifted_rows=table(rc/'scores.csv'); repeats=table(rc/'nominal_controls.csv')
    require(control_status['stage']=='complete' and len(shifted_rows)==96 and len(repeats)==32
            and Counter(r['role'] for r in shifted_rows)==dict(ridge_shift=64,nominal_control=32),
            'E1c requires 64 ridge shifts and 32 unchanged nominal controls')
    by_case={r['case']:r for r in table(rc/'case_summary.csv')}
    require(len(winners)==24 and by_case.keys()==winners.keys(), 'Incomplete E1c coverage')
    require(all(int(w['C'])==int(by_case[k]['nominal_C']) for k,w in winners.items())
            and sum(int(w['C']) for k,w in winners.items() if by_case[k]['layout']!='flat_original')==decision['nonflat_nominal_pass']
            and sum(int(w['C']) for k,w in winners.items() if by_case[k]['layout']=='gravel010')==decision['unmeasured_eligible'],
            'E1 winners differ from nominal E1c scores or unmeasured gravel denominator')
    counts['E1c']=control_status['completed']
    timing['E1c']=dict(batches=len(control_status['batches']),lanes=counts['E1c'],
                       wall_s=control_status['physics_wall_s'],sampling_s=control_status['sampling_s'],
                       analysis_s=control_status['scoring_s']+control_status['plotting_s'])
    counts['E2']=read(e2/'status.json')['completed_new']
    es=read(e2a/'summary.json'); ep=read(e2/'status.json')
    require(ep['stage']=='physics_complete' and es['matched_packing'] and es['physics_lanes']==counts['E2']
            and (es['planned_rows'],es['velocity_zero_references'],es['archive_rows'])==(1160,80,0),
            'Missing completed E2 matched design')
    for axis,num,size in [('ds',35,24),('dv',25,16)]:
        batches=[b for b in ep['batches'] if b['name'].startswith(axis)]
        require(len(batches)==num and all(b['new']==size for b in batches),f'Incomplete E2 {axis}')
        timing['E2 '+axis]=dict(batches=num,lanes=num*size,wall_s=sum(b['wall_s'] for b in batches),
                                 sampling_s=sum(b['sampling_s'] for b in batches),analysis_s=None)
    e2_analysis=es['scoring_s']+es['plotting_s']
    require(counts==dict(step0=72,E1a=978,E1b=6846,E1c=96,E2=1240)
            and diagnostic==672 and sum(counts.values())<=read(root/'paired_authorization.json')['main_experiment_rollout_cap'],
            'Final physical counts do not match the registered allocation')
    fa_rows=table(e2a/'flip_rates.csv')
    selected={r['metric']:r for r in fa_rows if r['subset']=='overall'}
    statements=joint_readout(decision['verdict'],es['decisions']['nominal_centers'],
                             es['flat_control_needs_review'],es['sensitivity_changes_verdict'])
    headline=(f"非平地名义案例中，搜索档案含通过任务 C 的解 **{decision['nonflat_nominal_pass']}/20**；"
              f"单脊 **{decision['ridge_nominal_pass']}/16**。")
    lines=['# PCRP 判决实验','',headline+' '+' '.join(statements),'','## 各案例的实际／参考 XY 与可达前沿','',
           '参考路线为半径 1.8 m 的 90° 左弯，原速度请求 2.6 m/s。每个案例包含匹配的 H75 五档、Zero 与 320 个 CEM 候选。表中端点属于最低 V 候选；XY 标出提前终止，前沿仅取满 10 秒且通过末秒保持的轨迹。','',
           '|布局 / 物理组|实际／参考 XY|可达前沿|入障时序、奖励|最佳候选|V|10 s / 保持 / A / B / C|实测终点|偏移与对照 XY|候选档案|',
           '|---|---|---|---|---|---:|---|---|---|---|']
    case_data=[]
    for s in summaries:
        key,d,r=s['case'],s['case_dir'],s['win']
        figures=[link(d/'xy.png','XY'),link(d/'frontier.png','Ep/Ev'),
                 link(d/'timeline.png','时序')+' · '+link(d/'reward.png','奖励'),
                 link(rc/'figures'/f'{key}.png','E1c')]
        values='/'.join(r[k] for k in ('full','hold','A','B','C'))
        lines.append(f"|{key}|{figures[0]}|{figures[1]}|{figures[2]}|{r['method']}|{float(r['V']):.4f}|{values}|"
                     f"{float(r['time_s']):.4f} s · {r['terminal']}|{figures[3]}|{link(d/'rollouts/index.csv','326 条 NPZ / 逐步 CSV')}|")
        case_data.append(dict(case=key,stage=s['stage'],method=r['method'],generation=int(r['generation']),
                              V=float(r['V']),terminal=r['terminal'],time_s=float(r['time_s']),
                              outcomes={k:int(r[k]) for k in ('full','hold','A','B','C')},
                              frontier_count=int(s['metrics']['pareto']),robustness=by_case[key]))
    successful_flat=sum(int(w['C']) for k,w in winners.items() if k.startswith('flat_original'))
    lines+=['','## E1 可行性与偏移稳定性','',
            f"平地阳性对照通过 {successful_flat}/4。16 个单脊案例各做 −2、−1、+1、+2 cm 四次偏移；"
            f"原地形的 32 次平地／碎石对照与名义轨迹相符 {decision['matched_nominal_controls']}/32。"
            f"名义通过 C 且可测的非平地案例中，±1 cm 两侧均通过 {decision['eligible_two_of_two']} 例、"
            f"仅一侧通过 {decision['eligible_one_of_two']} 例、两侧均失败 {decision['eligible_zero_of_two']} 例。"
            f"另有 {decision['unmeasured_eligible']} 个名义通过案例缺少单脊偏移定义。",'',
            '原定门槛为非平地名义通过至少 6/20 且 ±1 cm 仍有过半通过，或平地阳性对照通过且非平地至多 1/20 通过。其余情况保留逐案例结果。分母和未测界限如下。','',
            '|±1 cm 统计口径|实测通过 / 实测分母|全体名义通过分母|尚未测得的最大新增通过数|','|---|---:|---:|---:|']
    for key,label in [('pooled_trials','合并两侧轨迹'),('both_shifts_per_case','两侧均通过的案例'),('minus1','−1 cm'),('plus1','+1 cm')]:
        v=decision['readings'][key]
        lines.append(f"|{label}|{v['passes']}/{v['observed']}|{v['all_nominal_denominator']}|{v['possible_additional']}|")
    lines+=['',link(rc/'scores.csv','96 条终点与任务分数')+' · '+link(rc/'nominal_controls.csv','32 条原地形逐轨迹核对')+' · '+
            link(rc/'decision.json','判决及缺失界限')+' · '+link(rc/'README.md','四个脊偏移的逐案例结果'),'',
            '## E2 相位敏感性','',
            '完成指标 C10 指满 10 秒。主基准取三个名义脊位、8 个物理组和相邻 α 档。fₛ 或 fᵥ ≥ 0.5 fα 触发相位主导；两者均 ≤ 0.2 fα 且 3.9/4.5 m 的平滑端点反转保留时，判为 α 结构成立。','',
            '|实测量|翻转数 / 相邻对数|翻转率|物理组 bootstrap 描述性 95% 区间|','|---|---:|---:|---|']
    for key in ('f_alpha','f_s','f_v','f_v_flat'):
        r=selected[key]
        lines.append(f"|{key}|{r['flips']}/{r['pairs']}|{float(r['fraction']):.3f}|{float(r['bootstrap_low']):.3f}–{float(r['bootstrap_high']):.3f}|")
    lines+=['',f"平滑后的端点反转：{'保留' if es['smoothed_endpoint_reversal'] else '未保留'}。平地阴性对照"
            f"{'出现翻转，需核查' if es['flat_control_needs_review'] else '未观察到翻转'}。"
            f"组匹配基准分类为 {es['decisions']['matched_groups']}，全扰动基准为 {es['decisions']['all_shifts']}。",'',
            '|网格|完成热图|各物理组的同场景 XY|','|---|---|---|']
    for arm,centers,groups in [('E2a',(3.9,4.2,4.5),range(401,409)),('E2b',(3.9,4.2,4.5),range(401,405)),('E2c',(None,),range(401,405))]:
        for c in centers:
            prefix=arm+('_flat' if c is None else f'_s{c:g}')
            links=[link(e2a/'figures'/f'{prefix}_g{g}_xy.png',str(g)) for g in groups]
            lines.append(f"|{prefix}|{link(e2a/'figures'/f'{prefix}_completion.png','完成率')}|{' · '.join(links)}|")
    lines+=['',link(e2a/'README.md','完整网格、入障状态与逐步 H')+' · '+link(e2a/'episodes.csv','1160 个主工况')+' · '+
            link(e2a/'velocity_zero_references.csv','80 个初速轴新零点')+' · '+link(e2a/'actual_adjacent_pairs.csv','实际相邻配对')+
            ' · '+link(e2a/'flip_rates.csv','全体与条件翻转率')+' · '+link(e2a/'smoothed_windows.csv','平滑窗口'),'',
            '## 物理预算与耗时','',
            f"正式设计共 {sum(counts.values()):,} 条，含第 0 步的 {counts['step0']} 条。另列原排列诊断 {diagnostic} 条。"
            f"两项实际合计 {sum(counts.values())+diagnostic:,} 条，训练更新数 0。",'',
            '|阶段／轴|物理回合|批数|批次墙钟和 (s)|采样 (s)|评分／出图 (s)|','|---|---:|---:|---:|---:|---:|']
    for name in ('第0步','原排列诊断（单列）','E2 ds','E2 dv','E1a','E1b','E1c'):
        r=timing[name]; analysis=r['analysis_s']
        if r.get('scoring_and_archive_s') is not None:
            analysis+=r['scoring_and_archive_s']
        lines.append(f"|{name}|{r['lanes']}|{r['batches']}|{r['wall_s']:.1f}|{r['sampling_s']:.1f}|"
                     +('见两轴合并' if name.startswith('E2') else ('未单列' if analysis is None else f'{analysis:.1f}'))+'|')
    lines+=['',f'E2 两轴统一评分与出图 {e2_analysis:.1f} s。E2 和 E1 各物理批次的墙钟含共享 GPU 排队、初始化、采样、导出和退出；阶段可以交错。表中按批累加，采样列取运行器实测。','',
            link(root/'step0/README.md','第 0 步 2 × 36 复验')+' · '+link(root/'replay672/analysis/README.md','另列的原排列 672 条复验')+' · '+
            link(root/'e1/E1a/analysis/README.md','E1a 全候选与奖励')+' · '+link(root/'e1/E1b/analysis/README.md','E1b 全候选与总索引')]
    summary=dict(stage='complete',main_rollouts=sum(counts.values()),diagnostic_rollouts=diagnostic,
                 all_rollouts=sum(counts.values())+diagnostic,training_updates=0,main_allocation=counts,
                 role='Previously examined physical groups, paired DEV; E1 privileged case-specific search',
                 E1=decision,E2=es,conclusions=statements,
                 verdict_zh=f"非平地名义 {decision['nonflat_nominal_pass']}/20 通过任务 C。"+' '.join(statements[:2]),
                 flat_positive_pass=successful_flat,flat_positive_n=4,
                 cases=case_data,flip_rates=fa_rows,timing=timing,e2_scoring_plotting_s=e2_analysis)
    out.mkdir(exist_ok=True)
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    summarize(args.run.resolve())


if __name__=='__main__':
    main()
