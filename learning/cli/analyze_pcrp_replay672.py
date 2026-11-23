"""Compare a single replay in the entire original 672-lane scene against both archived packings."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pcrp_replay_gate import KEYS, LABELS, REPO, read, compare_case
sys.path.insert(0,str(REPO/'docs/paper/english'))
from analyze_matched_controls import read_case

FIELDS = ('nominal','target','filtered','offsets','lower_residual','final_command','measurement',
          'body_state','contact_force','pose','xy','progress','controller_output','controller_gains')
TAGS = ('new672_vs_original672','prior36_vs_new672','prior36_vs_original672')


def dump_rows(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def inspect_sources(a,b,small):
    keys=('identity','backend','origin','terrain','bodies','physics_dt','lower_dt','policy_dt',
          'randomization_readback','delay_steps')
    differences=[key for key in keys if a[key]!=b[key]]
    for k in a['settings'].keys()|b['settings'].keys():
        x,y=a['settings'].get(k),b['settings'].get(k)
        if k=='plant_options' and small:
            x={p:v for p,v in x.items() if p!='replay_clone_origins'}
            y={p:v for p,v in y.items() if p!='replay_clone_origins'}
        if x!=y:
            differences.append('settings.'+k)
    for k in a['lab_physics'].keys()|b['lab_physics'].keys():
        x,y=a['lab_physics'].get(k),b['lab_physics'].get(k)
        if small and k=='terrain_asset_count':
            continue
        if small and k=='terrain_layout':
            x={p:v for p,v in x.items() if p!='template_offsets'}
            y={p:v for p,v in y.items() if p!='template_offsets'}
        if x!=y:
            differences.append('lab_physics.'+k)
    return differences


def samples(folder):
    if read(folder/'status.json')['policy_steps']!=500:
        raise ValueError(f'Incomplete evaluation: {folder}')
    contract,lanes=read_case(folder,LABELS)
    motion=[]
    for row,trace,t,progress,_ in lanes:
        ticks=np.rint(t/contract['physics_dt']).astype(int)
        motion.append((trace[:,:2],ticks,row['terminal']))
    return dict(folder=folder,contract=contract,lanes=lanes,motion=motion)


def field_comparisons(key,name,a,b):
    """The first recorded lower clock is 5 ms; no reset-state or 5 kHz command trace is inferred."""
    records=[]
    with np.load(a['folder']/'trajectory.npz') as left,np.load(b['folder']/'trajectory.npz') as right:
        for field in FIELDS:
            if field=='xy':
                x,y=(data['lower_trace'].reshape(-1,len(LABELS),11)[:,:,:2] for data in (left,right))
            else:
                x,y=left[field],right[field]
            for lane in range(len(a['motion'])):
                ta,tb=a['motion'][lane][1],b['motion'][lane][1]
                size=min(len(ta),len(tb))
                xa,xb=x[:size,lane].reshape(size,-1),y[:size,lane].reshape(size,-1)
                equal=(xa==xb)|(np.isnan(xa)&np.isnan(xb))
                first=np.flatnonzero(~equal.all(axis=1))
                records.append(dict(case=key,comparison=name,lane=LABELS[lane],field=field,
                                    common_samples=size,first_changed_sample=int(first[0]+1) if len(first) else None,
                                    first_changed_tick_a=int(ta[first[0]]) if len(first) else None,
                                    first_changed_tick_b=int(tb[first[0]]) if len(first) else None,
                                    frame1_max_abs_in_native_units=float(np.abs(xa[0].astype(float)-xb[0]).max())))
    return records


def draw(key,inputs,route,output):
    """Each pane retains its terminal point with the same metric scales across all six roles."""
    fig,axes=plt.subplots(2,3,figsize=(9,6.8),layout='constrained',sharex=True,sharey=True)
    maximum=max(float(lane[3].max()) for _,s in inputs.values() for lane in s['lanes'])
    ref=route['xy'][route['s']<=maximum+.2]
    mesh=inputs['archive'][1]['contract']['terrain'][0]
    for i,(ax,label) in enumerate(zip(axes.flat,LABELS)):
        ax.plot(*ref.T,c='0.45',ls=':',lw=1.2,label='Original path')
        if mesh.get('height_m',0):
            sr=route['s']
            ridge=route['xy'][np.abs(sr-mesh['center_s_m'])<=mesh['length_m']/2]
            ax.plot(*ridge.T,c='sandybrown',lw=4,alpha=.65,label='Ridge on reference')
        for tag,(caption,s) in inputs.items():
            xy,t,end=s['motion'][i]
            color,style,width,mark={'archive':('#242424','-',2.0,'x'),'new':('#2978B5','--',1.6,'+'),
                                    'small':('#D55E00','-.',1.,'1')}[tag]
            ax.plot(*xy.T,c=color,ls=style,lw=width,label=caption)
            ax.plot(*xy[-1],c=color,marker='o' if end=='full_window' else mark,ms=6,fillstyle='none')
        ax.set(title=label,aspect='equal',xlabel='World x (m)',ylabel='World y (m)')
        ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f'{key} | H75 / Local300 | circle: full 10 s; crosses: early endpoints')
    fig.legend(handles,labels,loc='outside lower center',ncols=len(labels),fontsize=8)
    fig.savefig(output,dpi=105)
    plt.close(fig)


def summary_rows(rows):
    answer={}
    for tag, rr in rows.items():
        answer[tag]=dict(n=len(rr),passed=sum(r['passed'] for r in rr),
                         bitwise=sum(r['xy_bitwise'] for r in rr),close=sum(r['xy_close'] for r in rr),
                         same_ticks=sum(r['times_equal'] for r in rr),same_terminal=sum(r['terminal_equal'] for r in rr),
                         tested_config_matches=sum(r['identity_match'] for r in rr),
                         max_xy_m=max(r['max_xy_distance_m'] for r in rr))
        for metric in ('full','hold','C'):
            answer[tag][metric+'_flip']=sum(r[metric+'_a']!=r[metric+'_b'] for r in rr)
            answer[tag][metric+'_a']=sum(r[metric+'_a'] for r in rr)
            answer[tag][metric+'_b']=sum(r[metric+'_b'] for r in rr)
    return answer


def compare(root):
    start=time.monotonic()
    stage=root/'replay672'
    protocol=read(root/'step0/protocol.json')
    original=Path(protocol['archive'])
    manifest=read(original/'batch_000.json')
    name_order=[Path(c['output']).name for c in manifest]
    if len(set(name_order))!=112 or not set(KEYS)<=set(name_order):
        raise ValueError('Expected all 112 distinct original cases and the six overlapping cases')
    archive_contract=read(Path(protocol['original_batch'])/'contract.json')
    new_contract=read(stage/'batch/contract.json')
    if any(c['settings']['num_envs']!=672 for c in (archive_contract,new_contract)):
        raise ValueError('Invalid physical lane count in full-packing source or replay')
    batch_diff=inspect_sources(new_contract,archive_contract,False)
    out=stage/'analysis'
    (out/'figures').mkdir(parents=True,exist_ok=True)
    with np.load(Path(protocol['bundle'])/'route.npz') as z:
        route={k:z[k] for k in ('xy','s')}
    records=defaultdict(list)
    fields=[]
    identity=[]
    plotting=0.
    for j,(case,key) in enumerate(zip(manifest,name_order)):
        old=samples(Path(case['output']))
        new=samples(stage/key)
        p=[(TAGS[0],new,old)]
        inputs={'archive':('Original 672',old),'new':('New 672',new)}
        if key in KEYS:
            smaller=samples(root/'step0/repeat_1'/key)
            p.extend(((TAGS[1],smaller,new),(TAGS[2],smaller,old)))
            inputs['small']=('Prior 36',smaller)
        for name,a,b in p:
            ca,cb=a['contract'],b['contract']
            mismatches=inspect_sources(ca,cb,name!=TAGS[0])
            lo_a,lo_b=ca['batch_source']['lane_start'],cb['batch_source']['lane_start']
            if name==TAGS[0] and (lo_a!=6*j or lo_b!=6*j):
                mismatches.append('batch_source.lane_start')
            identity.append(dict(case=key,comparison=name,field_differences=mismatches,
                                 code_a=ca['code_revision'],code_b=cb['code_revision'],slot_start_a=lo_a,slot_start_b=lo_b))
            fields.extend(field_comparisons(key,name,a,b))
            for lane,label in enumerate(LABELS):
                ma,mb=a['lanes'][lane][0],b['lanes'][lane][0]
                stats=compare_case(a['motion'][lane],b['motion'][lane])
                records[name].append(dict(case=key,layout=key.split('_v2.6')[0],group=case['seed'],
                    lane=label,slot_a=lo_a+lane,slot_b=lo_b+lane,
                    comparison=name,identity_match=not mismatches,source_a=str(a['folder']),source_b=str(b['folder']),
                    **stats,**{f'{k}_{suffix}':rec[k] for k in ('full','hold','C','peak_path_m','speed_rmse_m_s')
                               for suffix,rec in (('a',ma),('b',mb))}))
        mark=time.monotonic()
        draw(key,inputs,route,out/'figures'/f'{key}.png')
        plotting+=time.monotonic()-mark
    all_rows=sum(records.values(),[])
    dump_rows(out/'lane_comparisons.csv',all_rows)
    dump_rows(out/'early_signals.csv',fields)
    (out/'identity.json').write_text(json.dumps(dict(full_batch_differences=batch_diff,cases=identity),indent=2)+'\n')
    summaries=summary_rows(records)
    overlap=[r for r in records[TAGS[0]] if r['case'] in KEYS]
    summaries['overlap_new672_vs_original672']=summary_rows({'overlap':overlap})['overlap']
    overlap_index={tag:{(r['case'],r['lane']):r for r in records[tag] if r['case'] in KEYS} for tag in TAGS}
    triangulation=[]
    for key,full in overlap_index[TAGS[0]].items():
        small_new=overlap_index[TAGS[1]][key]
        small_old=overlap_index[TAGS[2]][key]
        triangulation.append(dict(case=key[0],lane=key[1],new672_matches_original=full['passed'],
                                  prior36_matches_new672=small_new['passed'],prior36_matches_original=small_old['passed'],
                                  original_full=full['full_b'],new672_full=full['full_a'],prior36_full=small_old['full_a'],
                                  original_C=full['C_b'],new672_C=full['C_a'],prior36_C=small_old['C_a']))
    dump_rows(out/'three_way_overlap.csv',triangulation)
    recovered=sum(r['new672_matches_original'] and not r['prior36_matches_original'] for r in triangulation)
    previously_changed=sum(not r['prior36_matches_original'] for r in triangulation)
    main=summaries[TAGS[0]]
    passed=not batch_diff and main['tested_config_matches']==672 and main['passed']==672
    if batch_diff or main['tested_config_matches']!=672:
        verdict='setup_mismatch'
    elif passed:
        verdict='restored_original_packing_reproduces_archive'
    else:
        verdict='archive_mismatch_persists_in_original_packing'
    differences={tag:dict(C10=summaries[tag]['full_flip'],hold=summaries[tag]['hold_flip'],C=summaries[tag]['C_flip']) for tag in summaries}
    early=[]
    for tag in TAGS:
        for field in FIELDS:
            group=[r for r in fields if r['field']==field and r['comparison']==tag]
            changed=[r['first_changed_sample'] for r in group if r['first_changed_sample'] is not None]
            early.append(dict(comparison=tag,field=field,n=len(group),common_prefix_diff=len(changed),
                              first_sample_changed=sum(k==1 for k in changed),earliest_sample=min(changed,default=None),
                              latest_first_change_sample=max(changed,default=None),
                              max_first_frame_abs=max(r['frame1_max_abs_in_native_units'] for r in group)))
    dump_rows(out/'early_summary.csv',early)
    by_condition=[]
    for (layout,lane),rr in cluster(records[TAGS[0]],('layout','lane')).items():
        cell=summary_rows({'cell':rr})['cell']
        by_condition.append(dict(layout=layout,lane=lane,**cell))
    dump_rows(out/'layout_alpha.csv',by_condition)
    lines=['# 完整 672 环境回放','',
           '## 配对实际与原始参考 XY','',
           '每格覆盖同一物理组的五档 H75 及 Zero。重叠工况另叠加先前 36 环境回放。圆点为满窗，叉号为真实终止。','',
           '|布局|'+'|'.join(map(str,range(401,409)))+'|','|---|'+'---|'*8]
    for layout in dict.fromkeys(r['layout'] for r in records[TAGS[0]]):
        cells={r['group']:r['case'] for r in records[TAGS[0]] if r['layout']==layout}
        lines.append('|'+layout+'|'+'|'.join(f'[XY](figures/{cells[g]}.png)' for g in range(401,409))+'|')
    texts={'restored_original_packing_reproduces_archive':'原车道、顺序及场景规模恢复后，这一次完整回放复现了被检验的原档轨迹。',
           'archive_mismatch_persists_in_original_packing':'恢复原来的批次规模、车道和排列后，这一次完整回放仍有轨迹超过预先登记的允许偏差。',
           'setup_mismatch':'完整回放的合同存在差异，需先解释差异才能判断同配置复现。'}
    lines+=['','## 判定','','**'+texts[verdict]+'**','',
            '|比较（左为本行数据 A）|n|XY 逐位同|XY 容差内|终止时刻同|终止类型同|全部满足|满窗 A/B，翻转|保持 A/B，翻转|任务 C A/B，翻转|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for tag,v in summaries.items():
        count=lambda k:f"{v[k+'_a']}/{v[k+'_b']}，{v[k+'_flip']}"
        lines.append(f"|{tag}|{v['n']}|{v['bitwise']}|{v['close']}|{v['same_ticks']}|{v['same_terminal']}|{v['passed']}|{count('full')}|{count('hold')}|{count('C')}|")
    lines+=['',f"原档代码 {archive_contract['code_revision'][:8]}，本次代码 {new_contract['code_revision'][:8]}；"
            f"原档与本次均使用 672 个物理环境与 14 个地形资源。位置、车道顺序、固定物理组、延迟、测得材料与质量、"
            f"求解设置和完整解析配置的差异见 [合同核验](identity.json)。每个对照逐项比较全部有效 200 Hz XY，"
            f"采用 1e-8 m + 1e-5 × 对照坐标绝对值的容差；每个观测物理 tick 和终止原因需相同。",'',
            '672 对覆盖 14 个布局、8 个已用物理组及 6 条固定策略。最后一行是第一行的 36 对子集。先前两次 36 环境重复的被记录数组逐位一致，表中统一用第一次做配对。', '',
            '## 首个样点与差异出现次序','',
            '每个控制周期 5 ms，XY 使用原有去车道平移的世界坐标。下表统计第一个已记录样点出现数值差异的车道。各字段使用自身原始单位，200 Hz 记录不能判定同一周期内哪个物理子步最先产生差异。', '',
            '|输出量|新 672 / 原 672（每组 672）|先前 36 / 新 672（每组 36）|先前 36 / 原 672（每组 36）|','|---|---:|---:|---:|']
    for k in FIELDS:
        lines.append('|'+k+'|'+'|'.join(str(next(r['first_sample_changed'] for r in early if r['field']==k and r['comparison']==t)) for t in TAGS)+'|')
    lines+=['',f"先前 36 环境与原档有 {previously_changed} 条超过回放门槛，其中恢复完整批次后满足门槛的为 {recovered} 条。"
            "这一配对检验改变了全局环境数、车道排列和地形资源集合。整批规模与任一特定求解或网络计算因素的作用不能在本次数据中拆分。"
            " [36 个工况的三方逐条对照](three_way_overlap.csv)",'',
            '最初物理状态的身份由保存的初态文件、运行合同和逐车道回读核对；第一个记录点已执行了一个控制周期。不同批次结果可以定位到已记录信号首次不同的位置，归因到单一 GPU 内核或接触事件还需要该层的证据。', '',
            '[每个车道、真实终止时刻和类型、C10 / 保持 / C、XY 最大误差](lane_comparisons.csv) · [按布局与档位汇总](layout_alpha.csv) · [逐车道逐信号的首差](early_signals.csv) · [首次差异统计](early_summary.csv) · [运行身份与排序](identity.json)']
    if (stage/'timing.json').is_file():
        lines+=['','[完整场景的实际墙钟时间与采样用时](../timing.json)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    info=dict(stage='complete',passed=bool(passed),matched_lanes=main['passed'],total_lanes=672,
              c10_flips=main['full_flip'],hold_flips=main['hold_flip'],task_c_flips=main['C_flip'],
              verdict=verdict,comparisons=summaries,outcome_flips=differences,
              prior36_disagreements=previously_changed,restored_after_full_packing=recovered,
              replay_code=new_contract['code_revision'],archive_code=archive_contract['code_revision'],
              full_batch_contract_difference_fields=batch_diff,panels=len(manifest),report=str(out/'README.md'),
              plotting_s=plotting,analysis_total_s=time.monotonic()-start)
    (out/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def cluster(rows,fields):
    result=defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in fields)].append(row)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(compare(args.run.resolve()),ensure_ascii=False))
