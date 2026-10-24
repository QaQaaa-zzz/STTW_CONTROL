"""Exhaustive paired entering-state audit; no policy selection or new physics."""
import argparse,csv,json,pickle
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=a.run.resolve();out=root/'paired_risk_audit';out.mkdir(exist_ok=False)
    file=root/'joint_baseline_supplement/evaluation.json';j=json.loads(file.read_text())
    source=root/'analysis/analysis.json';primary=json.loads(source.read_text())
    rows=j['cases'];actual=primary['cases'];receipts={str(x):digest(x) for x in [file,source,Path(__file__).resolve()]}
    from sttw_control.path_command_policy import actor_feature_spec
    spec=actor_feature_spec(json.loads(Path('runs/reserve_feature_expansion_v1/physics_config.json').read_text()))
    details=[]
    for r in rows:
        orig=next(x for x in actual if all(x[k]==r[k] for k in ['side','state_id','force_N']))
        snap=Path(orig['snapshot']);receipts[str(snap)]=digest(snap)
        with snap.open('rb') as f:s=pickle.load(f)
        last=np.asarray(s.history.frames)[-1]
        details.append(dict(**r,**{k:float(last[spec.frame_names.index(k)]) for k in ['roll_rad','roll_rate_rad_s','steer_rad','eso_equilibrium_shift_rad']},fallback_time=orig['fallback_start_time_s'],event_time=orig['event_time_s']))
    pairs=[]
    for r0 in details:
        if r0['alpha']!=0:continue
        r1=next(x for x in details if x['alpha']==1 and all(x[k]==r0[k] for k in ['side','speed','fraction','force_N']))
        pairs.append(dict(context={k:r0[k] for k in ['side','speed','fraction','force_N']},alpha0=r0,alpha1=r1,
                          margin_increases=r1['formula_margin_N']>r0['formula_margin_N'],both_contact=not r0['safe'] and not r1['safe']))
    assert len(pairs)==16
    wrong=[]
    for yes in details:
        if not yes['safe']:continue
        for no in details:
            if not no['safe'] and yes['joint_score']<=no['joint_score']:
                wrong.append(dict(safe_case=yes,contact_case=no,formula_correct=yes['formula_margin_N']>no['formula_margin_N']))
    summary=dict(pairs=16,alpha1_gain=sum(not x['alpha0']['safe'] and x['alpha1']['safe'] for x in pairs),alpha1_loss=sum(x['alpha0']['safe'] and not x['alpha1']['safe'] for x in pairs),score_up_both_contact=sum(x['margin_increases'] and x['both_contact'] for x in pairs),all_safe_fallback=sum(x['fallback'] for x in details if x['safe']),all_safe=sum(x['safe'] for x in details),joint_wrong_success_contact_pairs=len(wrong),joint_wrong_cross_alpha=sum(x['safe_case']['alpha']!=x['contact_case']['alpha'] for x in wrong))
    atomic_json(out/'audit.json',dict(summary=summary,pairs=pairs,joint_wrong_pairs=wrong,input_sha256=receipts,new_physics_ticks=0,statistical_fits=0,scope='Own genuine prefixes, not same-state intervention; no future-informed deployable selection. Full nominal task costs on new geometry not collected.'))
    columns=['side','speed','fraction','force_N','alpha','body_speed','speed_proxy','roll_rad','roll_rate_rad_s','steer_rad','eso_equilibrium_shift_rad','formula_margin_N','joint_score','safe','fallback','fallback_time','event_time']
    with (out/'all32.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(details)
    lines=['# 偏好进入状态—方向性风险配对审计','',f'全部16配对、32受扰例，零新增物理/拟合。汇总：{summary}','',
           '|侧/请求速度/相位/力|α0实际速度/roll|α1实际速度/roll|F0→F1 N|α0→α1结局|','|---|---|---|---|---|']
    for pair in pairs:
        c=pair['context'];x=pair['alpha0'];y=pair['alpha1']
        label=lambda r:'无触地' if r['safe'] else '触地'
        lines.append(f"|{c['side']}/{c['speed']}/{c['fraction']}/{c['force_N']:+g}|{x['body_speed']:.3f}/{x['roll_rad']:.3f}|{y['body_speed']:.3f}/{y['roll_rad']:.3f}|{x['formula_margin_N']:.2f}→{y['formula_margin_N']:.2f}|{label(x)}→{label(y)}|")
    lines+=['','all32.csv保存全部速度、侧倾、侧倾角速度、转角、ESO、两评分、fallback与触地时刻。audit.json保存联合评分全部错误配对，含跨α标记，不只挑公式有利例。',
            '评分上升并不意味着跨过真实防摔边界。所有无触地例是否进入共同fallback单列；fallback不作为事前输入。新几何未采集完整名义任务/零中层对照，不能用旧几何代价冒充同条件成本。',
            '禁止以逐例结局选α作为可部署算法；两α来自各自真实前缀，不能据此解释同状态动作因果。特征训练无增量与同状态临时切换零救回的结论不变。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
