"""Frozen formula versus simple state scores on existing physical outcomes; no fitting."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from sttw_control.preference_best import atomic_json,digest


def auc(success, score):
    yes=np.asarray(score)[np.asarray(success,bool)]
    no=np.asarray(score)[~np.asarray(success,bool)]
    if not len(yes) or not len(no):return None
    return float(np.mean((yes[:,None]>no[None,:])+.5*(yes[:,None]==no[None,:])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path.cwd();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=root/'runs/zero_middle_matched_analysis_v1/analysis.json'
    config=root/'runs/reserve_feature_expansion_v1/physics_config.json'
    data=json.loads(source.read_text());cfg=json.loads(config.read_text())
    from sttw_control.path_command_policy import actor_feature_spec
    from sttw_control.reserve_feature_inputs import features_jax
    import jax.numpy as j
    spec=actor_feature_spec(cfg)
    receipts={str(source):digest(source),str(config):digest(config),str(Path(__file__).resolve()):digest(__file__)}
    for name in ['reserve_feature_inputs','path_command_policy']:
        file=root/'learning/src/sttw_control'/(name+'.py');receipts[str(file)]=digest(file)
    rows=[];snapshots={}
    for row in data['cases']:
        if not row['force_N']:continue
        file=Path(row['snapshot'])
        if file not in snapshots:
            prefix_file=file.parent/'prefix_results.json'
            metadata=json.loads(prefix_file.read_text())['cases']
            original=next(x for x in metadata if x['snapshot']==str(file))
            if digest(file)!=original['snapshot_sha256']:raise ValueError('Recorded prefix snapshot changed')
            receipts[str(prefix_file)]=digest(prefix_file)
            receipts[str(file)]=digest(file)
            with file.open('rb') as stream:state=pickle.load(stream)
            frames=np.asarray(state.history.frames)
            masks=np.asarray(state.history.mask)
            assert frames.shape==(16,20) and masks.all()
            obs=np.concatenate([(frames/np.asarray(spec.frame_scales)).reshape(-1),masks,np.zeros(15)]).astype('float32')
            obs[336]=float(state.alpha);obs=np.clip(obs,-5,5)
            f=np.asarray(features_jax(j.asarray(obs),spec,'physics'))
            if f[-1]!=1.:raise ValueError('Invalid last-two-frame physical feature')
            last=frames[-1];names=spec.frame_names
            snapshots[file]=dict(plus_N=float(f[0]*spec.force_scale_N),minus_N=float(f[1]*spec.force_scale_N),
                                 force_output_clipped=bool(np.any(np.abs(f[:2])>=spec.clip_abs)),
                                 speed_proxy=float(last[names.index('rear_speed_proxy_m_s')]),
                                 body_speed=float(last[names.index('forward_speed_est_m_s')]),
                                 roll=float(last[names.index('roll_rad')]),frame_time_s=float(state.tick)*.005-.005)
        s=snapshots[file];force=row['force_N'];margin=s['plus_N'] if force>0 else s['minus_N']
        rows.append(dict(route=row['route'],speed=row['requested_speed'],fraction=row['fraction'],model=row['model'],
                         force_N=force,label=row['balance_label'],safe=row['balance_label']=='NO_FLOOR_CONTACT',
                         formula_margin_N=margin,formula_predicts_pass=margin>=abs(force),
                         lower_abs_roll=-abs(s['roll']),
                         directional_lean_allowance=-np.sign(force)*s['roll'],**s))
    assert len(rows)==48 and len(snapshots)==24
    scores=['formula_margin_N','speed_proxy','body_speed','lower_abs_roll','directional_lean_allowance']
    labels=[r['safe'] for r in rows]
    summary={name:dict(success_ranking_AUC=auc(labels,[r[name] for r in rows])) for name in scores}
    summary['force_magnitude']=dict(success_ranking_AUC=auc(labels,[-abs(r['force_N']) for r in rows]))
    confusion=dict(predicted_pass_actual_pass=sum(r['safe'] and r['formula_predicts_pass'] for r in rows),
                   predicted_pass_actual_contact=sum(not r['safe'] and r['formula_predicts_pass'] for r in rows),
                   predicted_reject_actual_pass=sum(r['safe'] and not r['formula_predicts_pass'] for r in rows),
                   predicted_reject_actual_contact=sum(not r['safe'] and not r['formula_predicts_pass'] for r in rows))
    atomic_json(out/'analysis.json',dict(cases=rows,ranking=summary,formula_confusion=confusion,
                frozen_formula_spec=cfg['network']['reserve_feature_spec'],input_sha256=receipts,
                cases_count=48,prefixes_count=24,new_physics_ticks=0,optimizer_updates=0,parameter_fits=0,
                interpretation='Exploratory ranking on previously examined DEV outcomes. Future frozen controller/fallback differs by policy; ideal virtual-input prior is not a calibrated closed-loop probability or safety certificate.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    names=list(summary);ax.barh(names,[summary[n]['success_ranking_AUC'] for n in names],color='#0072B2')
    ax.axvline(.5,color='gray',ls=':');ax.set(xlim=(0,1),xlabel='descriptive success-ranking AUC; correlated DEV cases')
    fig.savefig(out/'ranking.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained',sharey=True)
    for ax,model in zip(axes,['H75_a0','H75_a1','zero_middle']):
        selected=[r for r in rows if r['model']==model]
        for r in selected:
            ax.scatter(r['formula_margin_N'],r['speed_proxy'],color='#009E73' if r['safe'] else '#D55E00',
                       marker='o' if r['safe'] else 'x',s=40)
        ax.axvline(80,color='gray',ls=':');ax.set(title=model,xlabel='frozen directional formula [N]');ax.grid(alpha=.2)
    axes[0].set_ylabel('observed rear speed proxy [m/s]')
    fig.suptitle('Actual +/-80N outcomes: green=o no contact; orange=x contact; not fitted')
    fig.savefig(out/'formula_vs_actual.png',dpi=140);plt.close(fig)
    lines=['# 既有物理工况：固定公式与简单状态指标','',
           '复用24个真实完整前缀、48个±80N物理结局。参数、时长与六特征实现均冻结，不拟合T/γ或标签；0新增物理/优化。',
           '输入来自真实历史最后两帧（5ms更新），保留Actor实际归一化/截断；输出乘50恢复为牛顿。body_speed仅是同一已保存观测中的诊断基线，部署仍需真实速度估计。','',
           '|指标|无触地排序AUC|','|---|---:|']
    for n,x in summary.items():lines.append(f"|{n}|{x['success_ranking_AUC']:.5f}|")
    lines+=['',f'固定F≥80判据的原始混淆计数：{confusion}。',
            '排序较好不意味着力边界准确；预测通过而触地的例子全部保留，不能称安全认证。48例共享24前缀且已有结果被看过，AUC是描述性数值，不是独立测试或一般化概率。',
            '这评价的是解析量的事后解释/排序，不推翻physics特征训练无额外收益，也不把同状态无救回变成控制成功。',
            '', '![固定评分比较](ranking.png)','', '![固定力下预测与真实结果](formula_vs_actual.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(ranking=summary,formula_confusion=confusion,cases=48,prefixes=24),ensure_ascii=False))


if __name__=='__main__':main()
