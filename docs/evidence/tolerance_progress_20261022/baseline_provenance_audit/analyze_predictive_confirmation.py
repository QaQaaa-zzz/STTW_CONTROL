"""Audit saved-before-force forecasts against every prospective outcome."""
import argparse,json,pickle,subprocess,sys
from pathlib import Path
import numpy as np
from audit_operating_point_margin import auc
from analyze_matched_control import load_trace
from sttw_control.preference_best import atomic_json,digest


def summarize(rows):
    success=[r['safe'] for r in rows]
    fields=['formula_margin_N','speed_proxy','body_speed','lower_abs_roll','directional_lean_allowance','force_magnitude']
    return dict(n=len(rows),safe=sum(success),contact=sum(not x for x in success),
                AUC={key:auc(success,[r[key] for r in rows]) for key in fields},
                confusion=dict(predicted_pass_actual_pass=sum(r['safe'] and r['predicted_pass'] for r in rows),
                               predicted_pass_actual_contact=sum(not r['safe'] and r['predicted_pass'] for r in rows),
                               predicted_reject_actual_pass=sum(r['safe'] and not r['predicted_pass'] for r in rows),
                               predicted_reject_actual_contact=sum(not r['safe'] and not r['predicted_pass'] for r in rows)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.run.resolve();out=args.output.resolve();status=json.loads((root/'campaign_status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=48:raise ValueError('Wait for every prospective case')
    if status['active_lower_ticks']>54400:raise ValueError('Prospective budget exceeded')
    plan=json.loads((root/'plan.json').read_text())
    for path,sha in plan['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Frozen source changed '+path)
    from sttw_control.path_command_policy import actor_feature_spec
    from sttw_control.reserve_feature_inputs import features_jax
    import jax.numpy as j
    rows=[];all_cases=[];receipts={str(p.resolve()):digest(p) for p in [root/'plan.json',root/'campaign_status.json',Path(__file__),Path(__file__).with_name('audit_operating_point_margin.py')]}
    timeline=[];max_forecast_error=0.
    for side in ['left','right']:
        child=root/side;forecasts=json.loads((child/'prefix_forecasts.json').read_text());manifest=child/'manifest.json'
        if not forecasts['no_force_outcomes_existed'] or not forecasts['saved_unix']<manifest.stat().st_mtime:
            raise ValueError('Saved forecasts not attested before force manifest; preserve chronology issue')
        config=Path(forecasts['formula_config'])
        if digest(config)!=forecasts['formula_config_sha256']:raise ValueError('Formula configuration changed')
        cfg=json.loads(config.read_text());spec=actor_feature_spec(cfg)
        prefixes=json.loads((child/'prefix_results.json').read_text())['cases']
        predicted=forecasts['cases'];assert len(predicted)==len(prefixes)==8
        with np.load(child/'forecast_observations.npz',allow_pickle=False) as z:saved_obs=z['observation']
        for index,row in enumerate(predicted):
            snapshot=Path(row['snapshot'])
            if digest(snapshot)!=row['snapshot_sha256']:raise ValueError('Forecast snapshot changed')
            with snapshot.open('rb') as stream:s=pickle.load(stream)
            obs=np.concatenate([(np.asarray(s.history.frames)/np.asarray(spec.frame_scales)).reshape(-1),np.asarray(s.history.mask),np.zeros(15)]).astype('float32')
            obs[336]=float(s.alpha);obs=np.clip(obs,-5,5)
            np.testing.assert_array_equal(obs,saved_obs[index])
            f=np.asarray(features_jax(j.asarray(obs),spec,'physics'))
            max_forecast_error=max(max_forecast_error,abs(float(f[0]*spec.force_scale_N)-row['plus_N']),abs(float(f[1]*spec.force_scale_N)-row['minus_N']))
        for file in [child/'prefix_forecasts.json',child/'prefix_results.json',child/'forecast_observations.npz',child/'results.json',manifest]:receipts[str(file)]=digest(file)
        timeline.append(dict(side=side,forecast_saved_unix=forecasts['saved_unix'],force_manifest_mtime_unix=manifest.stat().st_mtime,
                             lead_seconds=manifest.stat().st_mtime-forecasts['saved_unix']))
        cases=json.loads((child/'results.json').read_text())['cases'];assert len(cases)==24
        for row in cases:
            trace=load_trace(child/(row['name']+'.npz'),row,receipts)
            prediction=next(x for x in predicted if x['state_id']==row['state_id'])
            r=dict(**row,side=side,pre_force_forecast=prediction)
            all_cases.append(r)
            if row['force_N']:
                force=row['force_N'];margin=prediction['plus_N'] if force>0 else prediction['minus_N']
                rows.append(dict(side=side,state_id=row['state_id'],route=row['route'],speed=row['requested_speed'],alpha=row['alpha'],fraction=row['fraction'],force_N=force,
                                 safe=row['balance_label']=='NO_FLOOR_CONTACT',label=row['balance_label'],
                                 formula_margin_N=margin,predicted_pass=margin>=abs(force),
                                 speed_proxy=prediction['speed_proxy'],body_speed=prediction['body_speed'],
                                 lower_abs_roll=-abs(prediction['roll']),directional_lean_allowance=-np.sign(force)*prediction['roll'],force_magnitude=-abs(force),
                                 fallback=row['fallback_triggered']))
        # Existing parameterized all-case renderer creates12 alpha-pair figures
        # per side. It never steps physics or modifies these outcomes.
        report=child/'report/analysis.json'
        if not report.exists():
            subprocess.run([sys.executable,'docs/evidence/tolerance_progress_20261022/alpha_physical_boundary/analyze.py','--run',str(child)],check=True)
        rendered=json.loads(report.read_text());assert len(rendered['cases'])==24
        for file,sha in rendered['raw_sha256'].items():
            if digest(file)!=sha:raise ValueError('Rendered raw trace changed')
    assert len(rows)==32 and len(all_cases)==48
    summary=summarize(rows)
    strata={}
    for field in ['side','speed','alpha','force_N']:
        strata[field]=[dict(value=value,**summarize([r for r in rows if r[field]==value])) for value in sorted({r[field] for r in rows})]
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'analysis.json',dict(summary=summary,strata=strata,cases=all_cases,scored_nonzero_cases=rows,
                forecast_chronology=timeline,max_recomputed_forecast_error_N=max_forecast_error,
                input_sha256=receipts,new_lower_ticks=status['active_lower_ticks'],optimizer_updates=0,parameter_fits=0,
                zeros=dict(n=16,safe=sum(r['balance_label']=='NO_FLOOR_CONTACT' for r in all_cases if not r['force_N'])),
                scope='Prospective DEV confirmation, one frozen H75;16 correlated captures from8 trajectory chains and4 route-speed blocks; no independent training repeats or safety certificate.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=list(summary['AUC']);values=[summary['AUC'][key] for key in fields]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    if all(v is not None for v in values):ax.barh(fields,values,color='#0072B2')
    else:ax.text(.5,.5,'Only one outcome class; ranking undefined',ha='center',transform=ax.transAxes)
    ax.axvline(.5,color='gray',ls=':');ax.set(xlim=(0,1),xlabel='prospective descriptive success-ranking AUC')
    fig.savefig(out/'ranking.png',dpi=140);plt.close(fig)
    lines=['# 固定公式：新几何/速度的前瞻确认','',
           '左右110°/R2.6、2.35/2.75m/s、α0/1、几何20%/70%，0/±80N、.1s、4s。16真实完整前缀、48新分支；每个方向先落盘全部预测才开始施扰。',
           f"实际新增{status['active_lower_ticks']}lower ticks，0优化/参数拟合。32受扰结局：无触地{summary['safe']}、触地{summary['contact']}。",'',
           '|固定评分|受扰无触地排序AUC|','|---|---:|']
    for name,value in summary['AUC'].items():lines.append(f'|{name}|'+('未定义（单一结局）' if value is None else f'{value:.5f}')+'|')
    lines += ['',f"原F≥80判据的全部混淆计数：{summary['confusion']}。",f'原快照重建观测与保存输入逐值一致；CPU/GPU公式重算最大差{max_forecast_error:g}N，误差记录不改预测。',
              '预测时间早于首个force manifest文件创建/写入，执行源也固定forecast-before-scan顺序；时间证据、输入SHA、每个原预测和真实结局全部保留。',
              '这是一个冻结策略在4个新路线×速度块上的前瞻DEV确认，16捕获来自8条轨迹链；不当作独立训练重复、最终TEST或普遍安全证明。分方向/速度/alpha/施力符号的计数与undefined排序保留在analysis.json，不删除失败或饱和分组。',
              '', '![固定评分比较](ranking.png)','', '## 全部48例、24组XY及真实终点', '',
              f'- [左弯全部12组]({root / "left/report/REPORT.md"})',f'- [右弯全部12组]({root / "right/report/REPORT.md"})']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,new_lower_ticks=status['active_lower_ticks'],timeline=timeline,max_recomputed_forecast_error_N=max_forecast_error),ensure_ascii=False))


if __name__=='__main__':main()
