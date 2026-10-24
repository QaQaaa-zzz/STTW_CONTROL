"""Supplemental fixed-score transfer across reference policies; no fitting/physics."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from audit_operating_point_margin import auc
from sttw_control.preference_best import atomic_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=a.run.resolve();out=root/'formula_transfer_supplement'
    if json.loads((root/'analysis_status.json').read_text())['phase']!='completed':raise ValueError('Wait for all72 raw-audited outcomes')
    matched=root/'analysis/analysis.json';data=json.loads(matched.read_text())
    frozen=Path('runs/formula_prospective_confirmation_v1/joint_baseline_supplement/frozen_predictions.json').resolve()
    fit=json.loads(frozen.read_text());mean=np.asarray(fit['mean']);scale=np.asarray(fit['scale']);w=np.asarray(fit['weights'])
    config=Path('runs/reserve_feature_expansion_v1/physics_config.json').resolve()
    from sttw_control.path_command_policy import actor_feature_spec
    from sttw_control.reserve_feature_inputs import features_jax
    import jax.numpy as jp
    spec=actor_feature_spec(json.loads(config.read_text()));rows=[];cache={}
    receipts={str(x):digest(x) for x in [matched,frozen,config,Path(__file__).resolve()]}
    for r in data['cases']:
        if not r['force_N']:continue
        file=Path(r['snapshot']);receipts[str(file)]=digest(file)
        if str(file) not in cache:
            with file.open('rb') as f:s=pickle.load(f)
            last=np.asarray(s.history.frames)[-1]
            obs=np.concatenate([(np.asarray(s.history.frames)/np.asarray(spec.frame_scales)).reshape(-1),np.asarray(s.history.mask),np.zeros(15)]).astype('float32')
            obs[336]=float(s.alpha);obs=np.clip(obs,-5,5)
            physics=np.asarray(features_jax(jp.asarray(obs),spec,'physics'))
            if physics[-1]!=1 or np.any(np.abs(physics[:2])>=spec.clip_abs):raise ValueError('Invalid/clipped formula input')
            cache[str(file)]=(last,physics)
        last,physics=cache[str(file)];force=r['force_N'];v=float(last[spec.frame_names.index('rear_speed_proxy_m_s')])
        x=np.array([v,v*v]+[np.sign(force)*last[spec.frame_names.index(k)] for k in ['roll_rad','roll_rate_rad_s','steer_rad','eso_equilibrium_shift_rad']])
        rows.append(dict(model=r['model'],side=r['side'],speed=r['requested_speed'],fraction=r['fraction'],force_N=force,
                         safe=r['balance_label']=='NO_FLOOR_CONTACT',formula=float(physics[0 if force>0 else 1]*spec.force_scale_N),
                         joint=float(np.r_[1.,(x-mean)/scale]@w),speed_proxy=v,lean=float(-np.sign(force)*last[spec.frame_names.index('roll_rad')]),fallback=r['fallback_triggered']))
    assert len(rows)==48
    def summary(rs):return dict(n=len(rs),safe=sum(x['safe'] for x in rs),AUC={k:auc([x['safe'] for x in rs],[x[k] for x in rs]) for k in ['formula','joint','speed_proxy','lean']})
    per={m:summary([r for r in rows if r['model']==m]) for m in sorted({r['model'] for r in rows})}
    out.mkdir(exist_ok=False);atomic_json(out/'analysis.json',dict(per_model=per,pooled_descriptive=summary(rows),cases=rows,input_sha256=receipts,new_physics_ticks=0,statistical_fits=0,scope='Post-start supplemental exploratory transfer; original primary unchanged; joint coefficients frozen on oldDEV. Shared prefixes and fallback; no independent replication or control-benefit claim.'))
    lines=['# 公式跨参考生成策略的追加迁移检查','','先完成全部72原始结局检查，再一次性评价；不改公式/联合基线系数/阈值、0新增物理或拟合。此项是启动后追加探索分析，非原前瞻primary。','',
           '|模型|非零力数/无触地|公式AUC|旧DEV联合AUC|速度AUC|方向侧倾AUC|','|---|---:|---:|---:|---:|---:|']
    for m,s in per.items():lines.append(f"|{m}|{s['n']}/{s['safe']}|{s['AUC']['formula']}|{s['AUC']['joint']}|{s['AUC']['speed_proxy']}|{s['AUC']['lean']}|")
    lines+=['','先看各模型内部，再看混合描述；单类组记None，不追加力档。公式有效排序仍不是牛顿阈值校准、普遍物理极限或主动控制收益。联合基线接近时不主张物理信息必要。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(per))


if __name__=='__main__':main()
