"""Counterfactual TARGET replay only, not a new closed-loop trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def target(latent, interface):
    a=np.tanh(latent)
    return np.stack([a[...,0]*np.where(a[...,0]>=0,interface['speed_positive_scale_m_s'],interface['speed_negative_scale_m_s']),
                     a[...,1]*interface['steer_scale_rad']],axis=-1)


def audit(runs, output):
    records,hashes=[],{}
    metadata={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for root in runs:
        analysis=json.loads((root/'analysis/analysis.json').read_text())
        design=json.loads((root/'design.json').read_text())
        for plan_path in design['plans']:
            directory=Path(plan_path).parent
            manifest=json.loads((directory/'manifest.json').read_text())
            for p in (directory/'manifest.json',directory/'results.json',root/'analysis/analysis.json'):
                metadata[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            c=manifest['config']['residual_interface']
            lo=np.array([-c['speed_negative_scale_m_s'],-c['steer_scale_rad']])
            hi=np.array([c['speed_positive_scale_m_s'],c['steer_scale_rad']])
            for row in json.loads((directory/'results.json').read_text())['cases']:
                path=directory/(row['name']+'.npz')
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest==analysis['raw_sha256'][str(path)]
                hashes[str(path)]=digest
                if not row['override_lower_ticks']:
                    continue
                with np.load(path) as z:data={k:z[k] for k in z.files}
                indices=np.flatnonzero(data['override_active'])
                assert len(indices)==row['override_lower_ticks']
                first=int(indices[0]);mask=data['gate_mask'][first].astype(bool)
                assert mask.sum()==1
                parent=target(data['original_policy_latent'][indices],c)
                locked=target(data['executed_policy_latent'][indices],c)
                np.testing.assert_array_equal(locked[:,mask],np.broadcast_to(locked[0,mask],locked[:,mask].shape))
                # Preserve the SAME realizable detection-time target, adding
                # only subsequent parent changes; unsampled future is unused.
                before=locked[0]+(parent-parent[0])*mask
                follow=np.clip(before,lo,hi)
                np.testing.assert_array_equal(follow[0,mask],locked[0,mask])
                assert np.all(follow>=lo) and np.all(follow<=hi)
                channel=int(np.flatnonzero(mask)[0])
                records.append(dict(run=root.name,bank=directory.name,route=row['route'],speed=row['speed'],force_N=row['force_N'],
                                    physical_label=row['balance_label'],channel='speed' if channel==0 else 'steer',
                                    duration_s=len(indices)*.005,parent_anchor=float(parent[0,channel]),
                                    locked_target=float(locked[0,channel]),
                                    max_parent_change=float(np.max(np.abs(parent[:,channel]-parent[0,channel]))),
                                    max_candidate_vs_locked=float(np.max(np.abs(follow[:,channel]-locked[:,channel]))),
                                    candidate_vs_locked_absolute_integral=float(np.sum(np.abs(follow[:,channel]-locked[:,channel]))*.005),
                                    added_target_clipped_ticks=int(np.any(before!=follow,axis=1).sum())))
    output.mkdir(exist_ok=False)
    result=dict(records=records,raw_sha256=hashes,metadata_sha256=metadata,new_physical_ticks=0,fits=0,
                claim='Same existing observations, counterfactual target only. Later states, triggers, rewards and contact outcomes are NOT predictions for the candidate.')
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 父目标变化与实时跟随候选的零物理机会核查', '',
           *[f'[{r.name}全部实际XY]({r}/analysis/REPORT.md)' for r in runs], '',
           f'目的：判断锁定的.4s内父目标是否确有变化，避免为微小数值差异添加模块。认证全部{len(hashes)}条旧raw，不重跑物理或Actor；只在{len(records)}个真实干预窗口内回读目标。候选在检测时与旧可执行目标相同，之后加入父目标相对检测时的变化，仍夹紧到原权限。', '',
           '这是已有轨迹上的反事实目标检查；候选真正执行后观测会变化，因此不继承任何旧触地/任务标签，不称新闭环收益或偏好保持。speed单位m/s，steer单位rad；积分还乘秒。', '',
           '|run/bank/路线|力N|通道|窗s|父目标最大变化|候选相对锁定最大变化|目标差绝对积分|新增夹紧ticks|', '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['run']}/{r['bank']}/{r['route']}|{r['force_N']:g}|{r['channel']}|{r['duration_s']:.3f}|{r['max_parent_change']:.6f}|{r['max_candidate_vs_locked']:.6f}|{r['candidate_vs_locked_absolute_integral']:.6f}|{r['added_target_clipped_ticks']}|")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(output/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit([r.resolve() for r in args.run],args.output.resolve())
