"""Reconstruct published PP commands from saved post-tick geometry, no physics."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    from sttw_control.preference_best import atomic_json
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_reference import pursuit
    from sttw_control.geometric_path import build_path
    import jax
    import jax.numpy as j
    plan=json.loads((root/'plan.json').read_text());out=root/'follower_audit'
    out.mkdir(exist_ok=False);results=[];captures=[]
    for panel in plan['panels']:
        d=json.loads(Path(panel['declaration']).read_text());cfg=d['config'];ev=Path(panel['output'])
        env,initial=load_environment(cfg,d['source'],'terminal_contract_v2')
        _,initial_pose,initial_v=env.physics.observe(initial.data)
        cases=json.loads((ev/'results.json').read_text())['cases']
        paths={r['name']:build_path(r['builder'],initial_pose,cfg) for r in d['routes']}
        for route,path in paths.items():
            with np.load(ev/(route+'_path.npz')) as f:
                np.testing.assert_allclose(np.asarray(path.xy),f['xy'],atol=1e-6,rtol=0)
        for row in cases:
            with np.load(ev/(row['name']+'.npz')) as f:t={k:f[k] for k in f.files}
            path=paths[row['route']];n=len(t['time']);fc=cfg['follower']
            path_yaw=np.interp(t['path_progress'],np.asarray(path.s),np.asarray(path.heading))
            # Heading error is post-tick path tangent minus physical yaw, wrapped.
            # Trigonometric reconstruction is equivalent modulo 2pi.
            poses=np.c_[t['actual_xy'],path_yaw-t['path_heading_error']]
            pp=jax.vmap(lambda progress,pose,speed:pursuit(path,progress,pose,speed,row['speed'],cfg))(
                j.asarray(t['path_progress']),j.asarray(poses,dtype=j.float32),j.asarray(t['actual_forward_speed']))
            pp=jax.device_get(pp)
            target=np.asarray(pursuit(path,j.array(0.),initial_pose,initial_v,row['speed'],cfg).command)
            calculated=[];slew=[];targets=[]
            limits=np.array([fc['nominal_speed_slew_m_s2'],fc['nominal_steer_slew_rad_s']])*.005
            for k in range(n-1):
                if (k+1)%4==0:target=np.asarray(pp.command[k])
                current=t['nominal'][k];change=np.clip(target-current,-limits,limits)
                calculated.append(current+change);targets.append(target.copy());slew.append(np.abs(target-current)>limits+1e-7)
            calculated=np.asarray(calculated);error=calculated-t['nominal'][1:]
            end=float(path.turn_end);start=cfg['path']['straight_prefix_m'];target_progress=start+.5*(end-start)
            hit=np.flatnonzero(t['path_progress']>=target_progress)
            if len(hit):
                k=int(hit[0]);captures.append(dict(panel=panel['name'],policy=row['policy'],route=row['route'],speed=row['speed'],alpha=row['alpha'],
                    first_half_turn_time_s=float(t['time'][k]+.005),actual_progress_m=float(t['path_progress'][k]),target_progress_m=target_progress,
                    actual_speed_m_s=float(t['actual_forward_speed'][k]),roll_rad=float(t['phi'][k]),roll_rate_rad_s=float(t['phi_dot'][k]),
                    governed=t['governed'][k].tolist(),
                    complete_snapshot_available=False,source_note='Saved measurement trace only; cannot reconstruct hidden state/ESO/history for branching'))
            turn=(t['path_progress']>=start)&(t['path_progress']<=end)
            steer_error=t['actual_delta']-t['governed'][:,1]
            results.append(dict(panel=panel['name'],name=row['name'],policy=row['policy'],alpha=row['alpha'],speed=row['speed'],
                domain_exit=row['domain_failure'],task=row['task_recovered'],
                nominal_reconstruction_max_error_m_s_rad=np.max(np.abs(error),axis=0).tolist(),
                nominal_slew_fraction=np.mean(slew,axis=0).tolist(),
                turn_preview_beyond_turn_end_fraction=float(np.mean((t['path_progress'][turn]+pp.lookahead[turn])>end)) if turn.any() else None,
                turn_steer_tracking_rmse_rad=float(np.sqrt(np.mean(steer_error[turn]**2))) if turn.any() else None,
                turn_max_governed_steer_abs_rad=float(np.abs(t['governed'][turn,1]).max()) if turn.any() else None,
                terminal_preview_forward_m=float(pp.preview_body[-1,0]),
                terminal_heading_error_rad=float(t['path_heading_error'][-1]),
                terminal_preview_behind=bool(pp.preview_body[-1,0]<=fc['preview_body_x_min_m']),
                terminal_heading_out_of_domain=bool(abs(t['path_heading_error'][-1])>=np.pi/2)))
            np.savez_compressed(out/(panel['name']+'_'+row['name']+'.npz'),time=t['time'],
                actual_xy=t['actual_xy'],published_nominal=t['nominal'],reconstructed_next_nominal=calculated,
                pure_pursuit_target=pp.command,preview_body=pp.preview_body,lookahead=pp.lookahead,
                path_progress=t['path_progress'],actual_delta=t['actual_delta'],governed=t['governed'],
                steer_tracking_error=steer_error)
    maxerr=np.max([r['nominal_reconstruction_max_error_m_s_rad'] for r in results],axis=0)
    result=dict(cases=results,half_turn_measurements=captures,max_nominal_reconstruction_error=maxerr.tolist(),
        new_physics_steps=0,policy_updates=0,scope='Observed trajectory command reconstruction only; no prediction of a changed follower closed loop')
    atomic_json(out/'audit.json',result)
    lines=['# 紧弯跟随器离线核对','','目的：判断新场景的名义失败是否足以支持继续测容扰度，避免把任务跟随问题误当防摔问题。',
        '方法：原始完整bank和固定路径只加载、0积分；用记录的post-tick XY/路径进度/heading error/速度重建PP，再按原50Hz目标更新、200Hz slew发布，与下一tick记录的nominal比较。没有生成新闭环控制结果。',
        '',f'全32例最大速度/转角发布重建误差：{maxerr.tolist()}。浮点与航向绕回误差保留，不要求位级一致。',
        '', 'half_turn_measurements只有实测日志，不含可分支完整状态，不能伪装成mid-turn snapshot。任一lookahead候选的闭环效果仍需真实仿真，不能从离线重算直接推断。',
        '', '[本组全部实际/原任务XY](../analysis/all_paths.png)。逐例目标/预瞄/发布/转角误差在同目录NPZ，详细数字见audit.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cases=len(results),max_reconstruction_error=maxerr.tolist(),domain_preview_behind=sum(r['domain_exit'] and r['terminal_preview_behind'] for r in results))))


if __name__=='__main__':main()
