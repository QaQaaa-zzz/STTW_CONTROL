"""Save frozen, history-only scores before the prospective physical force scan."""
import argparse,json,os,pickle,time
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    root=a.run.resolve();plan=json.loads((root/'prepare_plan.json').read_text())
    status=dict(phase='running',pid=os.getpid(),stage='frozen_history_predictions',training_steps=0,active_lower_ticks=0)
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(root/'forecast_status.json',status)
    update()
    try:
        if (root/'results.json').exists():raise ValueError('Force outcomes already exist; not prospective')
        for path,sha in plan['source_sha256'].items():
            if digest(path)!=sha:raise ValueError('Declared forecast source changed '+path)
        from sttw_control.path_command_policy import actor_feature_spec
        from sttw_control.reserve_feature_inputs import features_jax
        import jax.numpy as j
        config=Path(plan['source_formula_config']);cfg=json.loads(config.read_text());spec=actor_feature_spec(cfg)
        prefixes=json.loads((root/'prefix_results.json').read_text())['cases'];rows=[];observations=[]
        for prefix in prefixes:
            file=Path(prefix['snapshot'])
            if digest(file)!=prefix['snapshot_sha256']:raise ValueError('Snapshot changed')
            with file.open('rb') as stream:s=pickle.load(stream)
            frames=np.asarray(s.history.frames);mask=np.asarray(s.history.mask)
            if not mask.all():raise ValueError('Incomplete real history')
            obs=np.concatenate([(frames/np.asarray(spec.frame_scales)).reshape(-1),mask,np.zeros(15)]).astype('float32')
            obs[336]=float(s.alpha);obs=np.clip(obs,-5,5);observations.append(obs)
            f=np.asarray(features_jax(j.asarray(obs),spec,'physics'));last=frames[-1];names=spec.frame_names
            if f[-1]!=1.:raise ValueError('Physical feature invalid')
            plus,minus=float(f[0]*spec.force_scale_N),float(f[1]*spec.force_scale_N)
            rows.append(dict(state_id=prefix['state_id'],snapshot=str(file),snapshot_sha256=digest(file),
                             plus_N=plus,minus_N=minus,predicted_pass_plus80=plus>=80,predicted_pass_minus80=minus>=80,
                             speed_proxy=float(last[names.index('rear_speed_proxy_m_s')]),
                             body_speed=float(last[names.index('forward_speed_est_m_s')]),
                             roll=float(last[names.index('roll_rad')]),force_output_clipped=bool(np.any(np.abs(f[:2])>=spec.clip_abs))))
        np.savez_compressed(root/'forecast_observations.npz',observation=np.asarray(observations))
        atomic_json(root/'prefix_forecasts.json',dict(cases=rows,saved_unix=time.time(),formula_config=str(config),
                    formula_config_sha256=digest(config),formula_spec=cfg['network']['reserve_feature_spec'],
                    no_force_outcomes_existed=True,source_sha256=plan['source_sha256'],training_steps=0))
        update(phase='completed',stage='all_history_scores_saved_before_force_scan',completed_cases=len(rows))
    except BaseException as exc:update(phase='error',error=repr(exc));raise


if __name__=='__main__':main()
