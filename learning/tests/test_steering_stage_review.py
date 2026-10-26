from pathlib import Path
import runpy
import numpy as np


def test_prefix_diagnostic_excludes_postforce_data_and_keeps_last_preforce_sample(tmp_path):
    cli=Path(__file__).resolve().parents[1]/'cli/review_steering_stage.py'
    audit=runpy.run_path(str(cli))['audit_prefixes']
    (tmp_path/'analysis').mkdir()
    roles=['goal_zero_feedback','neural_before_only','neural_after_only','goal_steer_feedback']
    cell={}
    for i,role in enumerate(roles):
        scalar=np.arange(8,dtype=float)
        # The two late-stage requests change their future only; that future
        # must not leak into either the prefix difference or onset sample.
        scalar[4:]+=i*10000
        data=dict(active_tick=np.ones(8,bool),physical_tick=np.arange(8),
                  actual_xy=np.column_stack([scalar,scalar]),actual_forward_speed=scalar,
                  phi=scalar,phi_dot=scalar,actual_delta=scalar,
                  applied_force_impulse=np.zeros((8,3)),
                  nominal_reference=np.column_stack([scalar,scalar]))
        file=tmp_path/f'{role}.npz';np.savez_compressed(file,**data)
        cell[role]=dict(raw_file=str(file),pulse_onset_task_tick=4,case_id=role)
    result=audit(tmp_path,{('ctx',.5,'event'):cell})
    assert len(result['comparisons'])==2
    assert all(r['common_planned_preswitch_ticks']==4 for r in result['comparisons'])
    assert all(all(x==0 for x in r['max_abs_recorded_differences'].values()) for r in result['comparisons'])
    assert all(s['sample_task_tick']==3 and s['values']['phi']==3. for s in result['precut_samples'])
