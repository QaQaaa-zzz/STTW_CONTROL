"""Changing an issued reference must not relax the frozen task-road contract."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def test_original_road_rejects_goal_reached_on_wrong_issued_reference(tmp_path):
    run=tmp_path/'run';case=run/'case';lane=case/'policy';lane.mkdir(parents=True)
    issued=tmp_path/'issued.npz';road=tmp_path/'road.npz'
    np.savez(issued,xy=np.array([[0.,0.],[20.,0.]]))
    np.savez(road,xy=np.array([[0.,2.],[20.,2.]]))
    (run/'protocol.json').write_text(json.dumps(dict(cases=['case'],goal=dict(x_min_m=16.5,x_max_m=18.5,abs_y_max_m=.5,min_vx_m_s=.5),road_half_width_m=.6)))
    cfg=dict(task_reference_path=str(road),ground_handoff=dict(reference_path=str(issued),approach=dict(speed_m_s=2.,alpha=.5,jump_start_x_m=9.)))
    (case/'config.json').write_text(json.dumps(cfg))
    (case/'summary.json').write_text(json.dumps(dict(cases=[dict(candidate=dict(name='policy'),terminal_code=9,duration_s=.2,final_xyz=[17.,0.,.12])])))
    (case/'handoff.json').write_text(json.dumps(dict(time_s=.05)))
    time=np.array([.1,.2]);root=np.zeros((2,1,13));root[:,:,0]=17;root[:,:,2]=.12;root[:,:,3]=1;root[:,:,7]=1
    np.savez(lane/'physics_contact.npz',time=time,root_state_wxyz=root)
    np.savez(lane/'trajectory.npz',time=time,reward=np.zeros((2,1)))
    np.savez(lane/'verified_diagnostics.npz',time=time,failed=np.zeros((2,1),bool))
    cli=Path(__file__).resolve().parents[1]/'cli/analyze_ditch_goal.py'
    out=tmp_path/'report'
    subprocess.run([sys.executable,str(cli),'--runs',str(run),'--output',str(out)],check=True,capture_output=True)
    result=json.loads((out/'summary.json').read_text())[0]
    assert result['arrival_s']==.1
    assert result['max_centerline_distance_m']==2.
    assert not result['goal_completed']
    # Without a separate task road, the legacy contract is the issued path.
    del cfg['task_reference_path'];(case/'config.json').write_text(json.dumps(cfg))
    subprocess.run([sys.executable,str(cli),'--runs',str(run),'--output',str(out)],check=True,capture_output=True)
    assert json.loads((out/'summary.json').read_text())[0]['goal_completed']
