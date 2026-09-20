"""Second validation: complete user-supplied numeric evidence, no invented samples."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest

ROOT=Path(__file__).parents[2]
EVIDENCE=os.environ.get('STTW_MODE_EVIDENCE')
pytestmark=pytest.mark.skipif(not EVIDENCE,reason='requires the supplied detailed diagnosis directory; not a physics test')


def auditor():
    p=ROOT/'learning/src/sttw_control/tracking_diagnostics.py';tree=ast.parse(p.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inspect_alpha_diagnosis']
    ns={'Path':Path,'np':np,'json':json,'hashlib':hashlib}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns['inspect_alpha_diagnosis']


def test_all_uploaded_steps_same_state_arrays_and_reward_reconstruction():
    r=auditor()(EVIDENCE)
    assert len(r['alphas'])==3 and r['total_recorded_transitions']==3600 and r['same_state_mode_queries']==10800
    for a in r['alphas']:
        assert a['postcommand_steps']==1200 and a['reference_turn_steps']==78
        assert .472<a['roll_peak']<.476
        assert a['roll_above_03_seconds']==.325
        assert .202<a['overspeed_peak']<.216
        assert .009<a['same_state_action_delta_rms'][0]<.011
        assert .006<a['same_state_action_delta_rms'][1]<.0065
        assert .995<a['rms_command_difference_retention'][0]<1.001
        assert .999<a['rms_command_difference_retention'][1]<1.001
        assert a['command_clipped_fraction']<.01
        assert a['reward_reconstruction_max_abs']<1e-7
    source=r['alphas'];assert source[2]['speed_rmse']>source[0]['speed_rmse']
    assert source[2]['turn_underspeed_integral']<source[0]['turn_underspeed_integral']
    assert source[2]['turn_path_integral']>source[0]['turn_path_integral']


@pytest.mark.parametrize('field',['raw_total_cost','reward','mapping_slope','prelimit_command_rear','alpha','time_post'])
def test_numeric_tampering_is_rejected(tmp_path,field):
    import csv
    target=tmp_path/'diagnosis';shutil.copytree(EVIDENCE,target)
    p=target/'steps_alpha_0.csv'
    with p.open() as f:rows=list(csv.DictReader(f))
    rows[700][field]=str(float(rows[700][field])+.5)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError):auditor()(target)


@pytest.mark.parametrize('field',['action','mu','alpha','time'])
def test_same_state_tampering_is_rejected(tmp_path,field):
    target=tmp_path/'diagnosis';shutil.copytree(EVIDENCE,target)
    p=target/'same_state_alpha_0.npz'
    with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    data[field].flat[0]+=.5
    np.savez_compressed(p,**data)
    with pytest.raises(ValueError):auditor()(target)
