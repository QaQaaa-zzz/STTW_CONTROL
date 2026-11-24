"""Acknowledge ordered E1 candidates, recover exported work, and retain CEM inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cli'))
import pcrp_oracle_search as search


def test_resident_generation_protocol_and_restart(tmp_path,monkeypatch):
    root=tmp_path/'decision'
    stage=root/'e1/E1a'
    group=stage/'g01'
    (root/'step0').mkdir(parents=True)
    search.write(root/'step0/protocol.json',{'bundle':'original-frozen-bundle'})
    config=tmp_path/'h75_contact_oat/position/flat_original_v2.6_seed401.json'
    config.parent.mkdir(parents=True)
    search.write(config,{'seed':401})
    repo=tmp_path/'repo'
    queue=repo/'learning/scripts/gpu_queue.sh'
    queue.parent.mkdir(parents=True)
    queue.write_text('echo acquired >> "$TEST_LEASES"\nexec "$@"\n')
    cli=repo/'learning/cli/evaluate_isaac_gpu.py'
    cli.parent.mkdir(parents=True)
    cli.write_text("""import sys, json, os
from pathlib import Path
assert sys.argv[-2:]==['--resident-jobs','-']
for i,line in enumerate(sys.stdin):
    job=json.loads(line)
    assert set(job)=={'config','terrain','case_manifest','output'}
    path=Path(job['output'])
    path.mkdir()
    assert json.loads(Path(job['config']).read_text())['oracle_knots_sha256']
    assert json.loads(Path(job['case_manifest']).read_text())[0]['alphas']==[0.25]
    (path/'status.json').write_text(json.dumps({'stage':'complete','policy_steps':500}))
    with open(os.environ['TEST_CALLS'],'a') as f: print(path.parent.name,file=f)
    if os.environ.get('TEST_EXIT_AFTER_EXPORT')==path.parent.name: sys.exit(21)
    print(json.dumps({'resident_job_complete':str(path),'index':i}),flush=True)
""")
    monkeypatch.setattr(search,'REPO',repo)
    monkeypatch.setattr(search,'LAB',sys.executable)
    monkeypatch.setenv('TEST_LEASES',str(tmp_path/'leases'))
    monkeypatch.setenv('TEST_CALLS',str(tmp_path/'calls'))
    monkeypatch.setenv('TEST_EXIT_AFTER_EXPORT','c01')
    cases=[dict(case='stable',group=401,terrain='/original/terrain',randomization={})]
    protocol=dict(cases=cases,times_s=[0.,10.],population=3)
    specs=[(group/f'c{i:02}',f'CEM_01_{i:02}',[.25],np.full((1,2,2),i/100.)) for i in range(3)]
    status={'completed_new':6,'submitted_new':6,'generation':0,'stage':'running'}
    def score(folder,*args,**kwargs):
        assert search.read(folder/'batch/status.json')['policy_steps']==500
        return [int(folder.name[1:])]
    monkeypatch.setattr(search,'score_batch',score)
    args=(root,stage,protocol,specs,status,None)
    with pytest.raises(RuntimeError,match='exited 21'):
        search.evaluate_group(*args,resident=True,inspect=False)
    assert search.read(stage/'status.json')['completed_new']==7
    assert (tmp_path/'calls').read_text().splitlines()==['c00','c01']
    inputs=group/'c01/knots.npz'
    recorded=inputs.read_bytes()
    first=search.read(group/'resident_from_c00.json')
    assert len(first)==3
    monkeypatch.delenv('TEST_EXIT_AFTER_EXPORT')
    assert search.evaluate_group(*args,resident=True,inspect=False)==[[0],[1],[2]]
    assert (tmp_path/'calls').read_text().splitlines()==['c00','c01','c02']
    assert (tmp_path/'leases').read_text().splitlines()==['acquired','acquired']
    assert json.loads((group/'c02/simulation.log').read_text().strip())['resident_job_complete']==str(group/'c02/batch')
    assert inputs.read_bytes()==recorded
    assert search.read(group/'resident_from_c02.json')==first[-1:]
    assert search.read(stage/'status.json')['completed_new']==9
    assert search.read(stage/'status.json')['submitted_new']==9
    altered=[*specs[:1],(*specs[1][:3],specs[1][3]+.001),specs[2]]
    with pytest.raises(ValueError,match='Regenerated CEM knots differ'):
        search.evaluate_group(root,stage,protocol,altered,status,None,resident=True,inspect=True)
    (group/'c02/batch/status.json').unlink()
    with pytest.raises(RuntimeError,match='without a complete batch status'):
        search.evaluate_group(*args,resident=True,inspect=False)
    (group/'c02/batch').rmdir()
    with pytest.raises(ValueError,match='physical archive is missing'):
        search.evaluate_group(*args,resident=True,inspect=False,completed_floor=9)
