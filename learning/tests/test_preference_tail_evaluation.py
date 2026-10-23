import copy,json
import pytest
from sttw_control.preference_best import digest
from sttw_control.preference_tail_evaluation import cached_policy_panel


def setup_panel(tmp_path):
    record={}
    for k in ('actor','checkpoint'):
        p=tmp_path/k;p.write_bytes(k.encode());record[k]=str(p);record[k+'_sha256']=digest(p)
    protocol=dict(config={'horizon':10},source={'lower':'frozen'},routes=[{'name':'left'}],
                  speeds=[2.35],alphas=[0],forces_N=[0,20],onsets_s=[2],duration_s=.1,
                  policies={'old':record})
    rows=[dict(name=f'case{i}',policy='old',route='left',speed=2.35,alpha=0,force_N=f,onset_s=2,task_recovered=True) for i,f in enumerate((0,20))]
    for r in rows:(tmp_path/(r['name']+'.npz')).write_bytes(b'fixture')
    (tmp_path/'manifest.json').write_text(json.dumps({'declaration':protocol}))
    (tmp_path/'results.json').write_text(json.dumps({'cases':rows}))
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='completed',completed_cases=2,declared_cases=2)))
    return protocol,record


def test_reuse_preserves_original_case_name_and_declares_alias(tmp_path):
    p,r=setup_panel(tmp_path);x=cached_policy_panel(tmp_path,p,r,'old','short4')
    assert [v['name'] for v in x['rows']]==['case0','case1']
    assert all(v['policy']=='short4' for v in x['rows'])
    assert x['receipt']['source_policy']=='old'


@pytest.mark.parametrize('change',['speed','model','missing','partial','intervention'])
def test_no_reuse_across_unmatched_or_incomplete_evidence(tmp_path,change):
    p,r=setup_panel(tmp_path);p=copy.deepcopy(p)
    if change=='speed':p['speeds']=[2.75]
    if change=='model':(tmp_path/'actor').write_bytes(b'changed')
    if change=='missing':(tmp_path/'case1.npz').unlink()
    if change=='partial':(tmp_path/'status.json').write_text(json.dumps(dict(phase='running',completed_cases=2,declared_cases=2)))
    if change=='intervention':p['recovery_intervention']={'start_s':2.1}
    with pytest.raises(ValueError):cached_policy_panel(tmp_path,p,r,'old','short4')
