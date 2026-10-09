import numpy as np
from sttw_control.preference_command_reporting import _stage1_metrics, mode_costs

def trace(n=1000):
    t=np.arange(n)*.005;raw=np.column_stack((np.full(n,2.6),np.where(t>=1,.25,0)))
    return dict(time=t,limited_command=raw,governed=raw.copy(),actual_forward_speed=np.where(t>=2,2.35,2.6),actual_delta=raw[:,1].copy(),phi=np.full(n,.29),peak_roll=np.full(n,.29),e_psi_unwrapped=np.zeros(n),physical_failure=np.zeros(n,bool),chi=np.where(t>=1,1.,0),g=np.where(t<1,1.,0))

def test_drop_requires_contiguous_actual_speed_and_complete_episode():
    d=trace();m=_stage1_metrics(d)
    assert m['complete5s'] and m['drop_ge_0p2_longest_seconds']==2.5
    d['actual_forward_speed'][400:900:40]=2.6
    assert _stage1_metrics(d)['drop_ge_0p2_longest_seconds']<.3
    d=trace(60);d['physical_failure'][-1]=True
    m=_stage1_metrics(d)
    assert not m['complete5s'] and m['speed_rmse'] is None

def test_initial_straight_is_not_recovery_cost():
    d=trace();d['scored_cost_speed']=np.ones(1000)
    m=mode_costs(d)
    assert m['recovery_after_turn']['seconds']==0
    assert m['ordinary']['cost_integrals']['speed']==1
    assert m['conflict_nonrecovery']['cost_integrals']['speed']==4

def test_stage1_requires_both_turn_signs(tmp_path,monkeypatch):
    import sttw_control.preference_command_reporting as reporting
    monkeypatch.setattr(reporting,'_plots',lambda *args:None)
    monkeypatch.setattr(reporting,'reward_audit',lambda *args:{'passed':True})
    for sign,sgn in [('positive',1),('negative',-1)]:
        folder=tmp_path/'evaluation40'/sign;folder.mkdir(parents=True)
        for method in ['alpha0','alpha1','B0']:
            d=trace();d['limited_command'][:,1]*=sgn;d['governed']=d['limited_command'].copy();d['actual_delta']=d['limited_command'][:,1].copy()
            if method!='alpha0':
                d['actual_forward_speed'][:]=2.6
                d['actual_delta'][400:]-=.03*sgn
            np.savez(folder/(method+'.npz'),**d)
    assert reporting.report_stage1(tmp_path,40)['passed']
    path=tmp_path/'evaluation40/negative/alpha0.npz';d=reporting._load(path);d['peak_roll'][600]=.303;np.savez(path,**d)
    report=reporting.report_stage1(tmp_path,40)
    assert not report['passed'] and report['signs']['positive']['passed']
