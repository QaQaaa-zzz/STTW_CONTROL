import numpy as np
import pytest

def rows():
    return dict(steer_rmse=np.array([.01,.1,.2]),speed_rmse=np.array([.3,.02,.4]),
      smoothness=np.zeros(3),heading_cost=np.ones(3),feasible=np.ones(3,bool),
      finite=np.ones(3,bool),physical_failure=np.zeros(3,bool),violation=np.zeros(3))

def test_lexicographic_priorities_and_unsafe_rejection():
    from sttw_control.preference_governor import select
    r=rows()
    assert select(r,0,1)==0 and select(r,1,1)==1
    r['feasible'][0]=False
    assert select(r,0,1)==1

def test_slacks_ties_nonfinite_and_emergency():
    from sttw_control.preference_governor import select
    r=rows();r['steer_rmse']=np.array([0.,.005,.00501]);r['speed_rmse']=np.array([.3,.02,0.])
    assert select(r,0,1)==1
    r['speed_rmse']=np.array([0.,.02,.02001]);r['steer_rmse']=np.array([.3,.02,0.])
    assert select(r,1,1)==1
    r['steer_rmse'][:]=0;r['speed_rmse'][:]=0
    assert select(r,0,1)==0
    r['finite'][0]=False;r['speed_rmse'][0]=np.nan
    assert select(r,0,1)==1
    r['finite'][:]=False
    assert select(r,1,1) is None
    r=rows();r['feasible'][:]=False;r['physical_failure'][0]=True;r['violation']=np.array([0,2,1])
    assert select(r,0,3)==2 and select(r,1,3)==2

def test_grid_shapes_fairness_and_recovery_mode():
    from sttw_control.preference_governor import coarse_grid,refine_grid,mode_update
    raw=np.array([2.6,.25]);c=coarse_grid(raw,raw,raw,.1,1,0.)
    assert c.goal.shape==(144,2) and c.bypass[0] and c.valid.all()
    assert c.goal[1:,0].min()>=1.6 and c.goal[1:,0].max()<=2.6
    assert c.goal[1:,1].min()==-.35 and c.goal[1:,1].max()==.35
    r=rows();r={k:np.resize(v,144) for k,v in r.items()}
    fine=refine_grid(c,r,raw,1,0.)
    assert fine.goal.shape==(80,2) and not fine.valid[75:].any()
    assert mode_update(False,False,50,2,.2,False)[0]==1
    assert mode_update(True,True,2,0,.2,False)[0]==0
    assert mode_update(True,True,3,0,.2,False)[0]==2
    assert mode_update(True,False,10,2,.2,False)[0]==0
    assert mode_update(True,True,5,2,.02,True)[0]==0

def test_persistent_budget_never_resets(tmp_path):
    from sttw_control.teleop_budget import Budget,BudgetExhausted
    b=Budget(tmp_path,500);b.charge(predictor=240)
    b=Budget(tmp_path,80000000);assert b.charge()['predictor_ticks']==240
    with pytest.raises(BudgetExhausted):b.charge(predictor=300)
    assert b.charge()['predictor_ticks']==240

def test_prediction_validity_accepts_readonly_device_views():
    from sttw_control.closed_loop_predictor import mask_validity
    a=np.ones(3,bool);a.flags.writeable=False
    out=mask_validity(dict(feasible=a,finite=a),np.array([True,False,True]))
    np.testing.assert_array_equal(out['feasible'],[True,False,True])
    np.testing.assert_array_equal(a,[True,True,True])
