import numpy as np
import jax
import jax.numpy as jp
from sttw_control import timed_reference as tr


def test_geometric_projection_tracks_lagged_curve_without_time_penalty():
    assert hasattr(tr, 'project_committed_geometry'), 'geometric projection must be implemented'
    c=tr.TimedReferenceConfig(fixed=((0.,2.,.5),))
    rows=jp.asarray(c.fixed)
    table=tr.committed_geometry_table(jp.zeros(3),jp.array([2.,0.]),rows,.02,c,100)
    pose=np.asarray(table[30,1:4]);progress=float(table[29,0])
    features,s,index=tr.project_committed_geometry(pose,table,29,progress,.041,80)
    np.testing.assert_allclose(features[:2],0,atol=2e-5)
    assert float(s)>progress
    assert abs(float(tr.errors(pose,table[80,1:4],table[80,4:])[3]))>1
    compiled=jax.jit(tr.project_committed_geometry)(jp.asarray(pose),table,jp.asarray(29),jp.asarray(progress),jp.asarray(.041),jp.asarray(80))
    np.testing.assert_allclose(compiled[0],features,atol=1e-6)


def test_projection_cannot_read_future_or_jump_branches():
    assert hasattr(tr, 'project_committed_geometry'), 'geometric projection must be implemented'
    c=tr.TimedReferenceConfig(fixed=((0.,2.,0.),))
    table=tr.committed_geometry_table(jp.zeros(3),jp.array([2.,0.]),jp.asarray(c.fixed),.02,c,100)
    changed=table.at[21:,1:4].set(999.)
    args=(jp.array([1.,0.,0.]),)
    a=tr.project_committed_geometry(*args,table,19,.76,.08,20)
    b=tr.project_committed_geometry(*args,changed,19,.76,.08,20)
    np.testing.assert_allclose(a[0],b[0]);assert float(a[1])<=.80001
    # A distant segment placed directly on the car cannot become its branch.
    loop=table.at[50:60,1:3].set(jp.array([.8,0.]))
    result=tr.project_committed_geometry(jp.array([.8,0.,0.]),loop,19,.76,.04,100)
    assert float(result[1])<=.80001


def test_inner_curve_projection_can_advance_faster_than_vehicle_distance():
    # Unit circle, car follows radius .8: path arc advances .01 while car moves .008.
    angles=np.arange(201)*.01
    table=np.column_stack((angles,np.sin(angles),1-np.cos(angles),angles,np.ones(201),np.ones(201)))
    pose=np.array([.8*np.sin(.51),1-.8*np.cos(.51),.51])
    f,s,_=tr.project_committed_geometry(pose,table,49,.50,.008,150,xp=np)
    assert abs(s-.51)<.001
    assert abs(f[0]+.2)<.001


def test_geometric_audit_rejects_tampered_path_feature():
    import pytest
    from sttw_control.tracking_diagnostics import audit_committed_reference
    ref=np.column_stack((np.arange(11)*.01,np.zeros(11),np.zeros(11)))
    trace={'reference_pose':ref,'reference_command':np.tile([2.,0.],(11,1)),
           'pose':ref.copy(),'path_features':np.zeros((11,3)),'path_progress':ref[:,0]}
    assert audit_committed_reference(trace,.005)<1e-9
    trace['path_features'][5,0]=.1
    with pytest.raises(ValueError,match='projection mismatch'):audit_committed_reference(trace,.005)


def test_geometric_audit_replays_float32_nearest_segment_tie():
    # Frozen baseline prefix: a large lateral offset creates a float32 distance
    # tie between adjacent segments. NumPy picks the neighbor at step 1522.
    from pathlib import Path
    import pytest
    from sttw_control.tracking_diagnostics import audit_committed_reference
    with np.load(Path(__file__).parent / "fixtures/geometric_projection_float32_tie.npz") as saved:
        trace = {k: saved[k] for k in saved.files}
    assert audit_committed_reference(trace, .005) < 1e-6
    trace['path_features'][1522, 0] += .01
    with pytest.raises(ValueError, match='projection mismatch'):
        audit_committed_reference(trace, .005)
