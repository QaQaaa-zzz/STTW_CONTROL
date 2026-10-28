import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('goal_replanning_review',Path(__file__).parents[1]/'cli/review_goal_replanning.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_distance_checks_segments_beyond_closest_vertex():
    path={'xy':np.array([[0,0],[100,0],[49,1],[49,2]])}
    np.testing.assert_allclose(review.original_distance([[50,.1]],path),[.1])


def test_distance_chunking_and_duplicate_vertices():
    path={'xy':np.array([[0,0],[0,0],[2,0],[2,2]])}
    xy=np.array([[1,.3],[3,1],[-1,0],[2,2]])
    np.testing.assert_allclose(review.original_distance(xy,path,chunk_size=1),[.3,1,1,0])
    assert review.original_distance(np.empty((0,2)),path).shape==(0,)


def test_runtime_guard_allows_only_declared_integration_files():
    core='/repo/learning/src/sttw_control/teleop_env.py'
    runner='/repo/learning/src/sttw_control/goal_endpoint_run.py'
    a={'sources':{core:'same',runner:'old'}}
    b={'sources':{core:'same',runner:'new','/repo/learning/src/sttw_control/goal_endpoint_replanning.py':'new'}}
    review.validate_shared_sources(a,b)
    b['sources'][core]='changed'
    with pytest.raises(ValueError,match='shared runtime'):review.validate_shared_sources(a,b)


def test_runtime_guard_rejects_missing_or_extra_dependency():
    a={'sources':{'/repo/learning/src/sttw_control/goal_replanning.py':'same'}}
    with pytest.raises(ValueError):review.validate_shared_sources(a,{'sources':{}})
    b={'sources':dict(a['sources'],**{'/repo/other/goal_endpoint_run.py':'changed'})}
    with pytest.raises(ValueError):review.validate_shared_sources(a,b)


def test_load_pair_rejects_changed_raw_trajectory(tmp_path,monkeypatch):
    item={'file':'raw.npz','sha256':'incorrect','case':{'case_id':'one'}}
    np.savez(tmp_path/'raw.npz',actual_xy=np.zeros((1,2)))
    monkeypatch.setattr(review,'_read_panel',lambda root,n:({},[item],{},{}))
    monkeypatch.setattr(review,'_validate_h75_comparison',lambda *args:None)
    with pytest.raises(ValueError,match='trajectory changed'):review.load_pair(tmp_path,tmp_path)
