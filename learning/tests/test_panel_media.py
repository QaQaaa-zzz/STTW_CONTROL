import json
import numpy as np
import pytest
from sttw_control.panel_media import verify_pair,verify_media


def pair(tmp_path):
    paths=[tmp_path/'baseline',tmp_path/'residual']
    for path in paths:
        path.mkdir()
        (path/'declaration.json').write_text(json.dumps({'config':{},'model':{},'seed':1,'backend':'cpu','reset_semantics':'test'}))
        np.savez(path/'trace.npz',time=[0.,.005,.01],qpos=np.zeros((3,7)),qvel=np.zeros((3,6)),measurement=np.zeros((3,7)),command=np.zeros((3,2)))
    return paths


def test_rejects_shorter_candidate_instead_of_silently_cutting_video(tmp_path):
    paths=pair(tmp_path);verify_pair(paths)
    tr=dict(np.load(paths[1]/'trace.npz'))
    np.savez(paths[1]/'trace.npz',**{k:v[:2] for k,v in tr.items()})
    with pytest.raises(ValueError,match='time grids'):verify_pair(paths)


def test_rejects_initial_state_mismatch(tmp_path):
    paths=pair(tmp_path);tr=dict(np.load(paths[1]/'trace.npz'));tr['qpos'][0,0]=1.
    np.savez(paths[1]/'trace.npz',**tr)
    with pytest.raises(ValueError,match='initial state'):verify_pair(paths)


def test_existing_render_cannot_be_reused_for_changed_trace(tmp_path):
    path=pair(tmp_path)[0];(path/'media').mkdir()
    (path/'media/manifest.json').write_text(json.dumps({'status':'complete','source_trace_sha256':'invalid'}))
    with pytest.raises(ValueError,match='stale media'):verify_media(path)
