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


def test_accepts_prefix_ending_in_recorded_failure(tmp_path):
    paths=pair(tmp_path)
    tr=dict(np.load(paths[1]/'trace.npz'));tr={k:v[:2] for k,v in tr.items()}
    tr['terminated']=np.array([False,True]);tr['end_code']=np.array([0,1])
    np.savez(paths[1]/'trace.npz',**tr)
    traces=verify_pair(paths)
    assert [len(t['time']) for t in traces]==[3,2]


def test_failure_does_not_allow_misaligned_time_grid(tmp_path):
    paths=pair(tmp_path)
    tr=dict(np.load(paths[1]/'trace.npz'));tr={k:v[:2] for k,v in tr.items()}
    tr['time']=np.array([0.,.004]);tr['terminated']=np.array([False,True])
    np.savez(paths[1]/'trace.npz',**tr)
    with pytest.raises(ValueError,match='time grids'):verify_pair(paths)


def test_terminal_padding_encodes_full_comparison(tmp_path):
    import subprocess
    from sttw_control.panel_media import comparison_filter
    graph,count=comparison_filter([{'frame_count':10},{'frame_count':5}],
                                  [{'time':np.array([0.,.9])},{'time':np.array([0.,.4])}])
    assert count==10 and 'TERMINATED' in graph and 'stop=5' in graph
    out=tmp_path/'paired.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=320x160:r=10:d=1',
                    '-f','lavfi','-i','color=c=red:s=320x160:r=10:d=0.5',
                    '-filter_complex_threads','1','-filter_complex',graph,'-map','[v]',
                    '-c:v','libx264','-threads','1',str(out)],check=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=nb_read_frames,width','-of','json',str(out)],text=True))['streams'][0]
    assert int(info['nb_read_frames'])==10 and info['width']==640
