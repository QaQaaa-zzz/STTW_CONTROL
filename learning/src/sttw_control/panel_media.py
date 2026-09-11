"""Complete paired media for a declared panel, using recorded trajectories only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .media import compare_runs, frame_indices, validate_trace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pair(paths):
    declarations=[json.loads((p/'declaration.json').read_text()) for p in paths]
    for key in ('config','model','seed','backend','reset_semantics'):
        if declarations[0][key]!=declarations[1][key]:raise ValueError('paired media mismatch: '+key)
    traces=[dict(np.load(p/'trace.npz',allow_pickle=False)) for p in paths]
    for trace in traces:validate_trace(trace)
    for key in ('qpos','qvel'):
        if not np.array_equal(traces[0][key][0],traces[1][key][0]):raise ValueError('paired initial state mismatch')
    n=min(len(t['time']) for t in traces)
    if not np.array_equal(traces[0]['time'][:n],traces[1]['time'][:n]):
        raise ValueError('paired time grids differ')
    for trace in traces:
        if len(trace['time'])<max(len(t['time']) for t in traces):
            if 'terminated' not in trace or not bool(trace['terminated'][-1]):
                raise ValueError('paired time grids differ without recorded termination')
    return traces


def verify_media(path):
    folder=path/'media';m=json.loads((folder/'manifest.json').read_text())
    if m.get('status')!='complete':raise ValueError('incomplete media: '+str(folder))
    for key,name in [('source_trace_sha256','trace.npz'),('source_declaration_sha256','declaration.json')]:
        if m[key]!=digest(path/name):raise ValueError('stale media: '+str(folder))
    if len(m['frame_indices'])!=m['frame_count']:raise ValueError('frame count/mapping mismatch')
    for name in ('replay.mp4','states.png','trajectory.png'):
        if not (folder/name).is_file() or (folder/name).stat().st_size==0:raise ValueError('missing media: '+name)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'replay.mp4')],text=True))['streams'][0]
    num,den=map(int,probe['r_frame_rate'].split('/'))
    if int(probe['nb_read_frames'])!=m['frame_count'] or not np.isclose(num/den,m['fps']):raise ValueError('encoded video disagrees with manifest')
    return m


def comparison_filter(manifests,traces):
    a,b=manifests
    count=max(a['frame_count'],b['frame_count'])
    filters=[]
    for i,(m,tr) in enumerate(zip((a,b),traces)):
        missing=count-m['frame_count']
        chain=f'[{i}:v]'
        if missing:
            label=f"TERMINATED at {float(tr['time'][-1]):.3f}s - frozen last recorded frame"
            chain+=f"tpad=stop_mode=clone:stop={missing},drawtext=text='{label}':x=20:y=30:fontsize=26:fontcolor=white:box=1:boxcolor=red@0.85:enable='gte(n,{m['frame_count']-1})'"
        else:chain+='null'
        filters.append(chain+f'[side{i}]')
    filters.append('[side0][side1]hstack=inputs=2[v]')
    return ';'.join(filters),count


def export_panel(panel_run,output,*,workers=2):
    panel_run,output=Path(panel_run).resolve(),Path(output).resolve()
    declaration=json.loads((panel_run/'declaration.json').read_text())
    cases=[name for name in declaration['scenarios'] if name!='nominal']
    if not cases:raise ValueError('empty disturbance panel')
    pairs={name:[panel_run/name/label for label in ('baseline','residual')] for name in cases}
    for name,paths in pairs.items():
        if Path(name).name!=name:raise ValueError('invalid scenario path')
        verify_pair(paths)
        for path in paths:
            d=json.loads((path/'declaration.json').read_text())
            if d['config']!=declaration['scenarios'][name] or d['seed']!=declaration['panel']['seed']:raise ValueError('scenario differs from panel declaration')
        if json.loads((paths[1]/'declaration.json').read_text())['policy']!=declaration['policy']:raise ValueError('mixed candidate policies in panel')
    output.mkdir(parents=True,exist_ok=False)
    def render(path):
        if not (path/'media').exists():
            env=os.environ.copy();env['MUJOCO_GL']='egl';env['JAX_PLATFORMS']='cpu'
            with (output/(path.parent.name+'_'+path.name+'.log')).open('x') as log:
                subprocess.run([sys.executable,'learning/cli/render.py','--run',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return verify_media(path)
    paths=[path for pair in pairs.values() for path in pair]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifests=dict(zip(paths,pool.map(render,paths)))
    results={}
    for name,pair in pairs.items():
        a,b=[manifests[p] for p in pair]
        if a['fps']!=b['fps']:raise ValueError('video frame rates differ')
        traces=verify_pair(pair)
        for trace,m in zip(traces,(a,b)):
            if not np.array_equal(m['frame_indices'],frame_indices(trace['time'],m['fps'])):raise ValueError('stale frame mapping')
        dest=output/name;dest.mkdir()
        compare_runs(*pair,dest,candidate_label='Frozen residual '+Path(declaration['checkpoint']).name)
        graph,count=comparison_filter((a,b),traces)
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(pair[0]/'media/replay.mp4'),'-i',str(pair[1]/'media/replay.mp4'),'-filter_complex_threads','1','-filter_complex',graph,'-map','[v]','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'comparison.mp4')],check=True)
        results[name]={'runs':[str(p) for p in pair],'trace_sha256':[digest(p/'trace.npz') for p in pair],'video_sha256':digest(dest/'comparison.mp4'),'fps':a['fps'],'frames':count,'source_frames':[a['frame_count'],b['frame_count']],'terminal_times_seconds':[float(t['time'][-1]) for t in traces],'terminal_frame_padding':[count-m['frame_count'] for m in (a,b)],'padding_semantics':'visual freeze with termination label only; no extrapolated states','event':json.loads((pair[1]/'event.json').read_text())}
        print('Complete paired media: '+name,flush=True)
    manifest={'status':'complete','panel_run':str(panel_run),'panel_declaration_sha256':digest(panel_run/'declaration.json'),'checkpoint':declaration['checkpoint'],'seed':declaration['panel']['seed'],'selection':'one predeclared seed and one frozen policy for every scenario; not best-seed selection','cases':results}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Standard disturbance comparison','',f"Frozen policy: {declaration['checkpoint']}; seed {declaration['panel']['seed']}. Left: baseline; right: residual. Event timing is overlaid. No physics rerun.",'']
    lines += [f'- {name}: [video]({name}/comparison.mp4) | [states and trajectory]({name}/comparison.png)' for name in cases]
    (output/'INDEX.md').write_text('\n'.join(lines)+'\n')
    return manifest
