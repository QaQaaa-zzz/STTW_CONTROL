"""Freeze empirical state envelopes for periodic finite-library planning."""
import argparse,copy,json,shutil,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--prefix-training',type=Path)
p.add_argument('--ground-training',type=Path,nargs='*',default=[])
a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
cs=json.loads((a.source/'candidates.json').read_text());sources=json.loads((a.source/'sources.json').read_text());cal=json.loads((a.source/'calibration.json').read_text());original=json.loads((a.source/'manifest.json').read_text());plans=json.loads((a.source/'plans.json').read_text())
x=np.arange(1.45,18.551,.025,dtype=np.float32);edges=np.array(cal['region_edges_x_m'],np.float32)
def read(path):
    with np.load(path) as d:return d['qpos'][:,0],np.concatenate((d['qpos'][:,1:7],d['qpos'][:,8:9],d['qpos'][:,10:12],d['qvel'],d['controller_state']),1),d['trace'][:,-1]
def interp(q,xx,values):return np.stack([np.interp(q,xx,values[:,i]) for i in range(values.shape[1])],1)
lows=[];highs=[];valid=[];plo=[];phi=[];raw=[]
for candidate in cs:
    data=[read(r['path']) for r in sources if r['candidate']==candidate['index']];raw.append(data)
    sampled=[interp(x,xx,v) for xx,v,phase in data];lows.append(np.minimum.reduce(sampled));highs.append(np.maximum.reduce(sampled))
    valid.append(np.logical_and.reduce([(x>=xx[0])&(x<=xx[-1]) for xx,v,phase in data]))
    phase=[np.interp(x,xx,phase) for xx,v,phase in data];plo.append(np.floor(np.minimum.reduce(phase)));phi.append(np.ceil(np.maximum.reduce(phase)))
low=np.array(lows,np.float32);high=np.array(highs,np.float32);margin=np.zeros((len(edges)-1,low.shape[-1]));counts=np.zeros(len(margin),int)
caldir=Path(original['calibration']);status=json.loads((caldir/'status.json').read_text());records=[]
for row in status['results']:
    if row['variant']['data_split']=='TRAIN':continue
    xx,values,phase=read(caldir/row['case_dir']/'trajectory.npz');data=raw[9+row['route']]
    shared=(xx>=max(d[0][0] for d in data))&(xx<=min(d[0][-1] for d in data))&np.isin(phase,[0,2])
    lo=np.minimum.reduce([interp(xx,sx,v) for sx,v,sp in data]);hi=np.maximum.reduce([interp(xx,sx,v) for sx,v,sp in data]);miss=np.maximum(np.maximum(lo-values,values-hi),0)
    regions=np.clip(np.searchsorted(edges,xx,side='right')-1,0,len(margin)-1)
    for region in range(len(margin)):
        mask=shared&(regions==region)
        if mask.any():margin[region]=np.maximum(margin[region],miss[mask].max(0));counts[region]+=int(mask.sum())
    records.append(dict(case=row['case_dir'],covered_ground_rows=int(shared.sum())))
margin[counts==0]=margin.max(0);floor=(high.max((0,1))-low.min((0,1)))*.01+1e-5;margin=margin*1.2+floor
prefix_sources=[]
if a.prefix_training:
    prefix_status=json.loads((a.prefix_training/'status.json').read_text())
    for row in prefix_status['results']:
        if row['variant']['data_split']!='TRAIN':continue
        sx,values,phase=read(a.prefix_training/row['case_dir']/'trajectory.npz')
        mask=(x>=sx[0])&(x<=8.3)&(x<=sx[-1]);sampled=interp(x,sx,values)
        for candidate in cs:
            if candidate['route']==row['route']:
                i=candidate['index'];low[i,mask]=np.minimum(low[i,mask],sampled[mask]);high[i,mask]=np.maximum(high[i,mask],sampled[mask])
        prefix_sources.append(dict(case=row['case_dir'],route=row['route'],root=row['variant']['root_variant']))
ground_sources=[]
for source in a.ground_training:
    config=json.loads((source/'resolved_config.json').read_text());activation=config['ground_handoff']['approach']['jump_start_x_m']
    for row in json.loads((source/'status.json').read_text())['results']:
        if row['variant']['data_split']!='TRAIN' or row['variant']['requested_speed_m_s']!=1.8:continue
        sx,values,phase=read(source/row['case_dir']/'trajectory.npz');sampled=interp(x,sx,values);phase_at=np.interp(x,sx,phase);inside=(x>=sx[0])&(x<=sx[-1])
        for candidate in cs:
            if candidate['route']!=row['route']:continue
            i=candidate['index'];mask=inside&valid[i]&(((x<=8.3)&(phase_at==0))|((abs(candidate['activation_x_m']-activation)<1e-5)&(phase_at==2)))
            low[i,mask]=np.minimum(low[i,mask],sampled[mask]);high[i,mask]=np.maximum(high[i,mask],sampled[mask])
        ground_sources.append(dict(evaluation=str(source),case=row['case_dir'],route=row['route'],root=row['variant']['root_variant'],activation_x_m=activation))
np.savez(a.output/'state_bank.npz',x=x,low=low,high=high,valid=np.array(valid),phase_low=np.array(plo,np.float32),phase_high=np.array(phi,np.float32),edges=edges,margin=margin.astype(np.float32),routes=np.array([c['route'] for c in cs]),activation=np.array([c['activation_x_m'] for c in cs],np.float32),cost=np.array([c['cost_s'] for c in cs],np.float32))
entries=[]
for layout in range(6):
    source=next(p for p in plans if p['layout']==layout and p['method']=='joint');folder=a.output/f'layout{layout:02d}';folder.mkdir();config=json.loads((Path(source['evaluation']).parent/'config.json').read_text())
    config['planned_activation_offset_m']=8.55-10.3
    config['evaluation_lanes']=[[3,'h75',i] for i in range(2)]
    config['rolling_planner']=dict(state_bank=str(a.output/'state_bank.npz'),clearance=source['all_candidate_clearance_m'],period_s=.2,first_query_s=.2,initial_valid_until_x_m=8.3,role='DEV rolling finite library; measured physical/controller-state empirical screening, delayed publication, no skill reset')
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n');entries.append(dict(layout=layout,config=str(folder/'config.json'),evaluation=str(folder/'evaluation')))
manifest=dict(ground_training=[str(p) for p in a.ground_training],ground_sources=ground_sources,ground_scope='Additional1.8m/s TRAIN histories, matching route; phase0 identical reference prefix throughx8.3 may support all activations, phase2 only exact source activation. Original xvalid/phasebands/calibrated margins and geometric clearances unchanged.',role='P2 rolling DEV integration, not independent TEST or safety guarantee',source=str(a.source),entries=entries,ground_only=False,routes=list(range(9)),seconds=10,physical_episodes=12,initial_plan='Route3 left upstream/straight downstream; only prefix throughx8.3 validated initially, activation8.55. New complete plan chosen from measured state startingt0.2.',screening='Same-x source min/max state including381 controller features; existing18DEV max excursion by fixed map region times1.2 plus1percent TRAIN feature range and1e-5. Ground phases only; RB remains frozen.',calibration=records,counts=counts.tolist(),budget='One12episode queue, no training or parameter sweep',expected_wallclock_s=[300,900],hard_limit_s=1500,stop='First runtime error or25min; preserve failures',physics='Unchanged RB225/H75, implicitfast1ms x5, fixed map/goal,2m flat buffers',prefix_training=str(a.prefix_training) if a.prefix_training else None,prefix_sources=prefix_sources,prefix_scope='Optional additional TRAIN histories only on identical ground prefix x<=8.3 before every RB activation; original calibrated margins unchanged',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(a.output/'source').mkdir()
for path in [Path(__file__),Path('learning/src/sttw_control/playground_rolling.py'),Path('learning/src/sttw_control/playground_handoff_evaluation.py')]:shutil.copyfile(path,a.output/'source'/path.name)
print(json.dumps(dict(shape=low.shape,episodes=12,counts=counts.tolist())))
