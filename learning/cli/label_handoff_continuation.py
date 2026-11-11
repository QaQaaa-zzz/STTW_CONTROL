"""Apply frozen nominal straight-ground envelope to existing straight-jump data."""
import argparse,collections,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.handoff_continuation import mounted_features,label_window
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True);p.add_argument('--envelope',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
envelope=json.loads(a.envelope.read_text());assert envelope['adopted'];bounds=np.array(envelope['absolute_bounds'])
sources={'source720':'chapter3_offroad_event_training/collection','source1440':'chapter3_offroad_distance_training/collection','source240':'chapter3_offroad_new_roots/source','rb400_240':'chapter3_offroad_new_roots/rb400'}
contract=dict(envelope=str(a.envelope),sources=sources,data_role='Retrospective DEV analysis; no new independent test; no changed original task labels',selection='Actual published straight_jump candidates with measured handoff and no prior road violation; retained exclusions counted',capture='Nominal envelope held continuously for0.2s, completed within1s after handoff; descriptive envelope entry, not validated capture',continuation='Observed2s, no original physical/domain failure or cumulative road violation, x progress>=1m measured from first posthandoff frame',censoring='Goal/timeout before2s without failure -> unknown, not negative; immediate task completion reported separately',scope='Only straight y=0 downstream at1.5m/s alpha0.5; no S/curved transfer or recoverability claims',budget='0 new physics,0training; CPU analysis of4 existing batches',frozen_before_analysis=True)
(a.output/'declaration.json').write_text(json.dumps(contract,indent=2)+'\n')
rows=[];groups=[];fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(name,rel) in zip(axs.flat,sources.items()):
    run=a.runs/rel;cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'evaluation/summary.json').read_text())['cases'];protocol=json.loads((run/'protocol.json').read_text())['cases']
    h=cfg['ground_handoff'];assert h['speed_m_s']==1.5 and h['alpha']==.5
    mount=json.loads(Path(cfg['asset_spec']).read_text())['mount_rotation_wxyz'];z=np.load(run/'evaluation/trajectories.npz');root=z['root_xyzw'];t=z['time'];valid=z['valid'];road=z['road_violation_history'];features=mounted_features(root,mount)
    records=json.loads((run/'evaluation/moving_planning.json').read_text())['records'];published={r['lane']:r for r in records if r['status']=='published'};roads=np.load(cfg['task_reference_path'])['xy'];counts=collections.Counter();local=[]
    for i,(c,q) in enumerate(zip(summary,protocol)):
        if not q['candidate'].startswith('straight_jump'):continue
        counts['requested_straight']+=1
        if i not in published:counts['unpublished']+=1;continue
        if c['handoff_s'] is None:counts['no_handoff']+=1;continue
        if z['event_road_violation'][i,3]:counts['prior_road_violation']+=1;continue
        route=roads[published[i]['selected']];postroad=route[route[:,0]>=z['event_root_xyzw'][i,3,0]-.5];assert len(postroad)>0 and abs(postroad[:,1]).max()<1e-5
        ids=np.flatnonzero(valid[:,i]);lab=label_window(t[ids],root[ids,i],features[ids,i],road[ids,i],bounds,c['handoff_s'],c['end_code'])
        row=dict(dataset=name,lane=i,root=q['root'],candidate=q['candidate'],query_x=q['x'],original_success=c['success'],end_code=c['end_code'],**lab);rows.append(row);local.append(row)
        k=ids[t[ids]>=c['handoff_s']-1e-6];color='tab:blue' if lab['continued_2s'] is True else 'tab:red' if lab['continued_2s'] is False else 'gray';ax.plot(root[k,i,0],root[k,i,1],color=color,alpha=.25,lw=.7);ax.scatter(*root[k[-1],i,:2],color=color,s=4)
    counts.update(eligible=len(local),original_goal=sum(r['original_success'] for r in local),envelope_entered=sum(r['envelope_entry_status']=='entered_held' for r in local),continued=sum(r['continued_2s'] is True for r in local),not_continued=sum(r['continued_2s'] is False for r in local),censored=sum(r['continued_2s'] is None for r in local),entered_then_not_continued=sum(r['envelope_entry_status']=='entered_held' and r['continued_2s'] is False for r in local))
    groups.append(dict(dataset=name,**counts));ax.axhline(0,color='k',ls='--');ax.axhline(.6,color='k',ls=':');ax.axhline(-.6,color='k',ls=':');ax.set_aspect('equal',adjustable='datalim');ax.set(title=f'{name}: {len(local)} eligible handoffs',xlabel='world x (m)',ylabel='world y (m)');ax.grid(alpha=.2)
fig.suptitle('All eligible straight-jump suffixes: blue2s continuation,red failure,gray censored\nFixed original y=0 reference; endpoints retained; original road half-width0.6m')
fig.savefig(a.output/'xy_all.png',dpi=160);plt.close(fig)
(a.output/'labels.json').write_text(json.dumps(dict(groups=groups,cases=rows),indent=2)+'\n')
lines=['# Frozen-envelope retrospective handoff labels','','![All eligible actual/reference suffixes](xy_all.png)','','No new physics or training. Straight-jump subset only; not a replacement for full-task denominators. Nominal-envelope entry is not yet validated capture.','','|Dataset|Eligible|Original goal|Envelope held|Continue2s|Fail2s|Censored|Held then fail2s|','|---|---:|---:|---:|---:|---:|---:|---:|']
for g in groups:lines.append('|'+ '|'.join(str(g.get(k,0)) for k in ['dataset','eligible','original_goal','envelope_entered','continued','not_continued','censored','entered_then_not_continued'])+'|')
lines+=['','Full exclusions and per-source labels: labels.json. Criteria were fixed in declaration.json before label evaluation. Nominal calibration was separate from these jump outcomes, but this is retrospective analysis of previously inspected data, not independent TEST. Early goals without2s observation are censored for2s continuation while retaining original_success=true.','', 'The nominal envelope achieved11/12 coverage on its own holdout; absence of envelope entry must not be called physically unrecoverable. Perturbation margins and rescuability need matched physical branching. Root-only snapshots are not used.']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(groups,indent=2))
