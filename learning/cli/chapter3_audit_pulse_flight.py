"""CPU archive audit of push timing versus the manuscript's measured step-crossing flight."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from report_platform_controls import flight, runs


BASE = Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/'
            'platform_test0_20261120/approach_alignment/replication_wide/chapter_mainline/post_review')
OUT = Path('docs/evidence/chapter3_continuous_planning/p0_a')
KEY = ['layout','root','approach','approach_alpha','force']
PANELS = ('veto_fallback','veto_fallback_followup')


def selected(base):
    df = pd.concat([pd.read_csv(base/p/'oracle_cases.csv').assign(panel=p) for p in PANELS])
    df = df[(df.skill=='rb_run1') & df.layout.isin(('config_r','config_double','config_l'))]
    grouped = df.groupby(KEY)
    a = grouped.agg(tags=('trajectory','size'), paths=('trajectory','nunique'), prefixes=('prefix','nunique'),
                    verification=('verified','nunique'),onsets=('force_onset','nunique'),ticks=('force_steps','nunique'))
    if len(a)!=4500 or not ((a==[18,3,1,1,1,1]).all().all()):
        raise ValueError('Unexpected oracle branch identities or task-label multiplicities')
    # One physical path per design cell, with the straight suffix. Gate labels on this row
    # never enter the physics. Every included exit has the same archived pre-handoff digest.
    return df[df.planned_dep==1].drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)


def interval_overlap(a,b,c,d):
    return max(0,min(b,d)-max(a,c))


def audit(records):
    models, configs, metadata, cache = {}, {}, {}, {}
    audited = []
    for idx,r in enumerate(records.itertuples(index=False)):
        path = Path(r.trajectory)
        folder = path.parents[1]
        if folder not in models:
            models[folder]=mujoco.MjModel.from_binary_path(str(folder/'terrain.mjb'))
            configs[folder]=json.loads((folder/'resolved_config.json').read_text())['ditch_terrain']
            metadata[folder]={row['case_dir']:row for row in json.loads((folder/'status.json').read_text())['results']}
        m,terrain=models[folder],configs[folder]
        row=metadata[folder][path.parent.name]
        variant=row['variant']
        if (r.force,r.force_steps,r.force_onset)!=(variant['force_y_N'],row['applied_force_ticks'],row['air_event_tick']):
            raise ValueError(f'Force timing disagrees with original physical record: {path}')
        with np.load(path) as z:
            q,v,trace,loads,support,world=[z[k] for k in ('qpos','qvel','trace','contact_substep','wheel_contact_flags','world_flags')]
        t=trace[:,0]
        tick=np.rint(t/.005).astype(int)
        if tick[0]!=0 or not np.all(np.diff(tick)==1) or loads.shape[1:]!=(5,4):
            raise ValueError(f'Unexpected telemetry time base: {path}')
        h=row['handoff_s'];begin=row['rb_entry_s']
        end=len(q) if h is None else round(h/.005)+2
        state_hash=hashlib.sha256(q[:end].tobytes()+v[:end].tobytes()).hexdigest()
        if r.prefix!=state_hash:
            raise ValueError(f'Physical state digest mismatch: {path}')
        # Only pool the timings when all physical inputs to the event evidence agree. Walls
        # lie downstream; check their contact flag too rather than assuming map labels repeat.
        telemetry_hash=hashlib.sha256(loads[:end].tobytes()+support[:end,2].tobytes()+world[:end,2].tobytes()).hexdigest()
        key=(state_hash,telemetry_hash,begin,h,terrain['front_x_m'],terrain['far_bank_height_m'],r.force_onset,r.force_steps,r.force)
        if key not in cache:
            supported=support[:end,2] & (loads[:end,-1,:2]>1e-5).all(1)
            take,leave,land=flight(m,q[:end],loads[:end],supported,t[:end],terrain['front_x_m'],terrain['front_x_m'],
                                   terrain['far_bank_height_m'],np.inf if begin is None else begin)
            has=take is not None and leave is not None
            verified=bool(land is not None and h is not None and land<=h+1e-5)
            requested=100 if r.force_onset>=0 else 0
            observed=min(requested,max(0,int(tick[-1]-r.force_onset)*5)) if requested else 0
            # Zero force denotes the same trigger and a virtual window only. No force was applied.
            actual=(int(r.force_steps)*5 if r.force else 0)
            if actual>observed:
                raise ValueError(f'Force active outside recorded trajectory: {path}')
            t0=int(r.force_onset*5)
            first,last=(round(take*200)*5,round(leave*200)*5) if has else (None,None)
            fine_start=fine_end=None
            if has:
                free=((loads[:end,:,:3]<=1e-5).all(2)&(loads[:end,:,3]==0)).ravel()
                matches=[(b-5,e-5) for b,e in runs(free) if (b+4)//5==first//5 and (e-1)//5==last//5]
                if len(matches)!=1:
                    raise ValueError(f'Ambiguous flight window in contact telemetry: {path}')
                fine_start,fine_end=matches[0]
            active=actual if r.force else observed
            budget=active if r.force else requested
            overlap=(interval_overlap(t0,t0+active,first,last) if requested and has else None)
            exact=(interval_overlap(t0,t0+active,fine_start,fine_end) if requested and has else None)
            requested_overlap=(interval_overlap(t0,t0+requested,fine_start,fine_end) if requested and has else None)
            status=('no_trigger' if not requested else 'no_crossing_flight' if not has else 'not_applied' if not active else
                    'zero_overlap' if exact==0 else 'full_overlap' if exact==budget else 'partial_overlap')
            # A telemetry row t contains the interval (t-5 ms,t]; the queried pulse starts
            # at tick n, so its first five force samples belong to row n+1.
            s=loads[r.force_onset+1:r.force_onset+1+(active+4)//5].reshape(-1,4)[:active] if active else np.empty((0,4))
            contact=((s[:,:3]>1e-5).any(1)|(s[:,3]!=0))
            cache[key]=dict(prefix=state_hash,telemetry=telemetry_hash,rb_entry_s=begin,handoff_s=h,
                verified=verified,has_crossing_flight=has,flight_start_s=None if not has else first/1000,
                flight_last_sample_s=None if not has else last/1000,flight_raw_start_s=None if not has else fine_start/1000,
                flight_raw_end_s=None if not has else fine_end/1000,loaded_s=land,trigger_tick=r.force_onset,
                triggered=bool(requested),trigger_s=t0/1000 if requested else None,requested_window_ms=requested,
                observed_marker_ms=observed,actual_applied_ms=actual,requested_overlap_ms=requested_overlap,
                overlap_ms=exact,overlap_s=None if exact is None else exact/1000,paper_grid_overlap_ms=overlap,
                actual_force_overlap_ms=(exact if r.force else 0),
                in_window_contact_substeps=int(contact.sum()),in_window_any_contact=bool(contact.any()),
                in_window_peak_normal_N=float(s[:,:3].max()) if len(s) else None,outcome=status,
                window_type='actual_push' if r.force else 'trigger_marker_only',
                unique_evidence=hashlib.sha256(repr(key).encode()).hexdigest())
        data=cache[key]
        if bool(r.verified)!=data['verified']:
            raise ValueError(f'Original verified landing disagrees with replay: {path}')
        audited.append(dict(panel=r.panel,**{k:getattr(r,k) for k in KEY},**data,
                            stage_termination=row['termination'],trajectory=str(path),force_record=str(folder/'status.json'),case_dir=path.parent.name))
        if idx and idx%500==0:print('cells',idx,'unique timings',len(cache),flush=True)
    return pd.DataFrame(audited)


def example(base):
    rows=[]
    source=base/'realtime_15ms/test_a'
    for i in range(3):
        root=source/f'part_{i}/rb_run1'
        row=next(r for r in json.loads((root/'status.json').read_text())['results']
                 if r['variant']['bank_row']==1001 and r['route']==0 and r['variant']['planner']=='oracle'
                 and r['variant']['planned_departure']==1)
        p=root/row['case_dir']/'trajectory.npz'; h=row['handoff_s'];end=round(h/.005)+2
        with np.load(p) as z: prefix=hashlib.sha256(z['qpos'][:end].tobytes()+z['qvel'][:end].tobytes()).hexdigest()
        rows.append(dict(layout='test_a',panel='realtime_15ms',root=1001,approach=0,approach_alpha=0,
            force=row['variant']['force_y_N'],force_steps=row['applied_force_ticks'],force_onset=row['air_event_tick'],
            trajectory=str(p),prefix=prefix,verified=True))
    return audit(pd.DataFrame(rows))


def case_figure(case,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(xy,force,load)=plt.subplots(3,1,figsize=(8,7.5),layout='constrained')
    colors={-5:'#a74728',0:'#555555',5:'#116698'}
    for r in case.sort_values('force').itertuples(index=False):
        with np.load(r.trajectory) as z:
            t=z['trace'][:,0];q=z['qpos'];reference=z['reference_xy'];f=z['contact_substep']
        ti=np.rint(t/.005).astype(int);mask=(ti>=round(r.trigger_s/.005)-8)&(ti<=round(r.loaded_s/.005)+8)
        label=f'{r.force:+g} N';color=colors[r.force]
        xy.plot(q[mask,0],q[mask,1],color=color,label=label)
        if r.force:
            force.step(t[mask],np.where((ti[mask]>=r.trigger_tick)&(ti[mask]<r.trigger_tick+r.actual_applied_ms//5),
                                        r.force,0.),where='post',color=color,label=label)
            ft=(ti[mask,None]*.005-.005+(np.arange(5)+1)*.001).ravel()
            load.plot(ft,f[mask,:,:2].sum(2).ravel(),color=color,alpha=.83,lw=.85,label=label)
        for when,marker in ((r.trigger_s,'o'),(r.flight_start_s,'^'),(r.loaded_s,'s')):
            j=int(np.abs(t-when).argmin());xy.plot(q[j,0],q[j,1],marker=marker,color=color,ms=4)
    xy.plot(reference[:,0],reference[:,1],':',color='black',alpha=.75,label='Ground reference')
    xy.axvline(10.3,color='#777777',ls='--',lw=.8)
    qr=case.iloc[0];floor=round(qr.trigger_s/.005)-8
    xy.set(xlim=(float(q[floor,0])-.03,float(q[mask,0][-1])+.03),ylim=(-.50,.06),aspect='equal',
           xlabel='World x (m)',ylabel='World y (m)',title='TEST A · root 1001 · RB · baseline-right approach · original straight exit')
    xy.legend(ncol=4,fontsize=8)
    for ax in (force,load):
        ax.axvspan(qr.trigger_s,qr.trigger_s+.100,color='#dcc48e',alpha=.30,label='100 ms trigger window')
        ax.axvspan(case.flight_start_s.max(),case.flight_last_sample_s.min(),color='#98c9ae',alpha=.27,
                   label='Common step-crossing flight')
        ax.set_xlim(qr.trigger_s-.04,case.loaded_s.max()+.04);ax.grid(axis='y',alpha=.2)
    force.set(ylabel='World-y push (N)',ylim=(-6,6));force.legend(ncol=2,fontsize=8)
    load.set(ylabel='Wheel normal load (N)',xlabel='Time (s)')
    fig.savefig(out/'pulse_test_a_1001_xy_force.png',dpi=160)
    plt.close(fig)


def totals(data):
    records=[]
    matched=data.groupby(['root','approach','approach_alpha','force'])
    decisions=['prefix','trigger_tick','actual_applied_ms','flight_raw_start_s','flight_raw_end_s',
               'loaded_s','handoff_s','verified','overlap_ms','paper_grid_overlap_ms','in_window_contact_substeps']
    if len(matched)!=1500 or not (matched[decisions].nunique(dropna=False)==1).all().all():
        raise ValueError('Cannot collapse layouts: physical state or event conclusions differ')
    for label,frame in [('all_layout_cells',data),('unique_physical_prefix',data.drop_duplicates('prefix')),
                        ('unique_raw_telemetry',data.drop_duplicates('unique_evidence')),
                        *data.groupby('layout')]:
        for force,z in frame.groupby('force'):
            for population,g in [('all_attempts',z),('verified_jump_return_only',z[z.verified])]:
                counts=g.outcome.value_counts()
                records.append(dict(scope=label,population=population,force=force,n=len(g),
                    **{k:int(counts.get(k,0)) for k in ('no_trigger','no_crossing_flight','not_applied','zero_overlap',
                        'partial_overlap','full_overlap')},
                    grid_zero=int((g.has_crossing_flight & (g.paper_grid_overlap_ms==0)).sum()),
                    grid_partial=int(((g.paper_grid_overlap_ms>0)&(g.paper_grid_overlap_ms<100)).sum()),
                    grid_full=int((g.paper_grid_overlap_ms==100).sum()),
                    trigger_window_with_contact=int(g.in_window_any_contact.sum()),
                    cross_flight_with_contact=int((g.in_window_any_contact & g.has_crossing_flight).sum()),
                    with_main_flight=int(g.has_crossing_flight.sum()),verified_return=int(g.verified.sum())))
    return pd.DataFrame(records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    fields=selected(args.base);data=audit(fields);counts=totals(data);case=example(args.base)
    args.output.mkdir(parents=True,exist_ok=True)
    data.to_csv(args.output/'pulse_flight_attempts.csv.gz',index=False)
    counts.to_csv(args.output/'pulse_flight_counts.csv',index=False)
    case.to_csv(args.output/'pulse_test_a_1001.csv',index=False)
    case_figure(case,args.output)
    print(counts[counts.scope=='unique_physical_prefix'].to_string(index=False),flush=True)
