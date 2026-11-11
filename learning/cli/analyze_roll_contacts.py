"""Offline wheel-contact reduction and cross-case roll-model diagnostics."""
import argparse, csv, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


def rotate(q,v):
    t=2*np.cross(q[...,:3],v)
    return v+q[...,3:]*t+np.cross(q[...,:3],t)


def roll_state(body):
    q=body[...,3:7];x,y,z,w=np.moveaxis(q,-1,0)
    raw=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    inv=q.copy();inv[...,:3]*=-1
    omega=rotate(inv,body[...,10:13])
    derivative=-(omega[...,0]+np.sin(raw)*np.tan(pitch)*omega[...,1]+np.cos(raw)*np.tan(pitch)*omega[...,2])
    return np.pi/2-raw,derivative


FIELDS=[f'{wheel}_{term}' for wheel in ('front','rear') for term in ('normal_N','friction_x_N','friction_y_N','friction_z_N','normal_roll_Nm','friction_roll_Nm')]+['steer_normal_Nm','steer_friction_Nm','steer_motor_Nm','steer_request_Nm','front_raised','rear_raised']


def extract(run,out,workers):
    raw=run/'batches/batch_000/physics_contacts';idx=json.loads((raw/'index.json').read_text())
    contracts=[(p.parent.name,json.loads(p.read_text())) for p in sorted(run.glob('*/contract.json')) if (p.parent/'trajectory.npz').exists()]
    lanes=max(c['shared_physics_contacts']['lane_start']+c['shared_physics_contacts']['lane_count'] for _,c in contracts)
    masses=np.zeros((lanes,6));coms=np.zeros((lanes,6,3))
    for _,c in contracts:
        start=c['shared_physics_contacts']['lane_start'];n=c['shared_physics_contacts']['lane_count']
        masses[start:start+n]=c['lab_physics']['masses'];coms[start:start+n]=np.asarray(c['lab_physics']['runtime_inertial']['coms'])[:,:3]
    names=idx['body_order'];ch=names.index('chasis');forkidx=names.index('frontfork');steer=idx['joint_order'].index('steering_joint')
    def chunk(entry):
        with np.load(raw/entry['file']) as z:
            body=z['body_state'];center=body[...,:3]+rotate(body[...,3:7],coms[None])
            center=(center*masses[None,:,:,None]).sum(2)/masses.sum(1)[None,:,None]
            rollaxis=-rotate(body[:,:,ch,3:7],np.array([1.,0.,0.]))
            fork=body[:,:,forkidx];steeraxis=rotate(fork[...,3:7],np.array([0.,1.,0.]))
            ticks=len(body);records=[];steermom=[];raised=[]
            for wheel in ('frontwheel','rearwheel'):
                values={}
                for kind in ('normal','friction'):
                    prefix=wheel+'_'+kind+'_';count=z[prefix+'count'][...,0];start=z[prefix+'start'][...,0]
                    slot=np.arange(int(count.max()));mask=slot<count[...,None]
                    point=z[prefix+'point'];ids=np.minimum(start[...,None]+slot,point.shape[1]-1);ti=np.arange(ticks)[:,None,None]
                    points=point[ti,ids];f=z[prefix+'force'][ti,ids]
                    if kind=='normal':
                        scalar=f[...,0];f=f*z[wheel+'_normal_direction'][ti,ids]
                        values['normal_N']=np.where(mask,scalar,0).sum(-1)
                        raised.append((mask&(scalar>1)&(points[...,2]>.002)).any(-1))
                    f=np.where(mask[...,None],f,0)
                    values[kind+'_moment']=(np.cross(points-center[:,:,None],f)*rollaxis[:,:,None]).sum((-1,-2))
                    values[kind+'_force']=f.sum(-2)
                    if wheel=='frontwheel':steermom.append((np.cross(points-fork[:,:,None,:3],f)*steeraxis[:,:,None]).sum((-1,-2)))
                records.extend([values['normal_N'],*np.moveaxis(values['friction_force'],-1,0),values['normal_moment'],values['friction_moment']])
            records.extend(steermom+[z['motor_torque'][:,:,steer],z['motor_requested_torque'][:,:,0],*raised])
            return z['physics_step'],z['alive_before'],np.stack(records,-1).astype(np.float32)
    sums=np.zeros((2000,lanes,len(FIELDS)),np.float64);counts=np.zeros((2000,lanes),np.int32)
    low=np.full(sums.shape,np.inf,np.float32);high=np.full(sums.shape,-np.inf,np.float32)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(steps,alive,values) in enumerate(pool.map(chunk,idx['chunks'])):
            bins=(steps-1)//25
            np.add.at(sums,bins,np.where(alive[...,None],values,0));np.add.at(counts,bins,alive.astype(np.int32))
            np.minimum.at(low,bins,np.where(alive[...,None],values,np.inf));np.maximum.at(high,bins,np.where(alive[...,None],values,-np.inf))
            if i%100==0:
                status=dict(stage='extract',completed=i+1,planned=len(idx['chunks']),elapsed_s=time.monotonic()-start)
                (out/'status.json').write_text(json.dumps(status));print(status,flush=True)
    mean=np.divide(sums,counts[...,None],out=np.full_like(sums,np.nan),where=counts[...,None]>0).astype(np.float32)
    low[counts==0]=np.nan;high[counts==0]=np.nan
    np.savez_compressed(out/'contact_bins.npz',mean=mean,minimum=low,maximum=high,valid_physics_steps=counts,fields=np.array(FIELDS))
    (out/'extraction.json').write_text(json.dumps(dict(raw_path=str(raw),physics_dt=.0002,bin_dt=.005,fields=FIELDS,raised_definition='normal force >1N and world z>2mm; not independent stone ID',moment_definition='sum (contact point - total vehicle COM) cross force projected on negative chassis local x; not Euler generalized force or terrain-only torque',cases=[dict(case=n,start=c['shared_physics_contacts']['lane_start'],labels=c['settings']['case_labels']) for n,c in contracts],elapsed_s=time.monotonic()-start),indent=2))
    (out/'status.json').write_text(json.dumps(dict(stage='extracted',elapsed_s=time.monotonic()-start)))
    print('extraction complete',flush=True)


def nominal_coeff(v,c):
    v=np.maximum(v,c['minimum_speed']);m=c['mass'];h=c['cg_height'];I=m*h*h+c.get('roll_inertia_com',.043929*m/5.434)
    a1=-m*c['cg_forward']*h*v*np.cos(c['caster'])/c['wheelbase']/I
    m2=-(m*v*v*h-m*c['cg_forward']*c['trail']*c['gravity'])*np.cos(c['caster'])/c['wheelbase']
    a2=np.where(m2<0,-1,1)*np.maximum(abs(m2),.1)/I
    return a1,a2,m*c['gravity']*h/I


def replay(phi,rate,u,valid,a4,horizon):
    # Exact constant-forcing transition for each5ms interval; measured future
    # steering/speed/contact are replay inputs, not a causal alpha-only forecast.
    n,l=phi.shape;h=round(horizon/.005);p=phi[:-h].copy();v=rate[:-h].copy();good=valid[:-h].copy()
    root=np.sqrt(a4);cs=np.cosh(root*.005);sn=np.sinh(root*.005)
    for k in range(1,h+1):
        forcing=u[k:n-h+k];good &= valid[k:n-h+k]&np.isfinite(forcing)
        p,v=cs*p+sn/root*v+(cs-1)/a4*forcing,root*sn*p+cs*v+sn/root*forcing
    result=np.full(phi.shape,np.nan);result[h:]=np.where(good,p,np.nan)
    return result


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def figures(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    import shutil
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.18,'figure.dpi':130})
    bins=np.load(out/'contact_bins.npz');means=bins['mean'];lows=bins['minimum'];highs=bins['maximum'];count=bins['valid_physics_steps']
    meta=json.loads((out/'extraction.json').read_text());config=json.loads(Path('runs/isaac_signals_parity/dynamic.json').read_text())
    cases=[];training=[];contacts_idx=[4,5,10,11];time_axis=(np.arange(2000)+1)*.005
    source_rows=json.loads((run/'analysis_summary.json').read_text())['rows']
    for row in meta['cases']:
        name=row['case'];c=json.loads((run/name/'contract.json').read_text());start=row['start'];n=len(row['labels']);sl=slice(start,start+n)
        with np.load(run/name/'trajectory.npz') as z:
            trace=z['lower_trace'].reshape(2000,n,11);valid=trace[:,:,6]>0;phi=z['measurement'][:,:,0];_,rate=roll_state(z['body_state'][:,:,c['bodies'].index('chasis')]);delta=z['measurement'][:,:,2];dr=z['measurement'][:,:,3];speed=trace[:,:,2]
            actual_time=np.arange(2000)[:,None]*.005+trace[:,:,7]*.0002
            a1,a2,a4=nominal_coeff(speed,config);forcing=a1*dr+a2*delta
            mean=means[:,sl];lo=lows[:,sl];hi=highs[:,sl];X=mean[:,:,contacts_idx]
            item=dict(name=name,seed=c['settings']['seed'],labels=row['labels'],phi=phi,rate=rate,speed=speed,delta=delta,dr=dr,valid=valid,forcing=forcing,a4=a4,X=X,mean=mean,lo=lo,hi=hi,time=actual_time,path=trace[:,:,3],command=z['final_command'][:,:,0],controller_request=z['controller_terms'][:,:,9],progress=z['progress'],pose=z['pose'],requested=c['settings']['user_speed_m_s'],contact_valid=count[:,sl]==25)
        for lane in range(16):
            ids=np.flatnonzero(valid[:,lane]&item['contact_valid'][:,lane])
            if len(ids)<30:continue
            # The native archive is contiguous from0.8s until each lane ends.
            acceleration=savgol_filter(rate[ids,lane],11,3,deriv=1,delta=.005)
            base=savgol_filter(forcing[ids,lane]+a4*phi[ids,lane],11,3)
            features=savgol_filter(X[ids,lane],11,3,axis=0)
            training.append(dict(seed=item['seed'],x=features[5:-5],y=(acceleration-base)[5:-5]))
        cases.append(item)
    models={}
    for held in sorted(set(c['seed'] for c in cases)):
        train=[x for x in training if x['seed']!=held];X=np.concatenate([x['x'] for x in train]);y=np.concatenate([x['y'] for x in train]);weight=np.concatenate([np.full(len(x['y']),1/len(x['y'])) for x in train]);weight/=weight.sum()
        center=(weight[:,None]*X).sum(0);scale=np.sqrt((weight[:,None]*(X-center)**2).sum(0));scale=np.maximum(scale,1e-8);A=np.column_stack((np.ones(len(X)),(X-center)/scale));penalty=np.diag([0,.001,.001,.001,.001]);coef=np.linalg.solve(A.T@(weight[:,None]*A)+penalty,A.T@(weight*y))
        models[held]=dict(held_out_physical_seed=held,train_seeds=sorted(set(t['seed'] for t in train)),center=center.tolist(),scale=scale.tolist(),coef_standardized=coef.tolist(),features=[FIELDS[j] for j in contacts_idx],fit_rows=len(y),ridge_penalty=.001)
    (out/'contact_correction_models.json').write_text(json.dumps(models,indent=2))
    metrics=[];events=[];index=[]
    colors=['#0072B2','#D55E00','#009E73','#777777'];alphas=[0,.25,.5,.75,1]
    for ci,c in enumerate(cases):
        name=c['name'];model=models[c['seed']];beta=np.asarray(model['coef_standardized']);d=beta[0]+((c['X']-model['center'])/model['scale'])@beta[1:]
        valid=c['valid']&c['contact_valid'];pred0=replay(c['phi'],c['rate'],c['forcing'],valid,c['a4'],.1);predc=replay(c['phi'],c['rate'],c['forcing']+d,valid,c['a4'],.1)
        p500=replay(c['phi'],c['rate'],c['forcing'],valid,c['a4'],.5);pc500=replay(c['phi'],c['rate'],c['forcing']+d,valid,c['a4'],.5)
        c['pred0']=pred0;c['predc']=predc;c['events']={}
        for lane,label in enumerate(c['labels']):
            mask=c['valid'][:,lane];raised=mask&((c['hi'][:,lane,16]>0)|(c['hi'][:,lane,17]>0));ids=np.flatnonzero(raised)
            groups=np.split(ids,np.flatnonzero(np.diff(ids)>10)+1) if len(ids) else []
            intervals=[];inband=(abs(c['phi'][:,lane])<=.18)&(abs(c['path'][:,lane])<=.3)&(abs(c['speed'][:,lane]-c['requested'])<=.2)
            for eid,g in enumerate(groups):
                lo=max(.8,g[0]*.005);end=c['time'][g[-1],lane];nexti=groups[eid+1][0] if eid+1<len(groups) else np.flatnonzero(mask)[-1]+1
                recovery=None;lastoff=np.any(~inband[g[0]:g[-1]+1]);search=g[-1]+1
                for j in range(search,nexti-39):
                    if np.all(mask[j:j+40]&inband[j:j+40]):recovery=float(c['time'][j,lane]);break
                intervals.append((lo,float(end),recovery))
                events.append(dict(case=name,label=label,event=eid,start_s=lo,end_s=float(end),recovery_entry_s=recovery,recovery_delay_s=None if recovery is None else recovery-end,left_band_during_event=bool(lastoff),interpretation='returned and held200ms' if lastoff else 'already inside or no observed departure'))
            c['events'][lane]=intervals
            row=next(x for x in source_rows if x['case']==name and x['label']==label)
            stat=dict(case=name,label=label,physical_seed=c['seed'],final_hold=row['final_hold'],cause=row['cause'],observed_s=row['observed_seconds'],first_raised_contact_s=intervals[0][0] if intervals else None,entry_speed_m_s=None if not len(ids) else float(c['speed'][ids[0],lane]),entry_roll_deg=None if not len(ids) else float(np.degrees(c['phi'][ids[0],lane])),peak_abs_roll_deg=float(np.max(abs(np.degrees(c['phi'][mask,lane])))),motor_request_clipped_fraction=float(np.mean((c['hi'][valid[:,lane],lane,15]>1)|(c['lo'][valid[:,lane],lane,15]<-1))))
            for title,pred in [('nominal100',pred0),('contact100',predc),('nominal500',p500),('contact500',pc500)]:
                ok=np.isfinite(pred[:,lane])&mask;stat[title+'_rmse_deg']=float(np.sqrt(np.mean(np.degrees(pred[ok,lane]-c['phi'][ok,lane])**2))) if ok.any() else None;stat[title+'_samples']=int(ok.sum())
            metrics.append(stat)
        np.savez_compressed(out/(name+'.npz'),time=c['time'],valid=c['valid'],phi=c['phi'],euler_roll_rate=c['rate'],speed=c['speed'],delta=c['delta'],delta_dot=c['dr'],pred_nominal100=pred0,pred_contact100=predc,pred_nominal500=p500,pred_contact500=pc500,contact_correction=d,contact_mean=c['mean'],contact_min=c['lo'],contact_max=c['hi'],contact_fields=np.array(FIELDS),command=c['command'],path_error=c['path'],xy=c['pose'][:,:,:2],labels=np.array(c['labels']))
        def decorate(ax,lane):
            for a,b,rec in c['events'][lane]:
                ax.axvspan(a,b,color='#bdbdbd',alpha=.22)
                if rec is not None:ax.axvline(rec,color='#009E73',lw=.6,ls=':')
            end=c['time'][np.flatnonzero(c['valid'][:,lane])[-1],lane]
            if end<9.999:ax.axvline(end,color='#CC3311',lw=.9,ls='--')
            ax.set_xlim(0,10)
        def series(lane,col):
            if col==0:return np.degrees(c['phi'][:,lane])
            if col==1:return c['speed'][:,lane]
            if col==2:return np.degrees(c['delta'][:,lane])
            if col==3:return c['dr'][:,lane]
            if col==4:return c['mean'][:,lane,4]+c['mean'][:,lane,5]
            return c['mean'][:,lane,10]+c['mean'][:,lane,11]
        titles=['Roll (deg)','Actual speed (m/s)','Steer angle (deg)','Steer rate (rad/s)','Front contact roll moment (Nm)','Rear contact roll moment (Nm)']
        fig,axs=plt.subplots(6,6,figsize=(24,16),sharex=True,layout='constrained')
        for row,lane in enumerate([0,3,6,9,12,15]):
            mask=c['valid'][:,lane];t=c['time'][:,lane];stat=next(x for x in metrics if x['case']==name and x['label']==c['labels'][lane])
            for col,ax in enumerate(axs[row]):
                ax.plot(t[mask],series(lane,col)[mask],color=colors[0],lw=1,label='Recorded')
                if col==0:
                    ax.plot(t,pred0[:,lane]*180/np.pi,color=colors[1],lw=.9,ls='--',label='Nominal100ms replay');ax.plot(t,predc[:,lane]*180/np.pi,color=colors[2],lw=.9,ls='-.',label='Contact-corrected100ms')
                    ax.text(.02,.98,f"RMSE nominal/contact: {stat['nominal100_rmse_deg']:.2f}/{stat['contact100_rmse_deg']:.2f} deg",transform=ax.transAxes,va='top',fontsize=8)
                if col in (4,5):
                    k=4 if col==4 else 10
                    # Envelope is a conservative sum of independently binned extrema.
                    lower=c['lo'][:,lane,k]+c['lo'][:,lane,k+1];upper=c['hi'][:,lane,k]+c['hi'][:,lane,k+1]
                    ax.fill_between(t,lower,upper,where=mask,color=colors[0],alpha=.12,label='5ms component-extrema envelope')
                if lane in (6,12):
                    dup=16 if lane==6 else 17;ok=c['valid'][:,dup];ax.plot(c['time'][ok,dup],series(dup,col)[ok],color='#CC79A7',lw=.8,ls=':',label='Repeated lane')
                decorate(ax,lane)
                if row==0:ax.set_title(titles[col])
                if col==0:ax.set_ylabel(c['labels'][lane]+'\n'+('hold' if stat['final_hold'] else stat['cause']))
                if row==5:ax.set_xlabel('Time (s)')
            axs[row,0].legend(fontsize=6,loc='lower right')
        fig.suptitle(name+' | H75 frozen | common-start | gray: raised-wheel contacts; green: band entry held200ms; red: termination\n100ms overlapping measured-input replays, initialized from measured state; correction fitted on OTHER physical seeds. Moment is about whole-vehicle COM.',fontsize=12)
        fig.savefig(out/(name+'_models.png'));plt.close(fig)
        fig,axs=plt.subplots(5,6,figsize=(24,13),sharex=True,layout='constrained')
        for row in range(5):
            for col,ax in enumerate(axs[row]):
                for mode,lane in enumerate([3*row,3*row+1,3*row+2,15]):
                    mask=c['valid'][:,lane];ax.plot(c['time'][mask,lane],series(lane,col)[mask],color=colors[mode],ls=['-','--','-.',':'][mode],lw=1,label=['Full','Speed only','Steer only','Zero'][mode])
                ax.set_xlim(0,10)
                if row==0:ax.set_title(titles[col])
                if col==0:ax.set_ylabel('alpha='+str(alphas[row]))
                if row==4:ax.set_xlabel('Time (s)')
            axs[row,0].legend(fontsize=7)
        fig.suptitle(name+' | Same-start channel ablation; all curves end at true valid endpoint | Wheel moments:5ms means about vehicle COM',fontsize=12)
        fig.savefig(out/(name+'_ablation.png'));plt.close(fig)
        fig,axs=plt.subplots(6,3,figsize=(15,15),sharex=True,layout='constrained')
        for row,lane in enumerate([0,3,6,9,12,15]):
            mask=c['valid'][:,lane];t=c['time'][:,lane];m=c['mean'][:,lane]
            axs[row,0].plot(t[mask],m[mask,0],label='Front normal N');axs[row,0].plot(t[mask],m[mask,6],label='Rear normal N',ls='--')
            axs[row,1].plot(t[mask],m[mask,12]+m[mask,13],label='Front contact about steer axis',color=colors[0]);axs[row,1].plot(t[mask],m[mask,14],label='Applied steering motor',color=colors[1],ls='--');axs[row,1].plot(t[mask],m[mask,15],label='Requested motor',color=colors[2],ls=':');axs[row,1].axhline(1,color='gray',lw=.5);axs[row,1].axhline(-1,color='gray',lw=.5)
            axs[row,2].plot(t[mask],c['controller_request'][mask,lane],label='ECBC before rate limit');axs[row,2].plot(t[mask],c['command'][mask,lane],label='Final steering-rate command',ls='--');axs[row,2].plot(t[mask],c['dr'][mask,lane],label='Actual steering rate',ls=':')
            for col,ax in enumerate(axs[row]):
                decorate(ax,lane);ax.legend(fontsize=6)
                if row==0:ax.set_title(['Normal contact force (N)','Steering-axis torques (Nm)','Steering-rate control (rad/s)'][col])
                if col==0:ax.set_ylabel(c['labels'][lane])
                if row==5:ax.set_xlabel('Time (s)')
        fig.suptitle(name+' | Contact and actuator timing; means over5ms (native0.2ms samples) | gray/green/red as model panel',fontsize=12)
        fig.savefig(out/(name+'_actuation.png'));plt.close(fig)
        shutil.copy2(run/name/'xy.png',out/(name+'_xy.png'))
        index.append(f"## {name}\n\n![XY]({name}_xy.png)\n\n[动态模型与五档响应]({name}_models.png) · [通道消融]({name}_ablation.png) · [接触力与执行器]({name}_actuation.png) · [图中数据]({name}.npz)\n")
        print('figures',ci+1,len(cases),name,flush=True)
    write_csv(out/'metrics.csv',metrics);write_csv(out/'events.csv',events)
    (out/'INDEX.md').write_text('# 全工况图索引\n\n12工况、216条；所有五alpha/三通道模式/Zero/重复均保留。XY先列。\n\n'+'\n'.join(index))
    summary=[]
    for terrain in ('dense10','single10'):
        for mode in ('full_0.5','full_1','speed_only_0.5','steer_only_0.5'):
            rows=[r for r in metrics if r['case'].startswith(terrain) and r['label']==mode]
            summary.append(dict(terrain=terrain,label=mode,n=len(rows),hold=sum(r['final_hold'] for r in rows),nominal100_mean_rmse_deg=float(np.mean([r['nominal100_rmse_deg'] for r in rows])),contact100_mean_rmse_deg=float(np.mean([r['contact100_rmse_deg'] for r in rows])),peak_roll_mean_deg=float(np.mean([r['peak_abs_roll_deg'] for r in rows]))))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'status.json').write_text(json.dumps(dict(stage='complete',cases=len(cases),lanes=len(metrics))))
    print(json.dumps(summary,indent=2),flush=True)


def positive_figures(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18,'figure.dpi':150})
    config=json.loads(Path('runs/isaac_signals_parity/dynamic.json').read_text());rows=json.loads((run/'analysis_summary.json').read_text())['rows']
    selections=[dict(case='single10_v2.6_seed101',positive='full_0.5',lanes=[6,12,7,8],title='10cm ridge | 2.6m/s | seed101',reason='First declared physical group at2.6m/s; full alpha.5 passes task C in all3 groups and final hold in6/6 contexts.'),dict(case='dense10_v2.3_seed101',positive='full_0.75',lanes=[9,10,11,15],title='10mm stones | 2.3m/s | seed101',reason='First declared physical group at2.3m/s; full alpha.75 passes A/B/C here; final hold6/6 overall, not ABC6/6.')]
    events=list(csv.DictReader((out/'events.csv').open()));colours=['#0072B2','#D55E00','#009E73','#777777'];styles=['-','--','-.',':'];data=[];detail=[]
    for s in selections:
        name=s['case'];d=dict(np.load(out/(name+'.npz')));c=json.loads((run/name/'contract.json').read_text());route=np.load(Path(c['bundle'])/'route.npz');goal=float(route['goal']);point=np.array([np.interp(goal,route['s'],route['xy'][:,j]) for j in (0,1)]);heading=np.interp(goal,route['s'],route['heading']);forward=np.array([np.cos(heading),np.sin(heading)])
        with np.load(run/name/'trajectory.npz') as z:
            progress=z['progress'];pose=z['pose'];measurement=z['measurement'];trace=z['lower_trace'].reshape(2000,18,11)
        d['goal_distance']=np.linalg.norm(d['xy']-point,axis=-1);d['route']=route['xy'];d['point']=point;d['events']=[e for e in events if e['case']==name and e['label']==s['positive']]
        d['nominal_demand']=np.zeros_like(d['phi']);a1,a2,a4=nominal_coeff(d['speed'],config);lam=np.sqrt(a4);d['z']=d['euler_roll_rate']+lam*d['phi'];d['unstable']=lam*abs(d['z']);d['restoring']=-np.sign(d['z'])*(a1*d['delta_dot']+a2*d['delta']);d['nominal_demand']=-(lam*d['z']+a2*d['delta'])/a1
        goals={}
        for lane in s['lanes']:
            valid=d['valid'][:,lane];signed=(d['xy'][:,lane]-point)@forward;cross=np.flatnonzero(valid&(signed>=0)&(progress[:,lane]>=goal));goals[lane]=None if not len(cross) else float(d['time'][cross[0],lane]);row=next(x for x in rows if x['case']==name and x['lane']==lane)
            es=[e for e in events if e['case']==name and e['label']==row['label']];j=np.flatnonzero(d['time'][:,lane]>=float(es[0]['start_s']))[0] if es else None
            detail.append(dict(case=name,label=row['label'],positive=row['label']==s['positive'],final_hold=row['final_hold'],task_A=row['tasks']['A'],task_B=row['tasks']['B'],task_C=row['tasks']['C'],cause=row['cause'],end_s=row['observed_seconds'],goal_crossing_s=goals[lane],path_peak_m=row['path_peak_m'],speed_rmse_m_s=row['speed_rmse_m_s'],peak_roll_deg=float(np.degrees(row['peak_roll_rad'])),first_contact_s=None if j is None else float(es[0]['start_s']),entry_speed_m_s=None if j is None else float(d['speed'][j,lane]),entry_phi_deg=None if j is None else float(np.degrees(d['phi'][j,lane])),minimum_goal_distance_m=float(d['goal_distance'][valid,lane].min())))
        d['goals']=goals;data.append(d)
    write_csv(out/'positive_cases.csv',detail);(out/'positive_selection.json').write_text(json.dumps(selections,indent=2))
    def decorate(ax,s,d,col,show_hold=True):
        for e in d['events']:
            ax.axvspan(float(e['start_s']),float(e['end_s']),color='#bdbdbd',alpha=.22)
            if e['recovery_entry_s']:ax.axvline(float(e['recovery_entry_s']),color='#009E73',ls=':',lw=.8)
        for k,lane in enumerate(s['lanes']):
            valid=d['valid'][:,lane];end=float(d['time'][valid,lane][-1])
            if end<9.999:ax.axvline(end,color=colours[k],ls='--',lw=.7)
        if show_hold:ax.axvspan(9,10,color='#56B4E9',alpha=.10)
        ax.set_xlim(0,10)
    def lines(ax,s,d,key,scale=1):
        for k,lane in enumerate(s['lanes']):
            valid=d['valid'][:,lane];ax.plot(d['time'][valid,lane],d[key][valid,lane]*scale,color=colours[k],ls=styles[k],lw=1.2 if k==0 else .9,label=str(d['labels'][lane]))
    fig,axes=plt.subplots(1,2,figsize=(10,9),layout='constrained')
    for ax,s,d in zip(axes,selections,data):
        ax.plot(*d['route'].T,color='black',ls='--',label='Original reference')
        for k,lane in enumerate(s['lanes']):
            valid=d['valid'][:,lane];xy=d['xy'][valid,lane];ax.plot(*xy.T,color=colours[k],ls=styles[k],label=str(d['labels'][lane]));ax.scatter(*xy[-1],marker='o' if valid.sum()==2000 else 'x',color=colours[k],s=25)
        ax.scatter(*d['point'],marker='*',s=85,color='black',label='Goal-section anchor');allxy=np.concatenate([d['xy'][d['valid'][:,l],l] for l in s['lanes']]);ax.set_xlim(allxy[:,0].min()-.6,allxy[:,0].max()+.6);ax.set_ylim(allxy[:,1].min()-.6,allxy[:,1].max()+.6);ax.set_aspect('equal',adjustable='box');ax.set(xlabel='World x relative to environment (m)',ylabel='World y relative to environment (m)',title=s['title']);ax.legend(fontsize=8)
    fig.suptitle('Positive examples and matched channel controls | fixed original reference; no trajectory relocation')
    fig.savefig(out/'POSITIVE_XY.png');plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(15,17),sharex=True,layout='constrained')
    keys=['phi','euler_roll_rate','speed','delta','delta_dot'];units=['Roll (deg)','Euler roll rate (rad/s)','Actual forward speed (m/s)','Actual steering (deg)','Actual steering rate (rad/s)']
    for col,(s,d) in enumerate(zip(selections,data)):
        for row,(key,unit) in enumerate(zip(keys,units)):
            ax=axes[row,col];lines(ax,s,d,key,180/np.pi if key in ('phi','delta') else 1);decorate(ax,s,d,col);ax.set_ylabel(unit)
            if key=='phi':ax.axhline(np.degrees(.7),color='black',lw=.6,ls=':');ax.axhline(-np.degrees(.7),color='black',lw=.6,ls=':')
            if key=='delta':
                for limit in [-.8,.8]:ax.axhline(np.degrees(limit),color='black',lw=.6,ls=':')
            if key=='delta_dot':
                for limit in [-3,3]:ax.axhline(limit,color='black',lw=.6,ls=':',label='Command bound, not measured-rate hard bound' if limit==3 else None)
            if row==0:ax.set_title(s['title']);ax.legend(fontsize=8,loc='best')
            if row==4:ax.set_xlabel('Time (s)')
    fig.suptitle('Recovery states | gray: raised contacts of POSITIVE lane; green: joint-band entry held200ms; blue: frozen final hold window\nComparators have their own contact timing in events.csv; dashed colored verticals are true termination times.',fontsize=12)
    fig.savefig(out/'POSITIVE_STATES.png');plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(15,17),sharex=True,layout='constrained')
    for col,(s,d) in enumerate(zip(selections,data)):
        lane=s['lanes'][0];valid=d['valid'][:,lane];t=d['time'][:,lane];m=d['contact_mean'][:,lane];lo=d['contact_min'][:,lane];hi=d['contact_max'][:,lane]
        for k,cidx in enumerate([0,6]):
            axes[0,col].plot(t[valid],m[valid,cidx],label=['Front','Rear'][k],color=colours[k],ls=styles[k]);axes[0,col].fill_between(t,lo[:,cidx],hi[:,cidx],color=colours[k],alpha=.12)
        for k,cidx in enumerate([4,10]):axes[1,col].plot(t[valid],(m[:,cidx]+m[:,cidx+1])[valid],label=['Front','Rear'][k],color=colours[k],ls=styles[k])
        axes[2,col].plot(t[valid],m[valid,15],color='#777777',ls=':',label='Requested steering motor');axes[2,col].plot(t[valid],m[valid,14],color=colours[0],label='Applied steering motor')
        for limit in [-1,1]:axes[2,col].axhline(limit,color='black',lw=.7,ls='--')
        lines(axes[3,col],s,d,'path_error');lines(axes[4,col],s,d,'goal_distance')
        threshold=.4 if col==0 else .3
        for limit in [-threshold,threshold]:axes[3,col].axhline(limit,color='black',lw=.6,ls=':')
        for k,l in enumerate(s['lanes']):
            goal=d['goals'][l]
            if goal is not None:
                j=np.argmin(abs(d['time'][:,l]-goal));axes[4,col].scatter(goal,d['goal_distance'][j,l],color=colours[k],marker='*',s=65)
        for row,unit in enumerate(['Normal wheel force (N)','Wheel roll moment about COM (Nm)','Steering motor torque (Nm)','Original path error (m)','Distance to goal-section anchor (m)']):
            decorate(axes[row,col],s,d,col);axes[row,col].set_ylabel(unit);axes[row,col].legend(fontsize=7)
            if row==0:axes[row,col].set_title(s['title']+' | contacts/motor: positive lane')
            if row==4:axes[row,col].set_xlabel('Time (s)')
    fig.suptitle('Contact -> actuator -> path and goal | force bands retain native extrema within5ms; torque limits +/-1Nm\nStars: crossing the original goal section. The task continues past this section; it is not a parking or1m-circle task.',fontsize=12)
    fig.savefig(out/'POSITIVE_CONTACT_TASK.png');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(15,11),sharex=True,layout='constrained')
    for col,(s,d) in enumerate(zip(selections,data)):
        lane=s['lanes'][0];valid=d['valid'][:,lane];t=d['time'][valid,lane]
        lines(axes[0,col],s,d,'z');axes[0,col].set_ylabel('z = Euler roll rate + sqrt(a4)*roll (rad/s)');axes[0,col].legend(fontsize=7)
        axes[1,col].plot(t,d['unstable'][valid,lane],label='Unstable growth sqrt(a4)*abs(z)',color=colours[1]);axes[1,col].plot(t,d['restoring'][valid,lane],label='Actual nominal steering correction',color=colours[0]);axes[1,col].set_ylabel('Nominal contribution (rad/s^2)');axes[1,col].legend(fontsize=8)
        axes[2,col].plot(t,d['nominal_demand'][valid,lane],label='Required rate for dot(z)=0 with d=0',color=colours[1],ls='--');axes[2,col].plot(t,d['delta_dot'][valid,lane],label='Actual steering rate',color=colours[0]);axes[2,col].plot(t,d['command'][valid,lane],label='Final rate command before servo clipping',color='#777777',ls=':')
        for lim in [-3,3]:axes[2,col].axhline(lim,color='black',ls=':',lw=.7)
        axes[2,col].set_ylabel('Steering rate (rad/s)');axes[2,col].legend(fontsize=8);axes[2,col].set_xlabel('Time (s)')
        for row in range(3):decorate(axes[row,col],s,d,col)
        axes[0,col].set_title(s['title'])
    fig.suptitle('Nominal recovery-demand diagnostic | positive lanes in rows2/3; measured v and delta\nExternal/contact residual is excluded from this demand formula. +/-3 is the servo command range, NOT a proved recovery boundary.',fontsize=12)
    fig.savefig(out/'POSITIVE_RECOVERY_DEMAND.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
    for ax,s,d in zip(axes,selections,data):
        lane=s['lanes'][0];valid=d['valid'][:,lane];t=d['time'][:,lane]
        ax.plot(t[valid],np.degrees(d['phi'][valid,lane]),label='Recorded',color=colours[0]);ax.plot(t,np.degrees(d['pred_nominal100'][:,lane]),label='Nominal100ms measured-input replay',color=colours[1],ls='--');ax.plot(t,np.degrees(d['pred_contact100'][:,lane]),label='Contact correction: other-seed fit',color=colours[2],ls='-.');decorate(ax,s,d,0);ax.set(title=s['title']+' | '+s['positive'],ylabel='Roll (deg)');ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)');fig.suptitle('Local dynamic-model check: overlapping100ms integrations, initialized from measured state\nNo alpha-only forward prediction; no same-case residual fitting. Full500ms errors are reported separately.',fontsize=12)
    fig.savefig(out/'POSITIVE_MODEL_CHECK.png');plt.close(fig)
    # A short-horizon inertia-only comparator makes the scale of the replay test explicit.
    metric_rows=list(csv.DictReader((out/'metrics.csv').open()))
    for path in sorted(out.glob('*_seed*.npz')):
        d=np.load(path);phi=d['phi'];rate=d['euler_roll_rate'];baseline=np.full(phi.shape,np.nan);baseline[20:]=phi[:-20]+.1*rate[:-20]
        for lane,label in enumerate(d['labels']):
            mask=np.isfinite(d['pred_nominal100'][:,lane]);r=next(x for x in metric_rows if x['case']==path.stem and x['label']==label);r['constant_roll_rate100_rmse_deg']=float(np.sqrt(np.mean(np.degrees(baseline[mask,lane]-phi[mask,lane])**2)))
    write_csv(out/'metrics.csv',metric_rows)
    np.savez_compressed(out/'positive_derived.npz',**{s['case']+'__'+key:d[key] for s,d in zip(selections,data) for key in ['z','unstable','restoring','nominal_demand','goal_distance']})
    print('positive figures complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--stage',choices=['extract','figures','positive'],default='extract');p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=a.output or a.run/'roll_contact_analysis';out.mkdir(exist_ok=True)
    if a.stage=='extract':extract(a.run,out,a.workers)
    elif a.stage=='figures':figures(a.run,out)
    else:positive_figures(a.run,out)
