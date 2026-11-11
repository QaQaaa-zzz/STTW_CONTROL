"""Close recorded contact angular impulse against multibody angular momentum."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sttw_control.contact_momentum import roll_momentum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_impulse_map(rows,output):
    selected=[r for r in rows if r['window']=='rear_250ms' and r['case'].startswith('single') and str(r['complete_window'])=='True']
    heights=sorted(set(int(r['case'].split('_')[0][6:]) for r in selected))
    speeds=sorted(set(float(r['case'].split('_')[1][1:]) for r in selected))
    if len(heights)<2:return
    labels=['full_0','full_0.25','full_0.5','full_0.75','full_1','Zero'];maps=[]
    for label in labels:
        matrix=np.full((len(heights),len(speeds)),np.nan)
        for i,h in enumerate(heights):
            for j,v in enumerate(speeds):
                vals=[float(r['front_impulse_Nms'])+float(r['rear_impulse_Nms']) for r in selected if r['label']==label and r['case'].startswith(f'single{h:02d}_v{v}_')]
                if vals:matrix[i,j]=np.mean(vals)
        maps.append(matrix)
    bound=max(np.nanmax(abs(x)) for x in maps);fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,label,matrix in zip(axs.flat,labels,maps):
        im=ax.imshow(matrix,origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
        ax.set_xticks(range(len(speeds)),speeds);ax.set_yticks(range(len(heights)),heights);ax.set_xlabel('Requested speed (m/s)');ax.set_ylabel('Ridge height (cm)');ax.set_title(label.replace('full_','alpha='))
        for i in range(len(heights)):
            for j in range(len(speeds)):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if abs(matrix[i,j])>.65*bound else 'black')
    fig.colorbar(im,ax=axs,label='Signed contact angular impulse (N m s)',shrink=.8)
    fig.suptitle('Measured wheel-contact demand across height and speed\nFirst rear raised contact + 250 ms; front + rear moments; mean of three physical groups; no fitted scaling law')
    fig.savefig(output/'height_speed_impulse.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--contacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    bins=np.load(a.contacts);contact=bins['mean'];counts=bins['valid_physics_steps'];fields=list(bins['fields']);rows=[];series={};property_checks=[]
    for cp in sorted(a.run.glob('*/contract.json')):
        folder=cp.parent
        if not (folder/'trajectory.npz').exists():continue
        c=json.loads(cp.read_text());start=c['shared_physics_contacts']['lane_start'];n=c['shared_physics_contacts']['lane_count'];dt=c['lower_dt']
        raw=np.load(folder/'trajectory.npz');body=raw['body_state'].astype(float);trace=raw['lower_trace'].reshape(-1,n,11);t=(np.arange(len(body))+1)*dt
        masses=np.asarray(c['randomization_readback']['masses']);nominal=np.asarray(c['lab_physics']['masses']);coms=np.asarray(c['randomization_readback']['coms'])[...,:3]
        ratios=masses/nominal;inertias=np.asarray(c['lab_physics']['runtime_inertial']['inertias']).reshape(-1,3,3)[None]*ratios[...,None,None]
        # Existing moment extraction uses nominal masses; uniform mass scaling
        # leaves total COM unchanged. Record this property for every case.
        nominal_coms=np.asarray(c['lab_physics']['runtime_inertial']['coms'])[...,:3]
        property_checks.append(dict(case=folder.name,max_mass_ratio_spread=float(np.ptp(ratios,axis=1).max()),max_com_offset_change_m=float(abs(coms-nominal_coms).max())))
        mom=roll_momentum(body,masses,coms,inertias,c['bodies'].index('chasis'));valid=trace[:,:,6].astype(bool)&(counts[:,start:start+n]>0)
        con=contact[:,start:start+n];front=con[:,:,fields.index('front_normal_roll_Nm')]+con[:,:,fields.index('front_friction_roll_Nm')];rear=con[:,:,fields.index('rear_normal_roll_Nm')]+con[:,:,fields.index('rear_friction_roll_Nm')]
        for lane,label in enumerate(c['settings']['case_labels']):
            ids=np.flatnonzero(valid[:,lane]);events={}
            for wheel in ['front','rear']:
                hit=ids[con[ids,lane,fields.index(wheel+'_raised')]>0]
                if len(hit):events[wheel]=int(hit[0])
            windows={'common_1.7_1.8':(1.7,1.8),'common_1.9_1.95':(1.9,1.95),'common_2.3_2.4':(2.3,2.4)}
            for wheel,i in events.items():
                windows[wheel+'_100ms']=(t[i],t[i]+.1)
                windows[wheel+'_250ms']=(t[i],t[i]+.25)
            # End-of-record window only when the physics recording covers it.
            last=np.flatnonzero(trace[:,lane,6].astype(bool))[-1]
            if valid[last,lane]:windows['terminal100ms']=(t[last]-.1,t[last])
            for name,(lo,hi) in windows.items():
                take=ids[(t[ids]>=lo-1e-8)&(t[ids]<=hi+1e-8)]
                if len(take)<2:continue
                i,j=take[0],take[-1]
                if len(take)!=j-i+1:continue
                f=float(front[i+1:j+1,lane].sum()*dt);r=float(rear[i+1:j+1,lane].sum()*dt)
                transport=float(np.trapezoid(mom['axis_transport_rate'][i:j+1,lane],dx=dt));actual=float(mom['roll_momentum'][j,lane]-mom['roll_momentum'][i,lane]);error=actual-f-r-transport
                rows.append(dict(case=folder.name,label=label,window=name,start_s=float(t[i]),end_s=float(t[j]),duration_s=float(t[j]-t[i]),complete_window=bool(abs((t[j]-t[i])-(hi-lo))<1e-7),delta_roll_momentum_Nms=actual,front_impulse_Nms=f,rear_impulse_Nms=r,axis_transport_Nms=transport,closure_residual_Nms=error))
            if folder.name.startswith('single10_v2.6') and label in ['full_0','full_0.5','full_1']:
                take=ids[(t[ids]>=1.1-1e-8)&(t[ids]<=2.6+1e-8)];i=take[0];j=take[-1]
                integral=lambda x:np.r_[0,np.cumsum(x[i+1:j+1,lane])*dt]
                transport=np.r_[0,np.cumsum(.5*(mom['axis_transport_rate'][i:j,lane]+mom['axis_transport_rate'][i+1:j+1,lane]))*dt]
                series[folder.name+'__'+label]=dict(t=t[i:j+1],actual=mom['roll_momentum'][i:j+1,lane]-mom['roll_momentum'][i,lane],front=integral(front),rear=integral(rear),transport=transport)
        print(folder.name,flush=True)
    plot_impulse_map(rows,a.output)
    with (a.output/'momentum_windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(3,3,figsize=(15,11),sharex=True,sharey=True,layout='constrained')
    for row,seed in enumerate([101,102,103]):
        for col,label in enumerate(['full_0','full_0.5','full_1']):
            s=series[f'single10_v2.6_seed{seed}__{label}'];ax=axs[row,col];total=s['front']+s['rear']+s['transport']
            for key,color,style,name in [('actual','#222222','-','Measured change'),('front','#0072B2','--','Front contact'),('rear','#D55E00','-.','Rear contact'),('transport','#777777',':','Axis transport')]:ax.plot(s['t'],s[key],color=color,ls=style,lw=1.6,label=name)
            ax.plot(s['t'],total,color='#009E73',lw=1,label='Contact + transport');ax.set_title(f'Group {seed} | '+label.replace('full_','alpha='));ax.grid(alpha=.18);ax.set_xlim(1.1,2.6)
            if col==0:ax.set_ylabel('Angular momentum change (N m s)')
            if row==2:ax.set_xlabel('Time (s)')
    axs[0,0].legend(fontsize=8);fig.suptitle('10 cm ridge, 2.6 m/s | whole-vehicle angular momentum budget\nMass/inertia from vehicle parameters and fixed randomization; no fitted coefficients',fontsize=13)
    fig.savefig(a.output/'momentum_budget.png',dpi=150);plt.close(fig)
    np.savez_compressed(a.output/'plot_data.npz',**{name+'__'+k:v for name,s in series.items() for k,v in s.items()})
    full=[r for r in rows if r['complete_window']];error=np.array([abs(r['closure_residual_Nms']) for r in full]);scale=np.array([abs(r['front_impulse_Nms'])+abs(r['rear_impulse_Nms'])+abs(r['axis_transport_Nms']) for r in full])
    summary=dict(run=str(a.run),contacts=str(a.contacts),windows=len(rows),complete_windows=len(full),closure_MAE_Nms=float(error.mean()),closure_median_Nms=float(np.median(error)),closure_p95_Nms=float(np.quantile(error,.95)),sum_absolute_residual_over_sum_absolute_terms=float(error.sum()/scale.sum()),property_checks=property_checks,formula='Delta(e dot L_COM) = integral e dot (tau_front+tau_rear) dt + integral (omega_chassis cross e) dot L_COM dt',inertia_rule='runtime nominal tensor in link coordinates times actual/nominal body mass; Isaac Lab randomize_rigid_body_mass(recompute_inertia=True)',sampling='Body states at 5ms endpoints; original 5kHz contact mean right bins. Axis transport trapezoid; no fitting.',new_simulations=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='property_checks'})

if __name__=='__main__':main()
