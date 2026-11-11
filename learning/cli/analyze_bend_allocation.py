"""Test the plan's local speed/curvature allocation on archived H03/H05 motion."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def project_original(xy,path):
    # Dense original polyline projection; eight nearby segment midpoints.
    a=path['xy'][:-1];d=np.diff(path['xy'],axis=0);span=np.diff(path['s']);tree=cKDTree(a+d/2)
    _,ids=tree.query(xy,k=8);aa=a[ids];dd=d[ids];u=np.clip(np.sum((xy[:,None]-aa)*dd,-1)/np.sum(dd*dd,-1),0,1);points=aa+u[:,:,None]*dd;best=np.argmin(np.sum((xy[:,None]-points)**2,-1),axis=1);ix=np.arange(len(xy));seg=ids[ix,best];s=path['s'][seg]+u[ix,best]*span[seg];heading=np.interp(s,path['s'],path['heading']);err=xy-points[ix,best];cross=-np.sin(heading)*err[:,0]+np.cos(heading)*err[:,1]
    return s,cross

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run_specs=[('H03',a.archive/name) for name in ['task_preference_middle_h75_recheck_v1','task_preference_middle_h75_extension_v1']]+[('H05',a.archive/'task_matrix_v2_20261104'/f'{speed}_{method}') for speed in ['low','mid','high'] for method in ['h75','pp_corrected']]
    records=[];curves={};series={};colours={0:'#0072B2',.25:'#56B4E9',.5:'#009E73',.75:'#E69F00',1:'#D55E00',-1:'#777777'}
    for dataset,run in run_specs:
        plan=json.loads((run/'plan.json').read_text());index=json.loads((run/'evaluation/case_index.json').read_text());cfg=plan['config'];cc=plan['initial_source']['controller'];rwd=cfg['path_reward'];sv=rwd['speed_scale_m_s'];se=rwd['cross_track_scale_m']
        for case in index:
            path=dict(np.load(run/'evaluation'/f'{case["path_key"]}_path.npz'));raw=np.load(case['raw_file']);n=int(raw['active_tick'].sum());xy=raw['actual_xy'][:n];time=raw['time'][:n];v=raw['actual_forward_speed'][:n];v0=case['speed'];s,cross=project_original(xy,path);kref=np.interp(s,path['s'],path['curvature']);alpha=-1. if 'pp_corrected' in run.name else float(case['alpha']);key=f'{dataset}_{case["route"]}_bank{case["bank_index"]}_{case["event"]}'
            sharp='sharp_90' in str(next((x for x in plan['protocol'].get('route_contexts',[]) if x['route']==case['route']),{}));window=41;dx=savgol_filter(xy,window,3,deriv=1,delta=.005,axis=0);dd=savgol_filter(xy,window,3,deriv=2,delta=.005,axis=0);den=np.linalg.norm(dx,axis=1)**3;ka=np.divide(dx[:,0]*dd[:,1]-dx[:,1]*dd[:,0],den,out=np.full(n,np.nan),where=den>1e-5)
            # Constant-curvature core, original path coordinates; no outcome selection.
            peak=np.max(abs(path['curvature']));core=(s>=3)&(s<=float(path['turn_end']))&(abs(kref)>=.9*peak)&(abs(kref)>.01)&np.isfinite(ka);core[:20]=False;core[-20:]=False
            if sharp:core[:]=False
            K=abs(kref);sign=np.sign(kref);slow=v0-v;outward=-sign*cross;reduce_k=K-sign*ka
            speed_term=2*v0*K*slow;curve_term=v0*v0*reduce_k;linear=speed_term+curve_term;exact=v0*v0*K-v*v*sign*ka
            # Quadratic two-error surrogate with osculating-circle delta_k=K^2*e_out.
            # q=delta/phi=-a4/a2; derive directly from nominal equilibrium.
            q=cc['wheelbase']*cc['gravity']*cc['cg_height']/((v0*v0*cc['cg_height']-cc['cg_forward']*cc['trail']*cc['gravity'])*np.cos(cc['caster']))
            chi=np.clip((abs(np.arctan(cc['wheelbase']*kref/np.cos(cc['caster']))/q)-.18)/.12,0,1)
            aa=max(alpha,0);wv=10-9*chi*(1-aa);wp=10*(1-(1-rwd['path_secondary_ratio'])*aa);gs=2*v0*K*sv;ge=v0*v0*K*K*se;r=wp/wv;rho=np.divide(ge*ge,gs*gs,out=np.zeros(n),where=gs!=0);pred=r/(r+rho)
            eligible=core&(slow>=0)&(reduce_k>=0)&(linear>.05)
            frac=np.divide(speed_term,linear,out=np.full(n,np.nan),where=linear>.05)
            quad=eligible&(slow<=sv)&(outward>=0)&(outward<=se)
            bothlinear=eligible&(slow>sv)&(outward>se)
            huber_pred=np.full(n,np.nan)
            ids=np.flatnonzero(eligible)
            lo=np.zeros(len(ids));hi=linear[ids]/gs[ids]
            for _ in range(45):
                x=(lo+hi)/2;y=(linear[ids]-gs[ids]*x)/ge[ids]
                derivative=wv[ids]*np.minimum(x,1)-wp*(gs[ids]/ge[ids])*np.minimum(y,1)
                hi=np.where(derivative>0,x,hi);lo=np.where(derivative<=0,x,lo)
            huber_pred[ids]=gs[ids]*(lo+hi)/2/linear[ids]
            sat=core&(slow>sv)
            def mean(x,mask):return float(np.mean(x[mask])) if mask.any() else None
            row=dict(dataset=dataset,route=case['route'],bank=case['bank_index'],event=case['event'],alpha=None if alpha<0 else alpha,method='PP' if alpha<0 else 'H75',sharp_polyline=sharp,source=case['raw_file'],core_samples=int(core.sum()),allocation_samples=int(eligible.sum()),quadratic_samples=int(quad.sum()),both_linear_samples=int(bothlinear.sum()),chi_mean=mean(chi,core),speed_mean=mean(v,core),outward_mean_m=mean(outward,core),curvature_reduction_mean=mean(reduce_k,core),speed_allocation_fraction=mean(frac,eligible),quadratic_prediction_mean=mean(pred,eligible),quadratic_only_MAE=mean(abs(frac-pred),quad),huber_allocation_MAE=mean(abs(frac-huber_pred),eligible),huber_prediction_mean=mean(huber_pred,eligible),all_eligible_MAE=mean(abs(frac-pred),eligible),first_order_MAE_m_s2=mean(abs(exact-linear),core),exact_demand_change_abs_mean=mean(abs(exact),core),curvature_geometry_MAE_1_m=mean(abs(reduce_k-K*K*outward),core),raw_speed_huber_linear_fraction=mean((slow>sv).astype(float),core),raw_path_huber_linear_fraction=mean((abs(cross)>se).astype(float),core),planned_chi1_prediction_mean=mean(1/(1+rho*(1+9*aa)/(10-9*aa)),eligible))
            records.append(row);curves.setdefault(key,[]).append((alpha,xy,path));series[key,alpha]=dict(time=time,s=s,v=v,xy=xy,kref=kref,kactual=ka,outward=outward,core=core,eligible=eligible,quadratic=quad,speed_term=speed_term,curvature_term=curve_term,exact=exact,linear=linear,actual_fraction=frac,predicted_fraction=pred,huber_fraction=huber_pred,chi=chi)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    # Original XY overview for every paired context, grouped into manageable pages.
    links=[];keys=sorted(curves)
    for page,first in enumerate(range(0,len(keys),12)):
        chosen=keys[first:first+12];fig,axs=plt.subplots(3,4,figsize=(16,11),layout='constrained')
        for ax,key in zip(axs.flat,chosen):
            for alpha,xy,path in sorted(curves[key],key=lambda x:x[0]):ax.plot(*xy.T,color=colours[alpha],lw=1,label='PP' if alpha<0 else f'a={alpha:g}')
            valid=path['s']<=path['goal'];ax.plot(*path['xy'][valid].T,'k:',lw=1,label='Original');ax.set_aspect('equal');ax.set_title(key.replace('tasktest_','').replace('H05_',''),fontsize=8);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        for ax in axs.flat[len(chosen):]:ax.set_visible(False)
        axs.flat[0].legend(fontsize=6);fig.suptitle('Actual / original XY | H03 five alpha, H05 three alpha + PP');name=f'xy_{page+1:02d}.png';fig.savefig(a.output/name,dpi=140);plt.close(fig);links.append(name)
    nominal=[x for x in records if x['method']=='H75' and x['event']=='zero' and not x['sharp_polyline']];groups=sorted(set((x['dataset'],x['route']) for x in nominal));fig,axs=plt.subplots((len(groups)+3)//4,4,figsize=(17,3.5*((len(groups)+3)//4)),layout='constrained')
    for ax,group in zip(axs.flat,groups):
        for bank in sorted(set(x['bank'] for x in nominal)):
            rr=sorted([x for x in nominal if (x['dataset'],x['route'])==group and x['bank']==bank],key=lambda x:x['alpha'])
            if not rr:continue
            x=[q['alpha'] for q in rr];ax.plot(x,[q['speed_allocation_fraction'] for q in rr],'o-',color='#0072B2',alpha=.7,label='Measured' if bank==rr[0]['bank'] else None);ax.plot(x,[q['quadratic_prediction_mean'] for q in rr],'s--',color='#D55E00',alpha=.7,label='Local quadratic surrogate');ax.plot(x,[q['huber_prediction_mean'] for q in rr],'^:',color='#009E73',alpha=.7,label='Conditional Huber surrogate')
        ax.set_title(group[0]+' '+group[1].replace('tasktest_',''),fontsize=8);ax.set(xlabel='alpha',ylabel='Speed share of linear demand reduction',ylim=(-.05,1.05))
    for ax in axs.flat[len(groups):]:ax.set_visible(False)
    handles,labels=axs.flat[0].get_legend_handles_labels();unique=dict(zip(labels,handles));axs.flat[0].legend(unique.values(),unique.keys(),fontsize=7);fig.suptitle('Bend allocation: archived nominal cases | each line is one prepared state\nMeasured share is defined only where both contributions reduce demand; no fitting');fig.savefig(a.output/'allocation.png',dpi=160);plt.close(fig)
    sums={}
    for dataset in ['H03','H05']:
        for event in sorted(set(x['event'] for x in records if x['dataset']==dataset)):
            rr=[x for x in records if x['dataset']==dataset and x['event']==event and x['method']=='H75' and not x['sharp_polyline']]
            def pooled(k):return float(np.mean([x[k] for x in rr if x[k] is not None])) if any(x[k] is not None for x in rr) else None
            sums[dataset+'/'+event]=dict(episodes=len(rr),core_samples=sum(x['core_samples'] for x in rr),eligible_samples=sum(x['allocation_samples'] for x in rr),quadratic_samples=sum(x['quadratic_samples'] for x in rr),mean_episode_allocation_MAE=pooled('all_eligible_MAE'),mean_episode_quadratic_only_MAE=pooled('quadratic_only_MAE'),mean_episode_huber_MAE=pooled('huber_allocation_MAE'),mean_episode_first_order_MAE=pooled('first_order_MAE_m_s2'),mean_episode_curvature_geometry_MAE=pooled('curvature_geometry_MAE_1_m'))
    trend={}
    for dataset in ['H03','H05']:
        paired={}
        for row in nominal:
            if row['dataset']==dataset:paired.setdefault((row['route'],row['bank']),[]).append(row)
        edges=[]
        for key,rr in paired.items():
            rr.sort(key=lambda x:x['alpha'])
            for left,right in zip(rr,rr[1:]):
                edges.append(dict(route=key[0],bank=key[1],alpha_pair=[left['alpha'],right['alpha']],speed_increases=right['speed_mean']>=left['speed_mean'],outward_increases=right['outward_mean_m']>=left['outward_mean_m'],speed_share_decreases=right['speed_allocation_fraction']<=left['speed_allocation_fraction']))
        trend[dataset]=dict(edges=len(edges),speed_increases=sum(x['speed_increases'] for x in edges),outward_increases=sum(x['outward_increases'] for x in edges),speed_share_decreases=sum(x['speed_share_decreases'] for x in edges),details=edges)
    (a.output/'trends.json').write_text(json.dumps(trend,indent=2))
    (a.output/'results.json').write_text(json.dumps(dict(rows=records,summary=sums,xy=links,filter='Savitzky-Golay XY derivatives41samples/205ms/order3; first/last20 excluded; core >=90% peak reference curvature, original bend only; sharp polyline excluded from steady-curvature formula but retained in XY',projection='nearest original dense polyline among8closest segment midpoints; original route, no execution-route substitution',new_simulations=0),indent=2))
    # Store only derived arrays needed to reproduce these figures, original sources remain immutable.
    np.savez_compressed(a.output/'derived.npz',**{f'{key}__a{alpha:g}__{field}':value for (key,alpha),fields in series.items() for field,value in fields.items() if field not in ['xy','time']})
    print(json.dumps(sums,indent=2));print('episodes',len(records),'contexts',len(curves))
if __name__=='__main__':main()
