"""Offline audit/publication of frozen unified-alpha runs; no simulator or PPO."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHAS=[0,.25,.5,.75,1]
TOKENS=['0','0p25','0p5','0p75','1']
CASES=['straight_2.0','straight_2.6','left90_R2_2.0','left90_R2_2.6','right90_R2_2.0','right90_R2_2.6']

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def bins(d,cfg):
    start=cfg['path']['straight_prefix_m'];end=start+cfg['path']['curvature_transition_length_m']+abs(cfg['path']['turn_angle_rad'])*cfg['path']['turn_radius_m']
    edges=np.r_[np.arange(start,end,.5),end];s=d['path_progress'];v=d['actual_forward_speed'].astype(float)-d['v_user'];y=d['path_cross_track'].astype(float)
    masks=[(s>=lo)&(s<hi if i<len(edges)-2 else s<=hi) for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:]))]
    assert all(m.any() for m in masks),'missing original turn bin'
    return np.sqrt(np.mean([np.mean(v[m]**2) for m in masks])),np.sqrt(np.mean([np.mean(y[m]**2) for m in masks]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=Path('/home/qy/STTW_CONTROL'));ap.add_argument('--output',type=Path,default=Path(__file__).parent);args=ap.parse_args();ws=args.workspace;out=args.output;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ws/'runs/worktrees/path-strong-preference/learning/src'))
    from sttw_control.path_command_reporting import reward_series
    roots={'warm':ws/'runs/worktrees/path-strong-preference/runs/path_strong_preference_200_reboot_resume_20261014/training','scratch':ws/'runs/worktrees/path-unified-scratch/runs/path_unified_scratch_200_20261014/resume_20261014_01'}
    initial={'warm':ws/'runs/worktrees/path-strong-preference/runs/path_strong_preference_200_20261014/training','scratch':ws/'runs/worktrees/path-unified-scratch/runs/path_unified_scratch_200_20261014/training'}
    manifests={k:load(r/'manifest.json') for k,r in roots.items()};assert manifests['warm']['source']==manifests['scratch']['source'];assert manifests['warm']['config']['path_reward']==manifests['scratch']['config']['path_reward']
    data={};paths={};rows=[];gaps=[];stages=[];provenance={};audit=[];curve={};baselines={}
    reference_expected={case:dict(np.load(roots['warm']/'zero_upper'/f'{case}_path.npz')) for case in CASES}
    for arm,root in roots.items():
        manifest=manifests[arm];cfg=manifest['config'];best=load(root/'shared/best_preference_model.json');status=load(root/'status.json');assert status['state']=='completed' and status['completed_updates']==200 and status['protected_unchanged']
        assert sha(best['checkpoint'])==best['checkpoint_sha256'] and sha(best['actor'])==best['actor_sha256']
        dynamic=load(root/'dynamic_preference_checks/receipt.json');assert dynamic['state']=='not_run' and dynamic['active_lower_ticks']==0
        provenance[arm]={'run':str(root),'manifest_sha256':sha(root/'manifest.json'),'status':status,'selected_update':best['source_update'],'selected_checkpoint_sha256':best['checkpoint_sha256'],'config':cfg,'source':manifest['source'],'dynamic':dynamic,'stages':[]}
        for update in [0,50,100,150,200]:
            folder=(initial[arm] if update==0 else root)/'shared/validation'/f'update_{update:04d}';src=load(folder/'source.json');meta=load(folder/'panel_identity.json');m=load(folder/'preference_metrics.json')
            assert meta['completed'] and src['source_update']==update
            for kind in ('checkpoint','actor'):assert sha(src[kind])==src[kind+'_sha256']
            provenance[arm]['stages'].append({'update':update,'source':src,'panel_identity':meta})
            stages.append(dict(arm=arm,update=update,selected=update==best['source_update'],physical_failures=m['physical_failures'],domain_failures=m['tracking_domain_failures'],strict_goal_hold_failures=m['goal_hold_failures'],own_goal_hold_failures=m.get('research_goal_hold_failures',m['goal_hold_failures']),own_highspeed_endpoint_failures=m['highspeed_endpoint_failures'],own_direction_continuity_failures=m.get('preference_direction_failures',m.get('preference_continuity_failures')),own_qualified=m['qualified_preference'],score=json.dumps(m['score_tuple'])))
            for alpha,token in zip(ALPHAS,TOKENS):
                for case in CASES:
                    file=folder/f'alpha_{token}'/f'{case}.npz';d=dict(np.load(file));n=len(d['time']);assert n==4000 and not np.any(d['physical_failure']) and not np.any(d['domain_exit']);assert not np.any(d['policy_fault']) and not np.any(d['lower_fault']);np.testing.assert_allclose(d['time'],np.arange(n)*.005,rtol=0,atol=2e-6);np.testing.assert_allclose(d['upper_alpha'],alpha,rtol=0,atol=1e-7)
                    r,parts=reward_series(d,alpha,cfg);err=float(np.max(abs(r-d['scored_tick_reward'])));assert err<2e-6
                    np.testing.assert_allclose(sum(parts.values()),r,rtol=0,atol=2e-10)
                    for field,value in parts.items():
                        if field!='failure_tail':np.testing.assert_allclose(value,-cfg['path_reward']['scale']*.005*d['effective_cost_'+field],rtol=2e-5,atol=2e-6)
                    pp=dict(np.load(folder/f'alpha_{token}'/f'{case}_path.npz'))
                    for field in pp:np.testing.assert_allclose(pp[field],reference_expected[case][field],rtol=0,atol=2e-6)
                    audit.append(dict(arm=arm,update=update,alpha=alpha,case=case,max_reward_error=err,steps=n,source_npz=str(file)))
                    ev=d['actual_forward_speed'].astype(float)-d['v_user'];ey=d['path_cross_track'].astype(float);eh=d['path_heading_error'].astype(float);t=d['time'].astype(float)+.005
                    hold=lambda seconds,v,y,h:bool(np.sum(t>20-seconds+1e-5)==round(seconds/.005) and np.all(abs(ev[t>20-seconds+1e-5])<=v) and np.all(abs(ey[t>20-seconds+1e-5])<=y) and np.all(abs(eh[t>20-seconds+1e-5])<=h))
                    goal=case.startswith('straight') or bool(np.any((d['path_progress']>=d['goal_progress'])&(d['goal_section_signed_distance']>=0)))
                    bv,by=(None,None) if case.startswith('straight') else bins(d,cfg)
                    if case.endswith('2.6') and not case.startswith('straight'):
                        bm=m['preference_cases'][case]['per_alpha']['alpha_'+token];np.testing.assert_allclose([bv,by],[bm['speed_rmse_m_s'],bm['path_rmse_m']],rtol=0,atol=1e-10)
                    row=dict(arm=arm,update=update,alpha=alpha,case=case,steps=n,goal=goal,strict_hold_1s=hold(1,.1,.1,.05),research_hold_0p5s=hold(.5,.15,.2,.1),speed_rmse_full=float(np.sqrt(np.mean(ev**2))),path_rmse_full=float(np.sqrt(np.mean(ey**2))),speed_rmse_fullturn_bins=bv,path_rmse_fullturn_bins=by,peak_roll=float(d['peak_roll'].max()),roll_gt_0p302_s=float(np.sum(d['peak_roll']>.302)*.005),peak_abs_path=float(abs(ey).max()),path_gt_0p40_s=float(np.sum(abs(ey)>.4)*.005),reward_sum=float(r.sum()),reward_max_abs_reconstruction_error=err)
                    om=m['per_alpha']['alpha_'+token]['cases'][case]
                    np.testing.assert_allclose([row['speed_rmse_full'],row['path_rmse_full'],row['peak_roll']],[om['speed_rmse'],om['path_rmse'],om['peak_roll']],rtol=0,atol=1e-8)
                    assert row['goal']==om['goal'] and row['strict_hold_1s']==om.get('legacy_final_hold',om['final_hold'])
                    row.update({'signed_reward_sum_'+k:float(v.sum()) for k,v in parts.items()});rows.append(row)
                    if update in (best['source_update'],200):
                        key=(arm,update,alpha,case);data[key]=(d,r,parts);p=dict(np.load(folder/f'alpha_{token}'/f'{case}_path.npz'));paths[case]=p
                        prefix=f'{arm}_u{update}_a{token}_{case}'
                        for field in ('time','actual_xy','actual_forward_speed','v_user','path_cross_track','path_heading_error','path_progress','peak_roll','phi','actual_delta','wheel_speed_proxy','nominal','governed','target_offset','filtered_offset','offsets','final_command'):
                            curve[prefix+'__'+field]=d[field]
                        curve[prefix+'__reward']=r
                        for field,value in parts.items():curve[prefix+'__reward_'+field]=value
            current=[v for v in rows if v['arm']==arm and v['update']==update]
            stages[-1]['strict_goal_hold_failures']=sum(not(v['goal'] and v['strict_hold_1s']) for v in current)
            stages[-1]['recorded_goal_hold_failures']=m['goal_hold_failures']
            for case,q in m['preference_cases'].items():
                a=q['per_alpha']['alpha_0'];b=q['per_alpha']['alpha_1'];gaps.append(dict(arm=arm,update=update,case=case,Ev0=a['speed_rmse_m_s'],Ev1=b['speed_rmse_m_s'],Ey0=a['path_rmse_m'],Ey1=b['path_rmse_m'],Dv=q['D_v_m_s'],Dy=q['D_y_m'],own_endpoint_failures=sum(q['endpoint_main_failures']),own_gap_failures=sum(q['gap_failures']),all_alpha_complete=q['all_alpha_complete_turn']))
        for case in CASES:
            d=dict(np.load(root/'zero_upper'/f'{case}.npz'));p=dict(np.load(root/'zero_upper'/f'{case}_path.npz'))
            for field in p:np.testing.assert_allclose(paths[case][field],p[field],rtol=0,atol=2e-6)
            if case in baselines:
                for field in ('time','actual_xy','peak_roll','actual_forward_speed','path_cross_track'):np.testing.assert_array_equal(baselines[case][field],d[field])
            baselines[case]=d
    csvout(out/'condition_metrics.csv',rows);csvout(out/'preference_metrics.csv',gaps);csvout(out/'checkpoint_summary.csv',stages);csvout(out/'reward_audit.csv',audit)
    for case,p in paths.items():
        for field,value in p.items():curve['reference_'+case+'__'+field]=value
    for case,d in baselines.items():
        for field in ('time','actual_xy','actual_forward_speed','v_user','path_cross_track','path_heading_error','path_progress','peak_roll','phi','actual_delta','wheel_speed_proxy','nominal','governed','target_offset','filtered_offset','offsets','final_command'):curve['B0_'+case+'__'+field]=d[field]
        for alpha,token in zip(ALPHAS,TOKENS):
            r,parts=reward_series(d,alpha,manifests['warm']['config']);curve['B0_'+case+'_a'+token+'__reward']=r
            for field,value in parts.items():curve['B0_'+case+'_a'+token+'__reward_'+field]=value
    public_fields={'time','actual_xy','actual_forward_speed','v_user','path_cross_track','path_heading_error','path_progress','peak_roll','phi','actual_delta','wheel_speed_proxy','reward','xy','s','heading','curvature','goal','turn_end'}
    np.savez_compressed(out/'curve_data.npz',**{k:v for k,v in curve.items() if k.rsplit('__',1)[-1] in public_fields});dump(out/'provenance.json',provenance)
    colors=plt.cm.viridis(np.linspace(.08,.9,5));plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    chosen={k:v['selected_update'] for k,v in provenance.items()}
    def series(case,last=False):
        for arm,ls in [('warm','--'),('scratch','-')]:
            update=200 if last else chosen[arm]
            for alpha,color in zip(ALPHAS,colors):yield arm,alpha,color,ls,data[arm,update,alpha,case]
    def line(ax,x,y,label,color=None,ls='-'):ax.plot(x,y,color=color,ls=ls,lw=.85,label=label)
    overview,ov=plt.subplots(3,4,figsize=(19,14))
    for ci,case in enumerate(CASES):
        fig,axes=plt.subplots(1,2,figsize=(13,5.5))
        phys,pa=plt.subplots(5,2,figsize=(13,14));comp,ca=plt.subplots(11,2,figsize=(13,26));cmd,cc=plt.subplots(10,2,figsize=(13,27));cumcomp,cua=plt.subplots(11,2,figsize=(13,26))
        for last in (False,True):
            col=int(last);title='last: warm200 / scratch200' if last else f'selected: warm{chosen["warm"]} / scratch{chosen["scratch"]}'
            for ax in (axes[col],ov[ci//2,ci%2*2+col]):
                line(ax,*paths[case]['xy'].T,'original reference','black',':');line(ax,*baselines[case]['actual_xy'].T,'B0 local300 + ECBC/ESO','gray');ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=case+' | '+title);ax.set_aspect('equal',adjustable='datalim')
            for arm,alpha,color,ls,(d,r,parts) in series(case,last):
                label=f'{arm} a={alpha:g}';t=d['time']+.005;steps=np.arange(1,len(t)+1)
                for ax in (axes[col],ov[ci//2,ci%2*2+col]):line(ax,*d['actual_xy'].T,label,color,ls);ax.plot(*d['actual_xy'][-1],marker='o',color=color,ms=2)
                for row,y in enumerate((d['actual_forward_speed']-d['v_user'],d['path_cross_track'],d['peak_roll'],r,np.cumsum(r))):line(pa[row,col],steps if row>=3 else t,y,label,color,ls)
                for row,(field,y) in enumerate(parts.items()):line(ca[row,col],steps,y,label,color,ls);ca[row,col].set_ylabel(field);line(cua[row,col],steps,np.cumsum(y),label,color,ls);cua[row,col].set_ylabel('cumulative '+field)
                for row,y in enumerate((d['nominal'][:,0],d['governed'][:,0],d['nominal'][:,1],d['governed'][:,1],d['final_command'][:,0],d['final_command'][:,1],d['actual_forward_speed'],d['actual_delta'],d['offsets'][:,0],d['offsets'][:,1])):line(cc[row,col],t,y,label,color,ls)
            for row,y in enumerate((baselines[case]['actual_forward_speed']-baselines[case]['v_user'],baselines[case]['path_cross_track'],baselines[case]['peak_roll'])):line(pa[row,col],baselines[case]['time']+.005,y,'B0','gray')
            for alpha,color,token in zip(ALPHAS,colors,TOKENS):
                br=curve['B0_'+case+'_a'+token+'__reward'];line(pa[3,col],np.arange(1,len(br)+1),br,f'B0 a={alpha:g}',color,':');line(pa[4,col],np.arange(1,len(br)+1),np.cumsum(br),f'B0 a={alpha:g}',color,':')
                for row,field in enumerate(next(series(case,last))[-1][2]):line(ca[row,col],np.arange(1,len(br)+1),curve['B0_'+case+'_a'+token+'__reward_'+field],f'B0 a={alpha:g}',color,':');line(cua[row,col],np.arange(1,len(br)+1),np.cumsum(curve['B0_'+case+'_a'+token+'__reward_'+field]),f'B0 a={alpha:g}',color,':')
            for row,y in enumerate((baselines[case]['nominal'][:,0],baselines[case]['governed'][:,0],baselines[case]['nominal'][:,1],baselines[case]['governed'][:,1],baselines[case]['final_command'][:,0],baselines[case]['final_command'][:,1],baselines[case]['actual_forward_speed'],baselines[case]['actual_delta'],baselines[case]['offsets'][:,0],baselines[case]['offsets'][:,1])):line(cc[row,col],baselines[case]['time']+.005,y,'B0','gray')
            for row,label in enumerate(('Speed error (m/s)','Signed path error (m)','Substep peak roll (rad)','Per lower-step reward','Cumulative reward')):pa[row,col].set_ylabel(label);pa[row,col].set_xlabel('Actual lower steps' if row>=3 else 'Time (s)')
            pa[2,col].axhline(.302,color='red',ls=':',lw=.7);pa[1,col].axhline(.4,color='red',ls=':',lw=.7);pa[1,col].axhline(-.4,color='red',ls=':',lw=.7)
            for row,label in enumerate(('Nominal speed (m/s)','Governed speed (m/s)','Nominal steer (rad)','Governed steer (rad)','Final steer rate (rad/s)','Final axle rate (rad/s)','Body / wheel / requested speed (m/s)','Actual steer (rad)','Applied speed offset (m/s)','Applied steer offset (rad)')):cc[row,col].set_ylabel(label);cc[row,col].set_xlabel('Time (s)')
            for arm,alpha,color,ls,(d,r,parts) in series(case,last):
                line(cc[6,col],d['time']+.005,d['v_user'],arm+' requested',color,':');line(cc[6,col],d['time']+.005,d['wheel_speed_proxy'],arm+' wheel proxy',color,':')
            for grid in (pa,ca,cua,cc):grid[0,col].set_title(title)
            for grid in (ca,cua):
                for axis in grid[:,col]:axis.set_xlabel('Actual lower steps')
        for figure,grid,suffix in ((fig,axes,'XY'),(phys,pa,'errors_rewards'),(comp,ca,'components'),(cmd,cc,'command_chain'),(cumcomp,cua,'components_cumulative')):
            ax=grid.flat[0];handles,labels=ax.get_legend_handles_labels();figure.legend(handles,labels,loc='lower center',ncol=6,fontsize=7);figure.suptitle(case+' | original requests, no external disturbance | seed911401');figure.tight_layout(rect=(0,.045,1,.97));figure.savefig(out/f'{case}_{suffix}.png',dpi=125);plt.close(figure)
    handles,labels=ov.flat[0].get_legend_handles_labels();overview.legend(handles,labels,loc='lower center',ncol=6,fontsize=8);overview.tight_layout(rect=(0,.04,1,1));overview.savefig(out/'ALL_XY.png',dpi=140);plt.close(overview)
    detail,da=plt.subplots(2,2,figsize=(13,10))
    for row,case in enumerate(('left90_R2_2.6','right90_R2_2.6')):
        path=paths[case];zoom=path['xy'][path['s']<=float(path['turn_end'])+4]
        for col in (0,1):
            ax=da[row,col];line(ax,*path['xy'].T,'original reference','black',':');line(ax,*baselines[case]['actual_xy'].T,'B0','gray')
            for arm,alpha,color,ls,(d,r,parts) in series(case,bool(col)):line(ax,*d['actual_xy'].T,f'{arm} a={alpha:g}',color,ls)
            ax.set(xlim=(zoom[:,0].min()-1,zoom[:,0].max()+1),ylim=(zoom[:,1].min()-1,zoom[:,1].max()+1),xlabel='World X (m)',ylabel='World Y (m)',title=case+(' last' if col else ' selected'));ax.set_aspect('equal',adjustable='box')
    h,l=da[0,0].get_legend_handles_labels();detail.legend(h,l,loc='lower center',ncol=6,fontsize=7);detail.tight_layout(rect=(0,.07,1,1));detail.savefig(out/'TURN_DETAIL.png',dpi=150);plt.close(detail)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for arm,ls in [('warm','--'),('scratch','-')]:
        for case,color in [('left90_R2_2.6','tab:blue'),('right90_R2_2.6','tab:orange')]:
            g=[x for x in gaps if x['arm']==arm and x['case']==case]
            for col,key in enumerate(('Dv','Dy')):line(axes[0,col],[x['update'] for x in g],[x[key] for x in g],arm+' '+case,color,ls)
        for col,case in enumerate(('left90_R2_2.6','right90_R2_2.6')):
            for update in sorted(set((chosen[arm],200))):
                vals=[bins(data[arm,update,alpha,case][0],manifests[arm]['config']) for alpha in ALPHAS];axes[1,col].plot([v[0] for v in vals],[v[1] for v in vals],ls+'o',label=f'{arm} u{update}');
                for alpha,v in zip(ALPHAS,vals):axes[1,col].annotate(str(alpha),v,fontsize=7)
    for col,label in enumerate(('Dv (m/s), positive is intended','Dy (m), positive is intended')):axes[0,col].set(xlabel='Evaluated update / sampling slot',ylabel=label);axes[0,col].axhline(0,color='gray',lw=.6);axes[0,col].axhline(.03 if col==0 else .02,color='red',ls=':',label='strong threshold');axes[0,col].axhline(.02 if col==0 else .001,color='green',ls=':',label='scratch research threshold')
    for col in (0,1):axes[1,col].set(xlabel='Full-turn equal-bin speed RMSE (m/s)',ylabel='Full-turn equal-bin path RMSE (m)',title=('Left' if col==0 else 'Right')+' high-speed turn')
    for ax in axes.flat:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'PREFERENCE.png',dpi=150);plt.close(fig)
    dump(out/'verification.json',dict(policy_traces_checked=len(audit),expected_policy_traces=300,complete_lower_ticks=sum(x['steps'] for x in audit),new_physics=0,new_PPO_updates=0,matched_B0_cases=6,fullturn_bin_recomputations=200,reward_max_abs_error=max(x['max_reward_error'] for x in audit),selected=chosen,dynamic='not_run for both; fixed preference failed',curve_npz_bytes=(out/'curve_data.npz').stat().st_size,curve_npz_sha256=sha(out/'curve_data.npz')))
    print(json.dumps(load(out/'verification.json'),indent=2))
if __name__=='__main__':main()
