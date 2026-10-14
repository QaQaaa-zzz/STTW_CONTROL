"""Scratch-only offline figures. Each figure uses exactly one checkpoint; no simulation."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES=['straight_2.0','straight_2.6','left90_R2_2.0','left90_R2_2.6','right90_R2_2.0','right90_R2_2.6']
ALPHAS=[0,.25,.5,.75,1]
TOKENS=['0','0p25','0p5','0p75','1']
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=Path('/home/qy/STTW_CONTROL'));a=ap.parse_args()
    wt=a.workspace/'runs/worktrees/path-unified-scratch';root=wt/'runs/path_unified_scratch_200_20261014/resume_20261014_01';out=Path(__file__).parent
    sys.path.insert(0,str(wt/'learning/src'))
    from sttw_control.path_command_reporting import reward_series
    cfg=load(root/'manifest.json')['config'];best=load(root/'shared/best_preference_model.json');assert best['source_update']==100
    colors=plt.cm.viridis(np.linspace(.08,.9,5));plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    coordinates={};audit=[];provenance={'run':str(root),'seed':911401,'B0':'frozen local300 + ECBC/ESO','new_physics_ticks':0,'new_PPO_updates':0,'figures_single_checkpoint':True,'stages':{}}
    def finish(fig,axes,path,title):
        h,l=np.asarray(axes).flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,fontsize=7);fig.suptitle(title+' | seed911401 | no external disturbance',fontsize=10);fig.tight_layout(rect=(0,.075,1,.95));fig.savefig(path,dpi=130);plt.close(fig)
    for slot,name in [(100,'selected100'),(200,'last200')]:
        dest=out/name;dest.mkdir(exist_ok=True);folder=root/'shared/validation'/f'update_{slot:04d}';src=load(folder/'source.json')
        for kind in ['checkpoint','actor']:assert sha(src[kind])==src[kind+'_sha256']
        provenance['stages'][name]={'sampling_slot':slot,'recoverable_learner_updates':slot-6,'source':src}
        title=f'Scratch {name} | slot{slot}, learner updates{slot-6} | SHA {src["checkpoint_sha256"][:12]}'
        if slot==100:
            for image in (root/'report').glob('*.png'):shutil.copyfile(image,dest/image.name)
        allfig,allaxes=plt.subplots(2,3,figsize=(14,8));detail,dax=plt.subplots(1,2,figsize=(12,6));detail_i=0
        for ci,case in enumerate(CASES):
            base=dict(np.load(root/'zero_upper'/f'{case}.npz'));path=dict(np.load(root/'zero_upper'/f'{case}_path.npz'));series=[]
            coordinates['reference_'+case]=path['xy'];coordinates['B0_'+case]=base['actual_xy']
            for alpha,token,color in zip(ALPHAS,TOKENS,colors):
                d=dict(np.load(folder/f'alpha_{token}'/f'{case}.npz'));assert len(d['time'])==4000
                assert not np.any(d['physical_failure']) and not np.any(d['domain_exit'])
                np.testing.assert_allclose(d['upper_alpha'],alpha,rtol=0,atol=1e-7)
                p=dict(np.load(folder/f'alpha_{token}'/f'{case}_path.npz'))
                for k in p:np.testing.assert_allclose(p[k],path[k],rtol=0,atol=2e-6)
                r,parts=reward_series(d,alpha,cfg);err=float(np.max(abs(r-d['scored_tick_reward'])));assert err<2e-6
                np.testing.assert_allclose(sum(parts.values()),r,rtol=0,atol=2e-10)
                series.append((alpha,color,d,r,parts));coordinates[f'{name}_{case}_alpha{token}']=d['actual_xy'];audit.append({'stage':name,'case':case,'alpha':alpha,'steps':len(r),'max_reward_error':err})
            xyfig,xyax=plt.subplots(figsize=(7,6))
            def xy(ax,zoom=False):
                ax.plot(*path['xy'].T,'k:',lw=1,label='Original fixed reference');ax.plot(*base['actual_xy'].T,color='gray',lw=1,label='B0 local300 + ECBC/ESO')
                for alpha,color,d,r,parts in series:ax.plot(*d['actual_xy'].T,color=color,lw=1,label=f'Scratch alpha={alpha:g}');ax.plot(*d['actual_xy'][-1],marker='o',color=color,ms=2)
                ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=case);ax.set_aspect('equal',adjustable='box')
                if zoom:
                    segment=path['xy'][path['s']<=float(path['turn_end'])+4];ax.set(xlim=(segment[:,0].min()-.4,segment[:,0].max()+.4),ylim=(segment[:,1].min()-.4,segment[:,1].max()+.4))
            xy(allaxes.flat[ci]);xy(xyax)
            if slot==200:finish(xyfig,[xyax],dest/f'{case}_XY.png',title)
            else:plt.close(xyfig)
            if case in ['left90_R2_2.6','right90_R2_2.6']:xy(dax[detail_i],True);detail_i+=1
            if slot==100:continue
            fig,axes=plt.subplots(5,1,figsize=(10,11),sharex=False)
            labels=['Original speed error (m/s)','Original signed path error (m)','Substep peak roll (rad)','Per-step reward','Cumulative reward']
            for alpha,color,d,r,parts in series:
                t=d['time']+.005;steps=np.arange(1,len(r)+1)
                for j,y in enumerate([d['actual_forward_speed']-d['v_user'],d['path_cross_track'],d['peak_roll'],r,np.cumsum(r)]):axes[j].plot(steps if j>=3 else t,y,color=color,lw=.8,label=f'Scratch alpha={alpha:g}')
                br,_=reward_series(base,alpha,cfg)
                for j,y in [(3,br),(4,np.cumsum(br))]:axes[j].plot(steps,y,color=color,ls=':',lw=.7,label=f'B0 rescored alpha={alpha:g}')
            for j,y in enumerate([base['actual_forward_speed']-base['v_user'],base['path_cross_track'],base['peak_roll']]):axes[j].plot(base['time']+.005,y,color='gray',label='B0')
            axes[2].axhline(.302,color='red',ls=':',lw=.8)
            for j,ax in enumerate(axes):ax.set(ylabel=labels[j],xlabel='Actual lower steps' if j>=3 else 'Time (s)')
            finish(fig,axes,dest/f'{case}_errors_rewards.png',title+' | '+case)
            fields=list(series[0][-1]);fig,axes=plt.subplots(len(fields),2,figsize=(14,24))
            for alpha,color,d,r,parts in series:
                _,bp=reward_series(base,alpha,cfg);steps=np.arange(1,len(r)+1)
                for j,k in enumerate(fields):
                    for col in [0,1]:
                        y=parts[k] if col==0 else np.cumsum(parts[k]);by=bp[k] if col==0 else np.cumsum(bp[k]);axes[j,col].plot(steps,y,color=color,lw=.8,label=f'Scratch alpha={alpha:g}');axes[j,col].plot(steps,by,color=color,ls=':',lw=.7,label=f'B0 alpha={alpha:g}');axes[j,col].set(ylabel=k,xlabel='Actual lower steps')
            axes[0,0].set_title('Signed per-step components');axes[0,1].set_title('Cumulative signed components');finish(fig,axes,dest/f'{case}_components.png',title+' | '+case)
            fig,axes=plt.subplots(6,2,figsize=(14,15));labels=['Requested speed (m/s)','Governed speed (m/s)','Nominal steer (rad)','Governed steer (rad)','Final steer rate (rad/s)','Final axle rate (rad/s)','Body speed (m/s)','Wheel proxy (m/s)','Wheel proxy minus body (m/s)','Applied speed offset (m/s)','Applied steer offset (rad)','Actual steer (rad)']
            for alpha,color,d,r,parts in series+[(None,'gray',base,None,None)]:
                ys=[d['v_user'],d['governed'][:,0],d['nominal'][:,1],d['governed'][:,1],d['final_command'][:,0],d['final_command'][:,1],d['actual_forward_speed'],d['wheel_speed_proxy'],d['wheel_speed_proxy']-d['actual_forward_speed'],d['offsets'][:,0],d['offsets'][:,1],d['actual_delta']]
                for ax,y,label in zip(axes.flat,ys,labels):ax.plot(d['time']+.005,y,color=color,lw=.8,label='B0' if alpha is None else f'Scratch alpha={alpha:g}');ax.set(ylabel=label,xlabel='Time (s)')
            finish(fig,axes,dest/f'{case}_command_chain.png',title+' | '+case)
        if slot==200:finish(allfig,allaxes,dest/'ALL_XY.png',title)
        else:plt.close(allfig)
        finish(detail,dax,dest/'TURN_DETAIL.png',title)
        text=f'# Scratch {name}\n\n![High-speed turn detail](TURN_DETAIL.png)\n\n{title}. One checkpoint only; five alpha values, original reference and matching B0. All 30 traces last 20 s, with no physical/domain failure. Circles mark real final samples; no external disturbance. Reward baseline is rescored separately for each alpha; returns across alpha are not physical rankings.\n\n[All six XY](ALL_XY.png)\n\n'
        for case in CASES:
            suffixes=['XY','physical','reward','components','command_chain'] if slot==100 else ['XY','errors_rewards','components','command_chain']
            text+='- '+case+': '+' · '.join(f'[{s}]({case}_{s}.png)' for s in suffixes)+'\n'
        (dest/'INDEX.md').write_text(text)
    np.savez_compressed(out/'xy_coordinates.npz',**coordinates)
    write(out/'provenance.json',provenance);write(out/'reward_verification.json',{'traces':len(audit),'max_error':max(x['max_reward_error'] for x in audit),'audit':audit,'new_physics_ticks':0,'new_PPO_updates':0})
    for filename in ['condition_metrics.csv','preference_metrics.csv','checkpoint_summary.csv']:
        source=a.workspace/'runs/worktrees/path-strong-preference/docs/evidence/alpha_unified_results_20261015'/filename
        with source.open() as f:rows=[r for r in csv.DictReader(f) if r['arm']=='scratch']
        with (out/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps({'stages':list(provenance['stages']),'traces':len(audit),'max_reward_error':max(x['max_reward_error'] for x in audit),'PNG':len(list(out.rglob('*.png')))}))
if __name__=='__main__':main()
