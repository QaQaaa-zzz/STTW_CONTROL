"""Pair terminal contact-support windows with every archived alpha in the same case."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from analyze_roll_contacts import roll_state
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--contacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    bins=np.load(a.contacts);contact=bins['mean'];counts=bins['valid_physics_steps'];rows=[];plots={}
    for cp in sorted(a.run.glob('*/contract.json')):
        if not (cp.parent/'trajectory.npz').exists():continue
        c=json.loads(cp.read_text());z=np.load(cp.parent/'trajectory.npz');n=c['shared_physics_contacts']['lane_count'];start=c['shared_physics_contacts']['lane_start'];trace=z['lower_trace'].reshape(-1,n,11);valid=trace[:,:,6].astype(bool);t=(np.arange(len(trace))+1)*c['lower_dt'];labels=c['settings']['case_labels'];phi,omega=roll_state(z['body_state'][:,:,c['bodies'].index('chasis')]);con=contact[:,start:start+n]
        ends={lab:np.flatnonzero(valid[:,j])[-1] for j,lab in enumerate(labels)}
        for lane,lab in enumerate(labels):
            windows={'own_terminal':t[ends[lab]]}
            for target in ['full_0','full_1']:windows['paired_'+target+'_terminal']=t[ends[target]]
            for name,end in windows.items():
                ids=np.flatnonzero(valid[:,lane]&(t>=end-.1-1e-8)&(t<=end+1e-8)&(counts[:,start+lane]>0))
                if len(ids)<2:continue
                i,j=ids[0],ids[-1];f=con[ids,lane,0];r=con[ids,lane,6];nonzero=np.flatnonzero(valid[:j+1,lane]&(counts[:j+1,start+lane]>0)&(con[:j+1,lane,0]>0))
                zero_start=float(t[nonzero[-1]+1]) if len(nonzero) and nonzero[-1]<j else None
                rows.append(dict(case=cp.parent.name,label=lab,window=name,start_s=float(t[i]),end_s=float(t[j]),complete_window=bool(abs(t[j]-t[i]-.1)<1e-7),front_normal_max_N=float(f.max()),front_normal_mean_N=float(f.mean()),front_exact_zero_fraction=float((f==0).mean()),rear_normal_max_N=float(r.max()),rear_normal_mean_N=float(r.mean()),front_zero_suffix_start_s=zero_start,phi_start_deg=float(np.degrees(phi[i,lane])),phi_end_deg=float(np.degrees(phi[j,lane])),omega_start_deg_s=float(np.degrees(omega[i,lane])),omega_end_deg_s=float(np.degrees(omega[j,lane])),steer_rate_mean_rad_s=float(z['measurement'][ids,lane,3].mean()),command_mean_rad_s=float(z['final_command'][ids,lane,0].mean())))
        if cp.parent.name.startswith('single10_v2.6'):
            plots[cp.parent.name]=(t,valid,phi,omega,con,labels)
    with (a.output/'support_windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(3,3,figsize=(15,10),sharex=True,sharey='col',layout='constrained')
    for row,seed in enumerate([101,102,103]):
        t,valid,phi,omega,con,labels=plots[f'single10_v2.6_seed{seed}']
        for lab,color,style in zip(['full_0','full_0.5','full_1'],['#D55E00','#009E73','#B22222'],['--','-','-.']):
            lane=labels.index(lab);mask=valid[:,lane]&(t>=1.5)&(t<=2.6)
            for col,y in enumerate([con[:,lane,0],con[:,lane,6],np.degrees(phi[:,lane])]):axs[row,col].plot(t[mask],y[mask],color=color,ls=style,lw=1.5,label=lab.replace('full_','alpha='))
        for col,title in enumerate(['Front normal force (N)','Rear normal force (N)','Actual roll (deg)']):
            ax=axs[row,col];ax.set_title(f'Group {seed} | '+title);ax.grid(alpha=.18);ax.axhline(0,color='.6',lw=.7);ax.set_xlim(1.5,2.6)
            if row==2:ax.set_xlabel('Time (s)')
    axs[0,0].legend();fig.suptitle('10 cm ridge, 2.6 m/s | loss of front support and subsequent roll failure\nOriginal common start; measured 5 ms mean normal forces; group 103 alpha=1 ends on domain exit',fontsize=13)
    fig.savefig(a.output/'front_support_failure.png',dpi=150);plt.close(fig)
    (a.output/'definition.json').write_text(json.dumps(dict(run=str(a.run),contact_source=str(a.contacts),windows=len(rows),normal_force_zero='exact zero in original nonnegative 5kHz normal-force means; no learned threshold',paired_windows='For each case, last100ms of full_0/full_1 applied to all original lanes, truncated/missing data not filled',new_simulations=0),indent=2));print('windows',len(rows))

if __name__=='__main__':main()
