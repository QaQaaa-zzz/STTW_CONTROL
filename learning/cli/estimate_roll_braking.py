"""Estimate realized stopping capacity and align the ridge failure with contact logs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def recovery_segments(src):
    rows=[]
    for path in sorted(src.glob('*_seed*.npz')):
        z=np.load(path)
        for lane,label in enumerate(z['labels']):
            if str(label).endswith('duplicate'):continue
            ok=z['valid'][:,lane];t=z['time'][ok,lane];p=np.degrees(z['phi'][ok,lane]);w=np.degrees(z['euler_roll_rate'][ok,lane]);sg=np.sign(w)
            cuts=np.r_[0,np.flatnonzero(sg[1:]!=sg[:-1])+1,len(t)]
            for a,e in zip(cuts[:-1],cuts[1:]):
                if e==len(t):continue
                k=a+np.argmax(abs(w[a:e]));direction=sg[k];duration=t[e]-t[k]
                if abs(w[k])<30 or duration<.02 or duration>.5 or abs(p[k:e+1]).max()>=np.degrees(.7):continue
                f=abs(w[e-1])/(abs(w[e-1])+abs(w[e]));endphi=p[e-1]+f*(p[e]-p[e-1]);dist=direction*(endphi-p[k])
                if dist<=0:continue
                contact=z['contact_mean'][ok,lane][k:e+1];support=np.column_stack([contact[:,0]>1,contact[:,6]>1])
                rows.append(dict(case=path.stem,label=str(label),start_s=float(t[k]),stop_s=float(t[e-1]+f*(t[e]-t[e-1])),omega_start_deg_s=float(w[k]),phi_start_deg=float(p[k]),speed_start=float(z['speed'][ok,lane][k]),duration_s=float(duration),B_eff_deg_s2=float(w[k]**2/(2*dist)),no_wheel_support_fraction=float((~support.any(1)).mean())))
    return rows

def main(src,out):
    out.mkdir(exist_ok=True,parents=True);rows=recovery_segments(src);write_csv(out/'recovery_segments.csv',rows)
    matched=[r for r in rows if r['case'].startswith('single') and not r['case'].endswith('101') and r['omega_start_deg_s']<0 and 1.8<=r['speed_start']<=2.6]
    write_csv(out/'reference_segments.csv',matched);values=np.array([r['B_eff_deg_s2'] for r in matched]);B=float(np.median(values))
    z=np.load(src/'single10_v2.6_seed101.npz');case='single10_v2.6_seed101';fields=list(z['contact_fields']);timeline=[];allcurves={};fits=[]
    fig,axes=plt.subplots(6,1,figsize=(12,15),sharex=True,layout='constrained')
    for lane,color,style in [(6,'#009E73','-'),(12,'#D55E00','--')]:
        label=str(z['labels'][lane]);ok=z['valid'][:,lane];t=z['time'][ok,lane];p=np.degrees(z['phi'][ok,lane]);w=np.degrees(z['euler_roll_rate'][ok,lane]);c=z['contact_mean'][ok,lane];cmd=z['command'][ok,lane];dr=z['delta_dot'][ok,lane];delta=z['delta'][ok,lane]
        req=np.divide(w*w,2*(np.degrees(.7)-np.sign(w)*p),out=np.full_like(w,np.nan),where=abs(p)<np.degrees(.7));front=c[:,4]+c[:,5];rear=c[:,10]+c[:,11]
        for aa,ee in [(1.83,1.92),(1.82,1.92),(1.84,1.92),(1.85,1.915)]:
            use=(t>=aa)&(t<=ee);b,offset=np.polyfit(t[use],w[use],1);fits.append(dict(label=label,start_s=aa,end_s=ee,B_realized_deg_s2=float(b),rate_fit_rmse_deg_s=float(np.sqrt(np.mean((w[use]-b*t[use]-offset)**2))),max_front_N=float(c[use,0].max()),max_rear_N=float(c[use,6].max())))
        for tt in [1.75,1.8,1.83,1.9,1.92,1.925,1.93,1.94,1.95,1.965,1.97]:
            i=np.argmin(abs(t-tt));timeline.append(dict(label=label,time_s=float(t[i]),phi_deg=float(p[i]),omega_deg_s=float(w[i]),B_required_deg_s2=float(req[i]),speed_m_s=float(z['speed'][ok,lane][i]),delta_rad=float(delta[i]),delta_dot_rad_s=float(dr[i]),command_rad_s=float(cmd[i]),front_normal_N=float(c[i,0]),rear_normal_N=float(c[i,6]),front_roll_moment_Nm=float(front[i]),rear_roll_moment_Nm=float(rear[i]),motor_Nm=float(c[i,14])))
        sel=(t>=1.65)&(t<=2.1)
        for ax,y in zip(axes,[p,w,req,cmd,rear,c[:,6]]):ax.plot(t[sel],y[sel],color=color,ls=style,label=label)
        axes[3].plot(t[sel],dr[sel],color=color,ls=':',alpha=.7,label=label+' actual')
        for key,val in [('time',t),('phi_deg',p),('omega_deg_s',w),('B_required_deg_s2',req),('command_rad_s',cmd),('actual_steer_rate_rad_s',dr),('front_roll_moment_Nm',front),('rear_roll_moment_Nm',rear),('front_normal_N',c[:,0]),('rear_normal_N',c[:,6])]:allcurves[label+'__'+key]=val
    for ax in axes:ax.axvspan(1.925,1.95,color='gray',alpha=.15);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
    axes[0].axhline(-np.degrees(.7),color='black',lw=.7);axes[2].axhline(B,color='black',ls=':',label=f'Other-group recovery median B={B:.0f}');axes[2].set_yscale('log');axes[2].legend(fontsize=8)
    axes[3].axhline(-3,color='black',lw=.7);axes[3].set_ylim(-4,12)
    for ax,label in zip(axes,['Roll (deg)','Euler roll rate (deg/s)','Required braking (deg/s^2; log)','Steering speed (rad/s)','Rear contact roll moment (Nm)','Rear normal force (N)']):ax.set_ylabel(label)
    axes[-1].set_xlabel('Time (s)');fig.suptitle('10cm ridge | 2.6m/s | seed101 | H75 fixed\nGray: rear re-contact shock in full_1; moments about whole-vehicle COM, not Euler generalized torque',fontsize=12)
    fig.savefig(out/'FAILURE_CHAIN.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');ax.hist(values,bins=16,color='#0072B2',alpha=.7);ax.axvline(B,color='black',label=f'Median {B:.0f}');ax.set(xlabel='Realized stopping-distance equivalent B (deg/s^2)',ylabel='Recovery segments',title=f'Other physical groups102/103: {len(values)} ridge recovery segments\nNegative roll rate, entry speed1.8-2.6m/s; events within traces are correlated');ax.legend();fig.savefig(out/'B_ESTIMATE.png',dpi=160);plt.close(fig)
    write_csv(out/'timeline.csv',timeline);write_csv(out/'local_braking_fits.csv',fits);np.savez_compressed(out/'curves.npz',**allcurves)
    summary=dict(source=str(src),total_recovery_segments=len(rows),reference_segments=len(matched),reference_traces=len(set((r['case'],r['label']) for r in matched)),reference_B_quantiles_deg_s2=dict(zip(['p10','p25','p50','p75','p90','max'],map(float,np.percentile(values,[10,25,50,75,90,100])))),definition='B_eff=omega_start^2/(2*directional roll displacement until first rate zero); selected sign-run peak |omega|>=30deg/s;20-500ms stop, all states inside0.7rad; duplicates omitted',role='Exploratory realized recovery capacity; empirical distribution, not maximum control authority; groups102/103 excluded focus101, not unseen test',local_fit=fits)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();main(args.source,args.output)
