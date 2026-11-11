"""Audit conditional two-sided braking-distance margins on saved trajectories."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def margin(phi,omega,limit,braking):
    plus=limit-phi-np.maximum(omega,0)**2/(2*braking)
    minus=limit+phi-np.maximum(-omega,0)**2/(2*braking)
    return plus,minus,np.minimum(plus,minus)


def demand(phi,omega,limit):
    room=limit-np.sign(omega)*phi
    result=np.divide(omega**2,2*room,out=np.full_like(phi,np.inf,dtype=float),where=room>0)
    return np.where(abs(phi)>=limit,np.inf,result)


def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=a.output;out.mkdir(exist_ok=True,parents=True);src=a.run/'roll_contact_analysis'
    selection=json.loads((src/'positive_selection.json').read_text())[0];focus=selection['case'];summary=json.loads((a.run/'analysis_summary.json').read_text())['rows'];limit=np.degrees(a.roll_limit_rad)
    attachment_checks=[]
    oldkeys=list(csv.DictReader((a.attachment/'key_calculations.csv').open()));hashes=json.loads((a.attachment/'source_sha256.json').read_text())
    for name,item in hashes.items():
        file=a.attachment/'data'/name;assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
        z=np.load(file);active=z['active_tick'].astype(bool);t=z['time'][active].astype(float);p=np.degrees(z['phi'][active].astype(float));bad=z['physical_failure'][active]|z['contact_failure'][active];stop=np.flatnonzero(bad)[0] if bad.any() else len(t);t=t[:stop];p=p[:stop];w=np.gradient(p,t,edge_order=1)
        for row in oldkeys:
            if float(row['alpha'])!=item['alpha']:continue
            j=np.argmin(abs(t-float(row['time_s'])));h=45-p[j]-max(w[j],0)**2/1000
            error=max(abs(p[j]-float(row['phi_deg'])),abs(w[j]-float(row['omega_deg_s_estimated'])),abs(h-float(row['H_deg_example'])));assert error<1e-8
            attachment_checks.append(dict(file=name,alpha=item['alpha'],time_s=float(t[j]),max_abs_recalculation_error=float(error)))
    traces=[]
    for path in sorted(src.glob('*_seed*.npz')):
        z=np.load(path);seed=int(path.stem.rsplit('seed',1)[1]);labels=z['labels']
        for lane,label in enumerate(labels):
            valid=z['valid'][:,lane];t=z['time'][valid,lane].astype(float);p=np.degrees(z['phi'][valid,lane].astype(float));w=np.degrees(z['euler_roll_rate'][valid,lane].astype(float));acc=np.diff(w)/np.diff(t)
            # Omit the last valid interval from empirical envelope calibration;
            # a contact-triggered terminal state may have mixed recording phases.
            inside=(abs(p[:-1])<limit)&(abs(p[1:])<limit);inside[-1:]=False
            row=next(x for x in summary if x['case']==path.stem and x['label']==label)
            traces.append(dict(case=path.stem,seed=seed,label=str(label),t=t,phi=p,omega=w,acc=acc,inside=inside,summary=row))
    calibration=[x for x in traces if x['seed'] in (101,102)]
    B=max(float(abs(x['acc'][x['inside']]).max()) for x in calibration)
    calibration_peak=max(calibration,key=lambda x:float(abs(x['acc'][x['inside']]).max()))
    protocol=dict(actual_roll_monitor_rad=a.roll_limit_rad,actual_roll_monitor_deg=limit,threshold_source='isaac_path_rollout.py: bad=(post_m[:,0].abs()>.7), checked every5ms; not geometric chassis-contact angle',example_B_deg_s2=a.example_B,example_limit_deg=45,empirical_B_deg_s2=B,calibration_physical_seeds=[101,102],test_physical_seeds=[103],calibration_method='maximum absolute5ms Euler-rate finite difference over calibration groups; exclude terminal interval and endpoints outside0.7rad; no outcome-based tuning; not a certified acceleration upper bound',calibration_peak_case=calibration_peak['case'],calibration_peak_label=calibration_peak['label'],focus=focus,source=str(src),attachment_identity='historical U17024N one-sided; current H75 ridge analyzed separately')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];counter=[];series={};alpha_rows=[]
    for x in traces:
        p,w,t=x['phi'],x['omega'],x['t'];br=demand(p,w,limit);r=dict(case=x['case'],label=x['label'],seed=x['seed'],role='TEST' if x['seed']==103 else 'CAL',cause=x['summary']['cause'],final_hold=x['summary']['final_hold'],observed_s=float(t[-1]),max_abs_observed_acc_deg_s2=float(abs(x['acc'][x['inside']]).max()),empirical_bound_violation_intervals=int(np.sum(abs(x['acc'][x['inside']])>B)),max_B_required_before_monitor_deg_s2=float(br[abs(p)<limit].max()))
        for key,pc,b in [('example45_B500',45,a.example_B),('actual_B500',limit,a.example_B),('actual_empiricalB',limit,B)]:
            hp,hm,h=margin(p,w,pc,b);negative=np.flatnonzero(h<0);pre=np.flatnonzero((h<0)&(abs(p)<limit));j=int(negative[0]) if len(negative) else None
            r[key+'_first_s']=None if j is None else float(t[j]);r[key+'_first_pre_monitor_s']=float(t[pre[0]]) if len(pre) else None;r[key+'_lead_to_end_s']=float(t[-1]-t[pre[0]]) if len(pre) else None;r[key+'_min_deg']=float(h.min())
            if key=='actual_B500' and j is not None:
                stop=np.flatnonzero((np.arange(len(t))>j)&(w*np.sign(w[j])<=0)&(abs(p)<limit))
                if len(stop) and x['summary']['cause']=='full_window':counter.append(dict(case=x['case'],label=x['label'],first_negative_s=float(t[j]),H_at_cross_deg=float(h[j]),rate_reversal_s=float(t[stop[0]]),final_hold=x['summary']['final_hold']))
            if x['case']==focus and x['label'] in ['full_0.5','full_1']:
                series[x['label']+'__'+key]=h
        rows.append(r)
        if x['case']==focus and x['label'] in ['full_0.5','full_1']:
            for name,value in [('time',t),('phi_deg',p),('omega_deg_s',w),('B_required_deg_s2',br),('observed_acc_deg_s2',np.r_[x['acc'],np.nan])]:series[x['label']+'__'+name]=value
        if x['case'].startswith('single10') and x['label'] in ['full_0.5','full_1']:alpha_rows.append(r)
    csv_write(out/'all216.csv',rows);csv_write(out/'ridge_pairs.csv',alpha_rows)
    if counter:csv_write(out/'observed_returns_after_example_crossing.csv',counter)
    np.savez_compressed(out/'focus_curves.npz',**series)
    tables=[]
    for role in ['CAL','TEST']:
        group=[r for r in rows if r['role']==role]
        for key in ['example45_B500','actual_B500','actual_empiricalB']:
            for cause in sorted(set(r['cause'] for r in group)):
                subset=[r for r in group if r['cause']==cause];alarms=[r for r in subset if r[key+'_first_pre_monitor_s'] is not None]
                tables.append(dict(role=role,margin=key,cause=cause,n=len(subset),pre_monitor_alarms=len(alarms),median_lead_s=None if not alarms else float(np.median([r[key+'_lead_to_end_s'] for r in alarms]))))
    csv_write(out/'outcome_table.csv',tables)
    focus_pair=[next(x for x in traces if x['case']==focus and x['label']==label) for label in ['full_0.5','full_1']]
    for x in focus_pair:
        for tt in [1.8,1.805,1.81,1.82,1.9,1.965,1.97]:
            j=np.argmin(abs(x['t']-tt));p=x['phi'][j];w=x['omega'][j];print(x['label'],x['t'][j],p,w,flush=True)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':150})
    colors=['#009E73','#D55E00'];fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    pp=np.linspace(-limit,limit,400);upper=np.sqrt(2*a.example_B*(limit-pp));lower=-np.sqrt(2*a.example_B*(limit+pp))
    axes[0].plot(pp,upper,'k--',label='H=0: actual monitor, EXAMPLE B500');axes[0].plot(pp,lower,'k--');axes[0].fill_between(pp,lower,upper,color='#56B4E9',alpha=.10)
    for color,x in zip(colors,focus_pair):
        t,p,w=x['t'],x['phi'],x['omega'];window=(t>=1.3)&(t<=3.2);axes[0].plot(p[window],w[window],color=color,label=x['label']);axes[0].scatter(p[window][-1],w[window][-1],marker='x',color=color)
        for pc,style in [(45,':'),(limit,'-')]:
            h=margin(p,w,pc,a.example_B)[2];axes[1].plot(t[window],h[window],color=color,ls=style,label=x['label']+(' | 45deg' if pc==45 else ' | 40.107deg'))
    axes[0].set(xlabel='Signed roll (deg)',ylabel='Euler roll rate (deg/s)',title='Recorded phase portrait; boundary conditional on B500')
    axes[1].axhline(0,color='black',lw=.8);axes[1].set(xlabel='Time (s)',ylabel='H (deg)',title='Correcting the angle changes first crossing');
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle(f'{focus} | Both-sided formula, no refit | B500 is VIOLATED by recorded acceleration, not a physical boundary',fontsize=12);fig.savefig(out/'H_formula_check.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True,layout='constrained')
    for color,x in zip(colors,focus_pair):
        t,p,w=x['t'],x['phi'],x['omega'];window=(t>=1.3)&(t<=3.2);required=demand(p,w,limit);axes[0].plot(t[window],required[window],color=color,label=x['label']);axes[1].plot(t[window],w[window],color=color,label=x['label']);inward=-np.sign(w[:-1])*x['acc'];sel=(t[:-1]>=1.3)&(t[:-1]<=3.2);axes[2].plot(t[:-1][sel],inward[sel],color=color,label=x['label'])
    axes[0].set_yscale('log');axes[0].axhline(a.example_B,color='black',ls=':',label='Example500');axes[0].set_ylabel('Required constant braking B_req (deg/s^2)');axes[1].set_ylabel('Euler roll rate (deg/s)');axes[2].set_ylabel('Observed inward acceleration\n5ms interval mean (deg/s^2)');axes[2].axhline(a.example_B,color='black',ls=':');axes[2].set_xlabel('Time (s)')
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle('No fitted B in B_req = omega^2 / [2(phi_c - sign(omega)*phi)]\nphi_c=0.7rad runtime monitor; measured trajectory demand, not a certificate of available control',fontsize=12);fig.savefig(out/'BRAKING_DEMAND.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,b,title in zip(axes,[a.example_B,B],['Example B500','Empirical B frozen on seeds101/102']):
        ax.plot(pp,np.sqrt(2*b*(limit-pp)),'k--');ax.plot(pp,-np.sqrt(2*b*(limit+pp)),'k--')
        for color,x in zip(colors,focus_pair):
            win=(x['t']>=1.3)&(x['t']<=3.2);ax.plot(x['phi'][win],x['omega'][win],color=color,label=x['label'])
        ax.set(xlabel='Roll (deg)',ylabel='Euler roll rate (deg/s)',title=title+f'\nB={b:.1f}deg/s^2');ax.legend(fontsize=8)
    axes[0].set_ylim(axes[1].get_ylim())
    fig.suptitle('A larger allowed braking capacity widens the conditional set; both panels are NOT certified physical recovery sets',fontsize=12);fig.savefig(out/'B_INDEPENDENT_CHECK.png');plt.close(fig)
    overall=dict(attachment_recalculation=attachment_checks,cases=12,trajectories=len(rows),empirical_B_deg_s2=B,test_trajectories=sum(r['role']=='TEST' for r in rows),test_traces_exceeding_empirical_B=sum(r['role']=='TEST' and r['empirical_bound_violation_intervals']>0 for r in rows),observed_full_window_returns_after_B500_cross=len(counter),protocol=protocol)
    (out/'summary.json').write_text(json.dumps(overall,indent=2));print(json.dumps({k:v for k,v in overall.items() if k not in ['attachment_recalculation','protocol']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--attachment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--roll-limit-rad',type=float,default=.7);p.add_argument('--example-B',type=float,default=500);main(p.parse_args())
