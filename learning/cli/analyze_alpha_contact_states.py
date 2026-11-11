"""Observed alpha-to-operating-state maps at paired wheel-contact events."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sttw_control.margin_theory import impulse_capacity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.controller.read_text());rows=[]
    for path in sorted(a.source.glob('*_seed*.npz')):
        z=np.load(path);raw=np.load(a.source.parent/path.stem/'trajectory.npz');fields=list(z['contact_fields'])
        for lane,label in enumerate(z['labels']):
            valid=np.flatnonzero(z['valid'][:,lane]);t=z['time'][:,lane];events={}
            for wheel in ['front','rear']:
                hits=valid[z['contact_mean'][valid,lane,fields.index(wheel+'_raised')]>0]
                if len(hits):events[wheel]=int(hits[0])
            for wheel,onset in events.items():
                for lag in [-.1,0.,.05,.1,.2]:
                    target=t[onset]+lag;i=int(np.argmin(abs(t-target)))
                    if not z['valid'][i,lane]:continue
                    window=valid[(t[valid]>=t[i])&(t[valid]<=t[i]+.1+1e-9)];speed=float(z['speed'][i,lane]);cap=impulse_capacity(speed,cfg) if speed>=.5 else {}
                    row=dict(case=path.stem,label=str(label),wheel=wheel,lag_s=lag,time_s=float(t[i]),contact_s=float(t[onset]),measured_front_rear_delay_s=float(t[events['rear']]-t[events['front']]) if len(events)==2 else None,speed_m_s=speed,phi_deg=float(np.degrees(z['phi'][i,lane])),omega_deg_s=float(np.degrees(z['euler_roll_rate'][i,lane])),delta_rad=float(z['delta'][i,lane]),delta_dot_rad_s=float(z['delta_dot'][i,lane]),command_rad_s=float(z['command'][i,lane]),speed_offset_m_s=float(raw['offsets'][i,lane,0]),steer_offset_rad=float(raw['offsets'][i,lane,1]),next100ms_rate_limit_fraction=float((abs(z['command'][window,lane])>=2.999).mean()),next_window_s=float(t[window[-1]]-t[i]),path_error_m=float(z['path_error'][i,lane]),nominal_J_sat_rad_s=cap.get('J_sat_rad_s'),nominal_p1=cap.get('p1_roll_rate'),nominal_L_over_v_s=cap.get('wheel_delay_s'))
                    rows.append(row)
    with (a.output/'contact_states.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels=['full_0','full_0.25','full_0.5','full_0.75','full_1'];alphas=[0,.25,.5,.75,1]
    for context in sorted(set(r['case'].split('_seed')[0] for r in rows)):
        fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for j,wheel in enumerate(['front','rear']):
            for col,(key,title) in enumerate([('speed_m_s','Actual speed (m/s)'),('delta_rad','Actual steering angle (rad)'),('omega_deg_s','Euler roll rate (deg/s)')]):
                ax=axs[j,col]
                for seed,color,style in zip([101,102,103],['#0072B2','#D55E00','#009E73'],['-','--','-.']):
                    values=[next((r[key] for r in rows if r['case']==f'{context}_seed{seed}' and r['wheel']==wheel and r['lag_s']==0 and r['label']==lab),np.nan) for lab in labels]
                    ax.plot(alphas,values,marker='o',color=color,ls=style,label=f'Group {seed}')
                ax.set_title(wheel.capitalize()+' contact | '+title);ax.set_xlabel('Alpha');ax.set_xticks(alphas);ax.grid(alpha=.18)
                if j==0 and col==0:ax.legend()
        fig.suptitle(context+' | actual operating states at first raised-wheel contact\nSame original start; each alpha has its own measured contact time (5 ms bins)')
        fig.savefig(a.output/(context+'.png'),dpi=150);plt.close(fig)
    (a.output/'definition.json').write_text(json.dumps(dict(source=str(a.source),controller=str(a.controller),rows=len(rows),events='First normal contact >1N at world z>.002m from original 5kHz contact bins; sample at bin time',lags_s=[-.1,0,.05,.1,.2],map='Observed per-case response, no fitting or causal equivalence of event states',capacity='Source-parameter linear unsaturated impulse limit, NOT actual terrain impulse',new_simulations=0),indent=2))
    print('rows',len(rows))

if __name__=='__main__':main()
