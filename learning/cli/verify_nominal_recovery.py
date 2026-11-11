"""Exact nominal optimal-angle envelope and independent constrained LP checks."""
import argparse,csv,itertools,json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog
from analyze_roll_contacts import nominal_coeff


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def segment(x,t,a1,a2,a4,u):
    p,w,d=x;lam=np.sqrt(a4);ch=np.cosh(lam*t);sh=np.sinh(lam*t);f0=a2*d+a1*u;f1=a2*u
    return (p*ch+w*sh/lam+f0*(ch-1)/a4+f1*(sh/lam-t)/a4,p*lam*sh+w*ch+f0*sh/lam+f1*(ch-1)/a4,d+u*t)


def envelope(v,x,t,c):
    a1,a2,a4=nominal_coeff(np.array(v),c);side=1 if x[1]>=0 else -1;y=side*np.asarray(x);hit=max(0,(.8-y[2])/3);first=segment(y,np.minimum(t,hit),a1,a2,a4,3);return segment(first,np.maximum(t-hit,0),a1,a2,a4,0),side


def classify(v,x,c):
    t=np.linspace(0,1,4001);(p,w,d),side=envelope(v,x,t,c);st=np.flatnonzero(w<=0);cross=np.flatnonzero(p>.7);s=int(st[0]) if len(st) else len(t);f=int(cross[0]) if len(cross) else len(t)
    outcome='braked' if s<f else 'unavoidable_nominal_limit' if f<len(t) else 'unresolved';j=min(s,f,len(t)-1)
    return dict(outcome=outcome,event_s=float(t[j]),phi_event_deg=float(np.degrees(side*p[j])),angle_certificate_excess_rad=float(p[j]-.7) if outcome=='unavoidable_nominal_limit' else None)


def main(repo,base,out):
    out.mkdir(exist_ok=True,parents=True);c=json.loads((repo/'runs/isaac_signals_parity/dynamic.json').read_text());tests=[]
    for v,phi,w,d in itertools.product([1.8,2.3,2.6],[-30,-15,0,15,30],[-400,-200,-100,-50,50,100,200,400],[-.4,0,.4]):
        row=dict(v=v,phi_deg=phi,omega_deg_s=w,delta_rad=d);row.update(classify(v,[np.radians(phi),np.radians(w),d],c));tests.append(row)
    write_csv(out/'exact_envelope360.csv',tests)
    # Independent optimization over100 control intervals with all angle/rate constraints.
    checks=[];N=100;dt=.005;L=np.tril(np.ones((N,N)))*dt
    for v,d,Tsign,w0 in itertools.product([1.8,2.3,2.6],[-.4,0,.4],[-1,1],[100,200]):
        a1,a2,a4=nominal_coeff(np.array(v),c);A=np.array([[0,1,0,0],[a4,0,a2,a1],[0,0,0,1],[0,0,0,0]],float);E=expm(A*dt);F=E[:3,:3];G=E[:3,3];x=np.array([0,np.radians(w0),Tsign*d]);coeff=np.array([(np.linalg.matrix_power(F,N-1-j)@G)[0] for j in range(N)]);const=(np.linalg.matrix_power(F,N)@x)[0]
        result=linprog(coeff,A_ub=np.vstack((L,-L)),b_ub=np.r_[np.full(N,.8-x[2]),np.full(N,.8+x[2])],bounds=[(-3,3)]*N,method='highs');assert result.success
        state=x.copy();actions=[]
        for j in range(N):
            u=min(3,(.8-state[2])/dt);actions.append(u);state=F@state+G*u
        error=abs(state[0]-(const+result.fun));assert error<1e-7
        (continuous,_,_),_=envelope(v,[0,Tsign*np.radians(w0),d],np.array([.5]),c)
        assert continuous[0]<=state[0]+1e-8
        checks.append(dict(v=v,delta0=d,direction=Tsign,omega0_deg_s=Tsign*w0,lp_min_angle_rad=float(const+result.fun),discrete_envelope_rad=float(state[0]),abs_error_rad=float(error),continuous_lower_angle_rad=float(continuous[0]),lp_max_violation=float(max(0,np.max(abs(x[2]+L@result.x))-.8,np.max(abs(result.x))-3))))
    write_csv(out/'independent_lp36.csv',checks)
    rows=[]
    for name in ['single10_v2.6_seed101','single10_v2.6_seed103','dense10_v2.3_seed101']:
        z=np.load(base/'h75_channel_scan/roll_contact_analysis'/f'{name}.npz')
        for lab in ['full_0','full_0.25','full_0.5','full_0.75','full_1']:
            l=list(z['labels']).index(lab)
            for tt in [1.5,1.75,1.9,1.94,2.1]:
                i=np.argmin(abs(z['time'][:,l]-tt))
                if not z['valid'][i,l] or abs(z['phi'][i,l])>=.7:continue
                x=[float(z[k][i,l]) for k in ['phi','euler_roll_rate','delta']];v=float(z['speed'][i,l]);r=dict(case=name,label=lab,t=float(z['time'][i,l]),v=v,phi_deg=float(np.degrees(x[0])),omega_deg_s=float(np.degrees(x[1])),delta_rad=x[2]);r.update(classify(v,x,c));rows.append(r)
    write_csv(out/'recorded_states_frozen_model.csv',rows)
    summary=dict(fixed_cases=len(tests),nominal_unavoidable=sum(x['outcome']=='unavoidable_nominal_limit' for x in tests),braked=sum(x['outcome']=='braked' for x in tests),lp_checks=len(checks),lp_max_angle_difference_rad=max(x['abs_error_rad'] for x in checks),recorded_states=len(rows),assumptions='Fixed speed, constant coefficients a1<0,a2<0,a4>0, d=0; ideal actual |delta_dot|<=3 and |delta|<=.8; recorded-state tests freeze subsequent speed and remove future contact disturbance; not full PhysX certificates')
    (out/'nominal_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--base',type=Path,default=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.repo,a.base,a.output)
