"""Local H75 reference-map derivatives; explicitly not a plant closed-loop Jacobian."""
import argparse,hashlib,json,csv,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.torch_path_policy import path_latent

def main(repo,base,out):
    torch.set_num_threads(1)
    bundle=repo/'runs/h75_selected_local300_flat/bundle'; ident=json.loads((bundle/'identity.json').read_text()); actor=bundle/'actor.npz'
    assert hashlib.sha256(actor.read_bytes()).hexdigest()==ident['source_actor']['artifact_sha256']
    w={k:torch.tensor(v,dtype=torch.float64) for k,v in np.load(actor).items()}
    c=json.loads((bundle/'config.json').read_text());r=c['residual_interface'];dt=c['timing']['lower_dt_s']
    decay=torch.exp(-dt/torch.tensor(r['filter_tau_s'],dtype=torch.float64))
    lo=dt*torch.tensor([-r['correction_speed_down_m_s2'],-r['correction_steer_rate_rad_s']],dtype=torch.float64);hi=dt*torch.tensor([r['correction_speed_up_m_s2'],r['correction_steer_rate_rad_s']],dtype=torch.float64)
    low=torch.tensor([r['speed_reference_bounds_m_s'][0],-r['steer_reference_bound_rad']]);high=torch.tensor([r['speed_reference_bounds_m_s'][1],r['steer_reference_bound_rad']])
    rows=[];target_errors=[]
    for folder in sorted((base/'h75_channel_scan').glob('*_seed10[123]')):
        z=np.load(folder/'trajectory.npz');obs=z['observation'];nom=z['nominal'];target_saved=z['target']
        for i in [260,300,340,380,420]:
            original=torch.tensor(obs[i,6],dtype=torch.float64);nominal=torch.tensor(nom[i-1,6],dtype=torch.float64)
            def fun(o):
                a=torch.tanh(path_latent(w,o[None]))[0]
                target=torch.stack((a[0]*torch.where(a[0]>=0,r['speed_positive_scale_m_s'],r['speed_negative_scale_m_s']),a[1]*r['steer_scale_rad']))
                offsets=o[346:348]*torch.tensor([1.,.2]);filtered=o[348:350]*torch.tensor([1.,.2])
                f=decay*filtered+(1-decay)*target
                ref=torch.maximum(low,torch.minimum(high,nominal+offsets+torch.maximum(lo,torch.minimum(hi,f-offsets))))
                return torch.cat((target,ref))
            target_errors.append(float(np.max(abs(fun(original).detach().numpy()[:2]-target_saved[i,6]))))
            for alpha in [0,.25,.5,.75,1]:
                o=original.clone();o[336]=alpha
                jac=torch.autograd.functional.jacobian(fun,o).numpy();f=fun(o).detach().numpy();errs=[]
                for j in [300,301,336]:
                    plus=o.clone();minus=o.clone();plus[j]+=1e-6;minus[j]-=1e-6
                    fd=((fun(plus)-fun(minus))/2e-6).detach().numpy();errs.append(float(np.max(abs(fd-jac[:,j]))))
                rows.append(dict(case=folder.name,step=i,alpha=alpha,target_speed=f[0],target_steer=f[1],reference_speed=f[2],reference_steer=f[3],d_target_speed_dalpha=jac[0,336],d_target_steer_dalpha=jac[1,336],d_reference_speed_dalpha=jac[2,336],d_reference_steer_dalpha=jac[3,336],d_reference_steer_dlatest_roll=jac[3,300]/.3,d_reference_steer_dlatest_IMUx=jac[3,301]/1.5,fd_max_abs_error=max(errs)))
    with (out/'alpha_reference_derivatives.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(samples=len(rows),common_states=len(target_errors),actor_sha256=ident['source_actor']['artifact_sha256'],native_target_max_abs_error=max(target_errors),jacobian_fd_max_abs_error=max(x['fd_max_abs_error'] for x in rows),speed_immediate_alpha_derivative_zero=sum(int(abs(x['d_reference_speed_dalpha'])<1e-12) for x in rows),steer_immediate_alpha_derivative_zero=sum(int(abs(x['d_reference_steer_dalpha'])<1e-12) for x in rows),scope='Frozen saved history; change only alpha; Actor + one 5ms filter/slew step. No vehicle, ESO or lower-policy state propagation. Latest roll-rate input is IMU x, not Euler derivative.')
    (out/'alpha_reference_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--base',type=Path,default=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.repo,a.base,a.output)
