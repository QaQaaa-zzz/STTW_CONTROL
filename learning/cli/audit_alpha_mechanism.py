"""Recompute report claims against the frozen controller and original 72 cases."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from sttw_control.torch_controller import TorchController
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
c=json.loads(a.controller.read_text());ctrl=TorchController(c,1,dtype=torch.float64);sweep=[];error=0.;response_error=0.
for v in sorted(set(np.linspace(1.5,3.5,81).tolist()+[2.25,2.567])):
 cc,b,aa,ratio,g=ctrl._system(torch.tensor([v],dtype=torch.float64));cc,b,aa=float(cc[0]),float(b[0]),float(aa[0]);kd,kf,kp=g[0].numpy();A=np.array([[-kd,-kf,-kp],[0,0,1],[b-cc*kd,aa-cc*kf,-cc*kp]])
 error=max(error,float(np.max(abs(np.poly(A)-np.array([1,15,75,125])))))
 def response(t):return np.exp(-5*t)*(t+(kd-5)*t*t/2)
 opt=minimize_scalar(lambda t:-response(t),bounds=(0,2),method='bounded')
 for t in np.linspace(0,2,31):response_error=max(response_error,abs((expm(A*t)@np.array([0,0,1]))[1]-response(t)))
 sweep.append(dict(speed=v,a=aa,b=b,c=cc,k_delta=kd,k_phi=kf,k_p=kp,roll_peak=float(response(opt.x)),peak_time=float(opt.x),roll_term_01=abs(.1*kf)))
x=next(r for r in sweep if r['speed']==2.25);y=next(r for r in sweep if r['speed']==2.567)
metrics=json.loads((a.evaluation/'gravel_task_metrics.json').read_text());methods=['alpha0','alpha025','alpha05','alpha075','alpha1','Zero'];counts={}
for method in methods:
 rows=[r for r in metrics if r['method']==method];counts[method]=dict(n=len(rows),complete=sum(not r['failed'] for r in rows),arrival_hold=sum(not r['failed'] and r['goal_crossed'] and r['final_hold'] for r in rows),**{t:sum(r['task_'+t] for r in rows) for t in ['A','B','C']})
assert [counts[m]['arrival_hold'] for m in methods]==[5,7,11,4,5,0]
contact=a.evaluation/'batches/case0/evaluation/physics_contacts';idx=json.loads((contact/'index.json').read_text());missing=[x['file'] for x in idx['chunks'] if not (contact/x['file']).exists()];assert not missing
summary=dict(counts=counts,nominal_peak_increase_pct=100*(y['roll_peak']/x['roll_peak']-1),nominal_roll_control_decrease_pct=100*(1-y['roll_term_01']/x['roll_term_01']),polynomial_max_error=error,expm_max_error=response_error,speed_points=len(sweep),contact_chunks=len(idx['chunks']),contact_record_window=idx.get('record_window_s'),contact_missing=missing,controller_sha256=hashlib.sha256(a.controller.read_bytes()).hexdigest(),scope='Frozen-speed nominal ECBC only; no ESO, Local300, saturation or nonlinear contact certificate')
(a.output/'audit.json').write_text(json.dumps(summary,indent=2));(a.output/'nominal_speed_sweep.json').write_text(json.dumps(sweep,indent=2));print(json.dumps(summary,indent=2))
