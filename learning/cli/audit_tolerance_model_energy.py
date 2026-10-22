"""DEV test of submitted scalar effort formula; no training or new simulation.

Calibrate one effort threshold per model using ONLY 2.3m/s labels. Identify
local force mapping from the first 5ms +/-0.5N probe response at each snapshot.
No force-bound, input-feasibility, exact dynamics, or safety certification.
"""
import sys,json,pickle,argparse,math,hashlib
from pathlib import Path
from dataclasses import asdict,replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sttw_control.tolerance_state_features import snapshot_features
from sttw_control.tolerance_batch_layout import contexts,ONSETS,case_name,FORCES
from sttw_control.env import load_config
from sttw_control.teleop_commands import CONFIG_PATH

def choose_threshold(energy,labels):
 energy=np.asarray(energy,float);labels=np.asarray(labels,bool)
 if not np.isfinite(energy).all() or not labels.any() or labels.all():return None
 u=np.unique(energy);candidates=np.r_[0.,(u[:-1]+u[1:])/2,np.nextafter(u[-1],np.inf)]
 best=None
 for t in candidates:
  pred=energy<=t;tpr=float(np.mean(pred[labels]));tnr=float(np.mean(~pred[~labels]));false_safe=int(np.sum(pred&~labels));key=((tpr+tnr)/2,-false_safe,-float(t))
  if best is None or key>best[0]:best=(key,float(t))
 return best[1]

def score(rows,threshold,method):
 if threshold is None:return dict(status='UNIDENTIFIABLE_SINGLE_CLASS_CALIBRATION')
 predictions=[r[f'energy_{method}']<=threshold for r in rows];labels=[r['actual_success'] for r in rows]
 pos=[p for p,y in zip(predictions,labels) if y];neg=[p for p,y in zip(predictions,labels) if not y]
 return dict(n=len(rows),correct=sum(p==y for p,y in zip(predictions,labels)),false_safe=sum(p and not y for p,y in zip(predictions,labels)),conservative=sum(not p and y for p,y in zip(predictions,labels)),success_recall=float(np.mean(pos)) if pos else None,failure_recall=float(np.mean(np.logical_not(neg))) if neg else None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.run
 if a.output.exists():raise FileExistsError(a.output)
 assert json.loads((root/'status.json').read_text())['phase']=='completed'
 manifest=json.loads((root/'manifest.json').read_text());cc=asdict(replace(load_config(CONFIG_PATH.parent/'fixed_endpoint_directional_alpha0.json').controller,fixed_roll_reference=None));prepared=[]
 # Read only pre-pulse snapshots and small probes before constructing every model energy.
 for c in contexts():
  for onset in ONSETS:
   snap=root/f"{c['route']}_v{c['speed']:g}_a{c['alpha']}_t{onset:g}_snapshot.pkl"
   with snap.open('rb') as f:s=pickle.load(f)
   x=snapshot_features(s,speed_request=c['speed'],onset_s=onset,controller=cc);lam=x['lambda_u'];bu=x['b_u'];datasets={force:np.load(root/(case_name(dict(**c,force_N=force),onset)+'.npz')) for force in [0,.5,-.5]}
   if any(len(d['time'])<=round(onset/.005) for d in datasets.values()):continue
   k=int(np.argmin(abs(datasets[0]['time']-onset)));assert abs(float(datasets[0]['time'][k])-onset)<1e-4
   def z(d):return x['a2']/lam*d['actual_delta'][k]+lam*d['phi'][k]+d['phi_dot'][k]
   impulse=float(np.linalg.norm(datasets[.5]['applied_force_impulse'][k])+np.linalg.norm(datasets[-.5]['applied_force_impulse'][k]))
   if not impulse>0:raise ValueError('no impulse in probe identification tick')
   gamma=float((z(datasets[.5])-z(datasets[-.5]))/impulse);K=gamma*(-math.expm1(-lam*manifest['duration_s']))/lam;W=bu*bu*(-math.expm1(-2*lam*x['remaining_s']))/(2*lam)
   if not math.isfinite(W) or W<=0:raise ValueError('invalid model Gramian')
   for force in FORCES:
    prepared.append(dict(**c,onset_s=onset,force_N=force,name=case_name(dict(**c,force_N=force),onset),gamma_identified=gamma,pulse_map_K=K,gramian_W=W,z_u=x['z_u'],energy_static=(K*force)**2/W,energy_dynamic=(x['z_u']+K*force)**2/W))
 a.output.mkdir(parents=True);(a.output/'model_energies_before_labels.json').write_text(json.dumps(prepared,indent=2))
 # Fit on calibration labels, then score the predeclared speed-transfer group.
 raw=json.loads((root/'results.json').read_text())['cases'];byname={r['name']:r for r in raw}
 for r in prepared:r['actual_success']=byname[r['name']]['success']
 cal=[r for r in prepared if r['data_role']=='calibration'];transfer=[r for r in prepared if r['data_role']=='speed_transfer'];summary={}
 for method in ['static','dynamic']:
  threshold=choose_threshold([r[f'energy_{method}'] for r in cal],[r['actual_success'] for r in cal]);summary[method]=dict(effort_threshold=threshold,calibration=score(cal,threshold,method),speed_transfer=score(transfer,threshold,method))
 (a.output/'summary.json').write_text(json.dumps(dict(methods=summary,rows=prepared,omitted_rows=len(raw)-len(prepared),scope='DEV model diagnostic; single global threshold per model; same-state amplitudes correlated, no statistical confidence claim'),indent=2))
 (a.output/'REPORT.md').write_text('# Scalar effort-model diagnostic\n\n'+json.dumps(summary,indent=2)+'\n\nStatic e=0 versus dynamic z_u, both with first-tick small-probe local force mapping. E=(z+KF)^2/W with R=1 and fixed-speed scalar unstable model. Threshold selected by balanced accuracy on 2.3m/s only, ties fewer false-safe then smaller threshold. 2.8m/s is DEV speed transfer; not final TEST. Unknown gamma is estimated, not asserted from rigid-body geometry. This mapping remains dependent on small physical probes and is not yet a cheap online input. No actuator peak feasibility enforced; effort is not joules. Omissions are incomplete first-probe states, not negative labels. Equilibrium recovery and full task recovery differ; failure here does not disprove the stated linear algebra.\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
