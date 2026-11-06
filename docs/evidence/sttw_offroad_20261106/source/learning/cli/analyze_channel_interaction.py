"""Measure nonadditive channel responses from an existing paired hold experiment."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
a=p.parse_args()
protocol=json.loads((a.run/'protocol.json').read_text())
labels=['natural_continuation','restored_control','hold_speed_correction','hold_steer_correction','hold_both_corrections']
if protocol['case_labels']!=labels:raise ValueError('Requires the five paired hold branches')
rows=[]
for case in json.loads((a.run/'status.json').read_text())['cases']:
 run=Path(case['run']);fork=json.loads((run/'fork.json').read_text())
 t0=fork['policy_step']*.02;hold=fork['protocol']['hold_s']
 with np.load(run/'trajectory.npz') as raw:
  trace=raw['lower_trace'].reshape(-1,5,11).astype(float)
 time=np.arange(len(trace))*.005+.005
 for window,left,right in [('hold',t0,t0+hold),('after_release_0p5',t0+hold,t0+hold+.5)]:
  mask=(time>left)&(time<=right+1e-8)&trace[:,:,6].astype(bool).all(1)&(trace[:,:,7]>=25).all(1)
  z=trace[mask,:,:4]
  interaction=z[:,4]-z[:,2]-z[:,3]+z[:,1]
  reload=z[:,0]-z[:,1]
  row=dict(phase=case['phase'],alpha=case['alpha'],window=window,n_ticks=int(mask.sum()),run=str(run))
  for name,indices in [('xy_m',[0,1]),('speed_m_s',[2]),('signed_path_m',[3])]:
   rms=lambda x:float(np.sqrt(np.mean(np.sum(x[:,indices]**2,axis=1))))
   row[name]=dict(interaction_rms=rms(interaction),reload_rms=rms(reload),speed_hold_effect_rms=rms(z[:,2]-z[:,1]),steer_hold_effect_rms=rms(z[:,3]-z[:,1]),both_hold_effect_rms=rms(z[:,4]-z[:,1]))
  rows.append(row)
out=a.run/'analysis'
(out/'channel_interaction.json').write_text(json.dumps(dict(formula='I(t)=both(t)-speed_hold(t)-steer_hold(t)+restored(t)',scope='Same-state finite intervention nonadditivity of signed trajectories, not an RMSE-cost interaction or an identified mechanical cross-term. One state per phase/alpha; reload contrast is descriptive, not a noise distribution.',rows=rows),indent=2))
lines=['# Paired channel interaction from saved trajectories','','[Actual/reference XY and reward panels](RESULTS.md) · [Full metrics](channel_interaction.json)','','I(t)=both(t)-speed_hold(t)-steer_hold(t)+restored(t). Zero would mean the two isolated finite responses add exactly. Compute this on signed raw XY/speed/path signals before RMS, not on separate RMSE costs. All five branches must have a full active physics tick.','', 'The hold lasts0.2s; release window is the following0.5s. Feedback, state evolution and reference processing all contribute. Nonzero interaction does not identify a unique vehicle coupling coefficient or establish coordination benefit. Natural/restored is a paired reload reference, not a confidence bound.','','|Phase|Alpha|Window|XY interaction RMS m|Reload XY RMS m|Speed interaction RMS m/s|Signed path interaction RMS m|','|---|---:|---|---:|---:|---:|---:|']
for r in rows:
 lines.append(f"|{r['phase']}|{r['alpha']}|{r['window']}|{r['xy_m']['interaction_rms']:.6f}|{r['xy_m']['reload_rms']:.6f}|{r['speed_m_s']['interaction_rms']:.6f}|{r['signed_path_m']['interaction_rms']:.6f}|")
(out/'CHANNEL_INTERACTION.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
