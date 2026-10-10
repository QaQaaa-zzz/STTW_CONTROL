#!/usr/bin/env python3
"""Stage A: only saved NPZ, fixed nominal windows, no simulator imports."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.path_frequency import signal_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((source/'manifest.json').read_text());upper=Path(manifest['upper_run']);results={};spectra={}
 # Equal-length 4s nominal plateaus: whole straight and post-turn primary/extension.
 windows={'straight_hold':[(1.,5.),(6.,10.)],'fast_turn':[(6.,10.),(12.,16.)]}
 for case in windows:
  for alpha in (0,1):
   traces={'local300':dict(np.load(source/'data'/case/f'alpha{alpha}.npz')),'R196':dict(np.load(upper/'evaluationbest'/case/f'alpha{alpha}.npz'))}
   fig,axes=plt.subplots(5,2,figsize=(14,16))
   for name,d in traces.items():
    raw=d['limited_command'].astype(float);gov=d['governed'].astype(float);actual=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)
    np.testing.assert_allclose(actual-raw,(gov-raw)+(actual-gov),atol=1e-12,rtol=0)
    z=np.tanh(d['latent_z']);target=np.column_stack([z[:,0]*np.where(z[:,0]>=0,.25,1.),z[:,1]*.2])
    signals=dict(nominal_speed=raw[:,0],nominal_steer=raw[:,1],target_speed_offset=target[:,0],target_steer_offset=target[:,1],requested_steer=raw[:,1]+target[:,1],offset_speed=d['offsets'][:,0],offset_steer=d['offsets'][:,1],governed_speed=gov[:,0],governed_steer=gov[:,1],actual_speed=actual[:,0],actual_steer=actual[:,1],actual_steer_rate=d['actual_delta_rate'])
    for key in ['u_nom','applied_residual','requested_residual','final_command']:
     for ch in range(2):signals[key+f'_{ch}']=d[key][:,ch]
    # Last physical tick of each 4-tick policy interval, no averaging/smoothing.
    ix=np.arange(3,len(raw),4);t=d['time'][ix]+.005
    row={};clip={}
    for key,mask in [('residual',abs(d['requested_residual'])>np.array([1.5,10.])),('final',d['final_command']!=d['u_prelimit'])]:
     clip[key]={'full_200Hz_count_by_channel':mask.sum(0).tolist(),'endpoint_50Hz_count_by_channel':mask[ix].sum(0).tolist(),'full_steps':len(raw),'endpoint_steps':len(ix)}
    # No fabrication from any-channel flags: preserve only aggregate if channel flag absent.
    clip['reference_any_count_200Hz']=int(d['reference_reference_clipped'].sum())
    clip['reference_per_channel']='UNKNOWN: per-channel preclip target not logged'
    if 'actuator_force_limit_substeps' in d:clip['actuator_force_limit_substeps']=d['actuator_force_limit_substeps'].sum(0).tolist()
    for lo,hi in windows[case]:
     mask=(t>lo+1e-6)&(t<=hi+1e-6);win={};assert mask.sum()==200
     assert np.max(abs(d['raw_rates'][ix][mask]))<1e-5,'window must depend only on external nominal'
     for key,values in signals.items():
      m,f,power=signal_metrics(values[ix][mask]);win[key]=m
      spectra[f'{case}_{alpha}_{name}_{lo}_{key}_psd']=power
     spectra[f'{case}_{alpha}_{name}_{lo}_frequency']=f
     row[f'{lo:g}-{hi:g}']=win
    final=d['time']>=9.5-1e-6 if case=='straight_hold' else d['time']>=15.5-1e-6
    row['final_bias']={k:float(v[final,0].mean()) for k,v in [('task',actual-raw),('upper',gov-raw),('lower',actual-gov)]}
    row['clip']=clip;results[f'{case}/alpha{alpha}/{name}']=row
    for ch in range(2):
     for value,label in [(raw,'nominal'),(raw+target,'requested'),(gov,'governed'),(actual,'actual')]:axes[0,ch].plot(t,value[ix,ch],label=name+' '+label,lw=.7)
     for value,label in [(target,'target'),(d['offsets'],'executed')]:axes[1,ch].plot(t,value[ix,ch],label=name+' '+label,lw=.7)
     axes[2,ch].plot(t,(actual-gov)[ix,ch],label=name+' lower error',lw=.7)
     axes[3,ch].plot(t,d['final_command'][ix,ch],label=name+' final',lw=.7)
    for name_signal,ax in [('requested_steer',axes[4,0]),('actual_steer',axes[4,1])]:
     for lo,hi in windows[case]:
      mask=(t>lo+1e-6)&(t<=hi+1e-6);m,f,power=signal_metrics(signals[name_signal][ix][mask]);ax.semilogy(f,power+1e-30,label=f'{name} {name_signal} {lo}-{hi}s')
   af,aa=plt.subplots(4,2,figsize=(14,12))
   for method,trace in traces.items():
    ix=np.arange(3,len(trace['time']),4);tm=trace['time'][ix]+.005
    for ch in range(2):
     for key,row in [('u_nom',0),('applied_residual',1)]:aa[row,ch].plot(tm,trace[key][ix,ch],label=method+' '+key,lw=.7)
     aa[2,ch].step(tm,(trace['final_command'][ix,ch]!=trace['u_prelimit'][ix,ch]).astype(int),label=method+' final clip',lw=.7)
    aa[3,0].plot(tm,trace['actual_delta_rate'][ix],label=method+' actual steer rate',lw=.7)
    aa[3,1].plot(tm,np.diff(trace['governed'][ix,1],prepend=trace['governed'][ix[0],1])/.02,label=method+' governed endpoint derivative',lw=.7)
   for row in range(4):
    for ch in range(2):
     ax=aa[row,ch];ax.legend(fontsize=7);ax.grid(alpha=.25);ax.set_xlabel('Time (s)');ax.set_ylabel('clip indicator' if row==2 else 'rad/s')
   af.suptitle(f'{case} alpha{alpha} | nominal ECBC / lower residual / final per-channel clip / measured versus reference rates |50Hz');af.tight_layout();af.savefig(out/f'{case}_alpha{alpha}_actuation.png',dpi=120);plt.close(af)
   for r in range(5):
    for c in range(2):
     ax=axes[r,c];ax.legend(fontsize=7);ax.grid(alpha=.25);ax.set_xlabel('Hz' if r==4 else 'Time (s)')
     ax.set_ylabel((['speed m/s','steer rad'] if r<3 else ['front rad/s','rear rad/s'])[c] if r<4 else 'PSD')
   fig.suptitle(f'{case} alpha{alpha} | same frozen upper best | seed77001 | 50Hz endpoints, no smoothing');fig.tight_layout();fig.savefig(out/f'{case}_alpha{alpha}_signals.png',dpi=120);plt.close(fig)
 (out/'metrics.json').write_text(json.dumps(dict(timing_hz=dict(physics=5000,lower=200,upper=50),windows=windows,endpoint='tick3,7,...; post-state time=tick_time+.005; no anti-alias filter; PSD is discrete 50Hz signal, cannot infer >25Hz physics',reversal_definition='raw adjacent rate sign changes; qualified adjacent direction runs rate>.05rad/s and both excursions>=.002rad; counts are events, not oscillation frequency',results=results),indent=2));np.savez_compressed(out/'spectra.npz',**spectra)
 print(json.dumps({k:{w:{s:v[s] for s in ['governed_steer','actual_steer']} for w,v in r.items() if '-' in w} for k,r in results.items()},indent=2))
if __name__=='__main__':main()
