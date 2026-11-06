"""Fit an exploratory integral roll model from saved pulse responses; no physics."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pair',type=int,nargs=2,action='append',default=[]);a=p.parse_args()
d=np.load(a.run/'trajectory.npz');trace=d['trace'];X=[];Y=[]
if a.pair:
 fits=[]
 dt=float(trace[1,0,0]-trace[0,0,0])
 for seconds in (.05,.1,.15,.2):
  horizon=round(seconds/dt);X=[];Y=[]
  for lo,hi in a.pair:
   for j in range(len(trace)-horizon):
    w=trace[j:j+horizon+1,hi];b=trace[j:j+horizon+1,lo]
    features=np.column_stack((w[:,4]*w[:,8]-b[:,4]*b[:,8],w[:,4]**2*w[:,7]-b[:,4]**2*b[:,7],w[:,5]-b[:,5]))
    X.append(np.trapezoid(features,dx=dt,axis=0));Y.append(w[-1,6]-w[0,6]-b[-1,6]+b[0,6])
  X=np.asarray(X);Y=np.asarray(Y);c=np.linalg.lstsq(X,Y,rcond=None)[0]
  fits.append(dict(integration_s=seconds,coefficients=c.tolist(),r_squared=float(1-np.sum((X@c-Y)**2)/np.sum((Y-Y.mean())**2)),samples=len(Y)))
 a.output.write_text(json.dumps(dict(source=str(a.run),pairs=a.pair,fits=fits,role='paired integral exploratory fit; overlapping windows, not independent validation'),indent=2)+'\n')
 raise SystemExit
# 150 ms integrals reduce contact impulses. Fit only the declared early pulse window.
for i in range(trace.shape[1]):
 z=trace[:,i]
 for j in range(30,min(100,len(z)-30)):
  w=z[j:j+31];v=w[:,4]
  X.append(np.trapezoid(np.column_stack((v*w[:,8],v*v*w[:,7],w[:,5],np.ones(len(w)))),dx=.005,axis=0))
  Y.append(w[-1,6]-w[0,6])
X=np.asarray(X);Y=np.asarray(Y);c=np.linalg.lstsq(X,Y,rcond=None)[0]
a.output.write_text(json.dumps(dict(source=str(a.run),coefficients=c.tolist(),r_squared=float(1-np.sum((X@c-Y)**2)/np.sum((Y-Y.mean())**2)),samples=len(Y),window_start_s=[.15,.495],integration_s=.15,role='exploratory overlapping-window fit, not independent validation; constant bias excluded from controller'),indent=2)+'\n')
