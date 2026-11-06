"""Fit an exploratory integral roll model from saved pulse responses; no physics."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=np.load(a.run/'trajectory.npz');trace=d['trace'];X=[];Y=[]
# 150 ms integrals reduce contact impulses. Fit only the declared early pulse window.
for i in range(trace.shape[1]):
 z=trace[:,i]
 for j in range(30,min(100,len(z)-30)):
  w=z[j:j+31];v=w[:,4]
  X.append(np.trapezoid(np.column_stack((v*w[:,8],v*v*w[:,7],w[:,5],np.ones(len(w)))),dx=.005,axis=0))
  Y.append(w[-1,6]-w[0,6])
X=np.asarray(X);Y=np.asarray(Y);c=np.linalg.lstsq(X,Y,rcond=None)[0]
a.output.write_text(json.dumps(dict(source=str(a.run),coefficients=c.tolist(),r_squared=float(1-np.sum((X@c-Y)**2)/np.sum((Y-Y.mean())**2)),samples=len(Y),window_start_s=[.15,.495],integration_s=.15,role='exploratory overlapping-window fit, not independent validation; constant bias excluded from controller'),indent=2)+'\n')
