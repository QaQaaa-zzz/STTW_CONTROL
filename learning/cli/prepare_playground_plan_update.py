"""Build a small smooth suffix change on the existing common progress grid."""
import argparse
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--amplitude',type=float,default=.05)
p.add_argument('--start-x',type=float,default=12.6)
p.add_argument('--end-x',type=float,default=16.)
p.add_argument('--preserve-through-s',type=float,default=10.5)
a=p.parse_args()
bank=dict(np.load(a.source));s=bank['s'];xy=bank['xy'].copy()
for lane in range(len(xy)):
    t=np.clip((xy[lane,:,0]-a.start_x)/(a.end_x-a.start_x),0,1)
    xy[lane,:,1]+=a.amplitude*np.sin(np.pi*t)**4
    anchor=np.searchsorted(bank['xy'][lane,:,0],a.start_x-.6)
    arc=np.r_[s[anchor],s[anchor]+np.cumsum(np.linalg.norm(np.diff(xy[lane,anchor:],axis=0),axis=1))]
    tangent=xy[lane,-1]-xy[lane,-2];tangent/=np.linalg.norm(tangent)
    endpoint=xy[lane,-1].copy()
    xy[lane,anchor:]=np.column_stack([np.interp(s[anchor:],arc,xy[lane,anchor:,j]) for j in (0,1)])
    # A shorter modified curve needs a forward tail, not repeated clamped points.
    tail=s>arc[-1]
    xy[lane,tail]=endpoint+(s[tail]-arc[-1])[:,None]*tangent
heading=np.unwrap(np.arctan2(np.gradient(xy[...,1],s,axis=1),np.gradient(xy[...,0],s,axis=1)),axis=1)
curvature=np.gradient(heading,s,axis=1)
for key,value in [('xy',xy),('heading',heading),('curvature',curvature)]:
    value[:,s<=a.preserve_through_s]=bank[key][:,s<=a.preserve_through_s]
np.savez_compressed(a.output,s=s,xy=xy,heading=heading,curvature=curvature)
