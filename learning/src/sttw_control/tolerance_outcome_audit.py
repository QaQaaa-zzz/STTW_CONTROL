"""Descriptive causal windows; never replaces original episode labels."""
import numpy as np

def split_outcome(data,*,onset,limit,final_hold,goal,dt=.005):
 t=np.asarray(data['time'],float);roll=np.asarray(data['peak_roll'],float)
 if not np.isfinite(t).all() or not np.isfinite(roll).all():raise ValueError('nonfinite trajectory')
 if t.shape!=roll.shape or t.ndim!=1:raise ValueError('trajectory shape mismatch')
 # Logs use pre-tick time and post-tick outcome. Pulse starts on a tick boundary.
 before=t+dt<=onset+1e-6;after=t>=onset-1e-6
 if np.any(before&after) or not np.all(before|after):raise ValueError('onset must align with tick boundary')
 out={}
 for prefix,mask in [('prefix',before),('suffix',after)]:
  out[prefix+'_ticks']=int(mask.sum())
  for name,key in [('working_failure','peak_roll'),('physical_failure','physical_failure'),('domain_failure','domain_exit')]:
   values=np.asarray(data[key]);failed=values>limit if key=='peak_roll' else values.astype(bool)
   out[prefix+'_'+name]=bool(np.any(failed[mask])) if mask.any() else None
 out['suffix_task_success']=bool(final_hold and goal and not any(out['suffix_'+k] for k in ['working_failure','physical_failure','domain_failure'])) if after.any() else None
 out['reconstructed_full_success']=bool(final_hold and goal and not any(v for k,v in out.items() if k.endswith('_failure')))
 return out
