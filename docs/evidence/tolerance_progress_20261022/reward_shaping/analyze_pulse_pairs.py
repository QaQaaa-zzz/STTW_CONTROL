import json
from pathlib import Path
import numpy as np
r=Path('/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/tolerance_shaping_post_v1/evaluation')
rows=json.loads((r/'results.json').read_text())['cases'];lookup={(x['policy'],x['alpha'],x['route'],x['speed'],x['onset_s'],x['force_N']):x for x in rows};out=[]
for x in rows:
 if x['force_N']==0:continue
 key=(x['policy'],x['alpha'],x['route'],x['speed'],x['onset_s'],0);nom=lookup[key]
 with np.load(r/(x['name']+'.npz')) as d,np.load(r/(nom['name']+'.npz')) as z:
  n=min(len(d['time']),len(z['time']));t=d['time'][:n];assert np.allclose(t,z['time'][:n],atol=1e-6)
  mask=(t>=x['onset_s']-1e-5)&(t<x['onset_s']+2.-1e-5)
  item=dict(policy=x['policy'],alpha=x['alpha'],route=x['route'],speed=x['speed'],onset_s=x['onset_s'],force_N=x['force_N'],window_complete=int(mask.sum())==400,window_samples=int(mask.sum()),success=x['success'],nominal_success=nom['success'],prefix_valid=x['prefix_valid'])
  for field in ['phi','path_cross_track','actual_forward_speed']:
   delta=d[field][:n][mask]-z[field][:n][mask];item[field+'_delta_rms']=float(np.sqrt(np.mean(delta**2))) if len(delta) else None;item[field+'_delta_peak']=float(np.max(abs(delta))) if len(delta) else None
  out.append(item)
def context(x):return (x['alpha'],x['route'],x['speed'],x['onset_s'],x['force_N'])
policies=['baseline_update50','static_update50','dynamic_update50'];complete=[{context(x) for x in out if x['policy']==p and x['window_complete'] and x['prefix_valid']} for p in policies];shared=set.intersection(*complete)
summary=[]
for a in [0,1]:
 for p in policies:
  allrows=[x for x in out if x['alpha']==a and x['policy']==p];g=[x for x in allrows if context(x) in shared];s=dict(alpha=a,policy=p,total=len(allrows),incomplete=sum(not x['window_complete'] for x in allrows),invalid_prefix=sum(not x['prefix_valid'] for x in allrows),shared_n=len(g))
  for f in ['phi','path_cross_track','actual_forward_speed']:
   s[f+'_delta_rms_mean']=float(np.mean([x[f+'_delta_rms'] for x in g]));s[f+'_delta_peak_p95']=float(np.quantile([x[f+'_delta_peak'] for x in g],.95))
  summary.append(s)
result=dict(protocol='posthoc paired same-time response minus own nominal; onset to onset+2s; shared complete valid-prefix contexts across baseline/static/dynamic; excluded counts retained; not a total-policy success estimate',cases=out,summary=summary)
(r/'report/pulse_pair_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
