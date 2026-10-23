"""Fixed-k grouped transfer diagnostic on already-used TRAIN branches."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
RUNS=('preference_train_coordination_records_v1','tolerance_recovery_opportunity_v1')
KEYS=('route','speed','alpha','force_N','onset_s')
def load(root,name):
 run=root/'runs'/name;decl=json.loads((run/'declaration.json').read_text());rows=json.loads((run/'evaluation/results.json').read_text())['cases'];groups={};sources={}
 for row in rows:groups.setdefault(tuple(row[k] for k in KEYS),{})[row['policy']]=row
 result=[]
 for key,arms in groups.items():
  obs={}
  for pol in ('control','speed_off'):
   path=run/'evaluation'/(arms[pol]['name']+'_teaching.npz');sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
   with np.load(path) as d:
    i=np.flatnonzero(np.isclose(d['time'],decl['recovery_intervention']['start_s'],atol=1e-5));assert len(i)==1;obs[pol]=d['observations'][i[0]]
  np.testing.assert_array_equal(obs['control'],obs['speed_off'])
  result.append(dict(key=key,x=obs['control'],control=arms['control'],speed_off=arms['speed_off']))
 return result,sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data={};sources={}
 for name in RUNS:data[name],ss=load(a.root,name);sources.update(ss)
 folds=[]
 for train,test in (RUNS,RUNS[::-1]):
  aa=data[train];bb=data[test];X=np.array([r['x'] for r in aa],float);Z=np.array([r['x'] for r in bb],float);y=np.array([int(r['speed_off']['task_recovered'])-int(r['control']['task_recovered']) for r in aa])
  mean=X.mean(0);scale=X.std(0);scale=np.where(scale>1e-8,scale,1.);dist=np.mean((((Z-mean)/scale)[:,None,:]-(X-mean)/scale)**2,axis=-1);idx=np.argsort(dist,axis=1,kind='stable')[:,:8];dd=np.take_along_axis(dist,idx,axis=1);w=np.exp(-.5*(dd-dd.min(axis=1,keepdims=True)));pred=(w*y[idx]).sum(1)/w.sum(1);choose=pred>0
  cases=[]
  for r,v,ch in zip(bb,pred,choose):
   selected=r['speed_off'] if ch else r['control'];cases.append(dict(context=dict(zip(KEYS,r['key'])),score=float(v),choose_speed=bool(ch),control_success=r['control']['task_recovered'],speed_success=r['speed_off']['task_recovered'],selected_success=selected['task_recovered'],return_change=selected['logged_return']-r['control']['logged_return']))
  summary={}
  for label,rr in [('all',cases),('nonzero',[r for r in cases if r['context']['force_N']!=0])]:
   summary[label]=dict(n=len(rr),control=sum(r['control_success'] for r in rr),always_speed=sum(r['speed_success'] for r in rr),selected=sum(r['selected_success'] for r in rr),chosen=sum(r['choose_speed'] for r in rr),gains=sum(not r['control_success'] and r['selected_success'] for r in rr),losses=sum(r['control_success'] and not r['selected_success'] for r in rr),mean_return_change=float(np.mean([r['return_change'] for r in rr])))
  folds.append(dict(train=train,test=test,summary=summary,cases=cases))
 out=dict(scope='Posthoc TRAIN-only two-batch transfer; fixed k8, input351 pre-action obs, signed task gain target, threshold0; no force/case IDs/future as input. Different intervention windows confound transfer; not new closed-loop evaluation or final TEST.',folds=folds,sources=sources)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([f['summary'] for f in folds],indent=2))
if __name__=='__main__':main()
