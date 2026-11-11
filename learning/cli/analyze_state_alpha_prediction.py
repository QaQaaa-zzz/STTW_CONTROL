"""Factual held-out prediction from each lane's own pre-switch state, not selector success."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;spec=json.loads((r/'protocol.json').read_text());rows=json.loads((r/'analysis_summary.json').read_text())['rows'];samples=[]
for job in json.loads((r/'jobs.json').read_text()):
 z=np.load(r/job['name']/'trajectory.npz');o=z['observation'];ticks=z['switch_tick']
 for lane,label in enumerate(spec['case_labels'][:25]):
  reached=ticks[:,lane][ticks[:,lane]>=0]
  if not len(reached):continue
  i=max(0,int(reached[0])-1);f=np.r_[o[i,lane,300:305],o[i,lane,337:340],o[i,lane,346:350],o[i,lane,345]]
  m=next(x for x in rows if x['case']==job['name'] and x['label']==label)
  samples.append(dict(case=job['name'],terrain=m['terrain'],seed=job['physical_seed'],target=float(label.split('_to_')[1]),features=f,metrics=m))
predictions=[]
for s in samples:
 train=[t for t in samples if t['seed']!=s['seed'] and t['terrain']==s['terrain'] and t['target']==s['target']]
 nearest=sorted(train,key=lambda t:float(np.sum((t['features']-s['features'])**2)))[:3]
 for outcome in ['full_window','A','B','C']:
  def truth(t):return not t['metrics']['failed'] if outcome=='full_window' else t['metrics']['tasks'][outcome]
  predictions.append(dict(case=s['case'],label=s['metrics']['label'],outcome=outcome,truth=bool(truth(s)),state_probability=float(np.mean([truth(t) for t in nearest])),context_probability=float(np.mean([truth(t) for t in train]))))
summary={}
for outcome in ['full_window','A','B','C']:
 g=[x for x in predictions if x['outcome']==outcome];summary[outcome]=dict(n=len(g),positives=sum(x['truth'] for x in g),state_brier=float(np.mean([(x['state_probability']-x['truth'])**2 for x in g])),context_brier=float(np.mean([(x['context_probability']-x['truth'])**2 for x in g])))
(r/'factual_prediction.json').write_text(json.dumps(dict(role='exploratory physical-group-held-out DEV; own pre-switch state predicts observed action, not counterfactual selection',summary=summary,predictions=predictions),indent=2));print(json.dumps(summary,indent=2))
