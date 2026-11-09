"""Audit actual online selections, paired gains/losses and endpoint errors."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
m=json.loads((a.run/'manifest.json').read_text());roots=list(dict.fromkeys(c['origin_id'] for c in m['cases']));rows=[];figures=[]
for root in roots:
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for i,c in enumerate(c for c in m['cases'] if c['origin_id']==root):
        d=Path(json.loads(Path(c['config']).read_text())['output']);summary=json.loads((d/'chain_summary.json').read_text())
        z=np.load(d/'trajectory.npz');t=z['trace'][z['valid'][:,0],0]
        ax.plot(t[:,1],t[:,2],color=['#0072B2','#D55E00','#009E73'][i],linestyle=['-','--',':'][i],label=f"{c['method']}: complete={int(summary['complete'])}")
        ax.plot(t[-1,1],t[-1,2],'o' if summary['complete'] else 'x',color=ax.lines[-1].get_color())
        event=json.loads((d/'connection_decision.json').read_text()) if (d/'connection_decision.json').exists() else None
        error=None;prediction=None;actual=None
        if event and 'selection' in event and (d/'handoff.json').exists():
            h=json.loads((d/'handoff.json').read_text());r=np.array(h['root'][0]);qx,qy,qz,qw=r[3:7]
            yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));sample=t[np.argmin(abs(t[:,0]-h['time']))]
            actual=np.array([r[1],yaw,np.cos(yaw)*r[7]+np.sin(yaw)*r[8],-np.sin(yaw)*r[7]+np.cos(yaw)*r[8],sample[5],sample[6]])
            chosen=event['candidate'];estimate=next(e for e in event['selection']['estimates'] if all(abs(e['candidate'][k]-chosen[k])<1e-6 for k in ('speed_m_s','alpha')))
            prediction=np.array(estimate['predicted_entry']);actual=actual[:len(prediction)];error=prediction-actual;error[1]=np.arctan2(np.sin(error[1]),np.cos(error[1]))
            prediction=prediction.tolist();actual=actual.tolist();error=error.tolist()
        rows.append(dict(root=root,method=c['method'],complete=summary['complete'],checks=summary['checks'],stages=summary['stages'],decision=event,
                         predicted_entry=prediction,actual_entry=actual,entry_error=error,trajectory=str(d/'trajectory.npz')))
    ref=np.load(d/'ground_route.npz')['xy'];ax.plot(*ref.T,'k:',label='Original reference');ax.axvline(-2,color='gray',linewidth=.6)
    ax.set_aspect('equal',adjustable='box');ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=root+'\nValidation DEV, frozen update2; shared upstream alpha=.5');ax.legend(fontsize=8)
    name=root+'_online_xy.png';fig.savefig(a.run/name,dpi=140);plt.close(fig);figures.append(name)
scores={}
for method in ('fixed','geometry','dynamics'):
    subset=[r for r in rows if r['method']==method];pairs=[(r,next(b for b in rows if b['root']==r['root'] and b['method']=='fixed')) for r in subset]
    errors=[r['entry_error'] for r in subset if r['entry_error'] is not None]
    scores[method]=dict(complete=sum(r['complete'] for r in subset),denominator=len(roots),
        reached=sum(r['decision'] is not None for r in subset),new_successes=[r['root'] for r,b in pairs if r['complete'] and not b['complete']],
        lost_successes=[r['root'] for r,b in pairs if b['complete'] and not r['complete']],
        endpoint_rmse=np.sqrt(np.mean(np.square(errors),axis=0)).tolist() if errors else None,
        endpoint_error_sample_count=len(errors))
result=dict(role='validation_tuning DEV; not independent TEST',scores=scores,rows=rows,figures=figures,
            endpoint_units=['m','rad','m/s','m/s','rad','rad/s'],decision='Neither endpoint planner adopted as an improvement; fixed baseline retained')
(a.run/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Online endpoint planning DEV','',result['decision'],'','| Method | Complete | Added / lost versus fixed |','|---|---:|---:|']
for name,s in scores.items():lines.append(f"| {name} | {s['complete']}/{s['denominator']} | {len(s['new_successes'])} / {len(s['lost_successes'])} |")
lines+=['','All 12 roots retained, including three upstream failures. Prefix decisions are online; outcome labels are post-run physical adjudication.',
        'Endpoint errors are scored only for actually selected actions that reached the release line; unexecuted candidates are not validated.',
        'Metrics and actual predictions: comparison.json. No LLM was used in these 36 physical runs.','',*[f'![{r}]({f})' for r,f in zip(roots,figures)]]
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(scores))
