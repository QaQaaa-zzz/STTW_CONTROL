"""Audit recorded online choices and paired outcomes, retaining every declared lane."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
m=json.loads((a.run/'manifest.json').read_text());cases=[]
for case in m['cases']:
    spec=json.loads(Path(case['config']).read_text());out=Path(spec['output'])
    if spec.get('batch_candidates'):
        for lane in json.loads((out/'batch_results.json').read_text())['cases']:
            cases.append(dict(case,config=str(Path(lane['output'])/'config.json'),method=m['lane_methods'][lane['lane']]))
    else:cases.append(case)
roots=list(dict.fromkeys(c['origin_id'] for c in cases));methods=list(dict.fromkeys(c['method'] for c in cases));rows=[];figures=[]
for root in roots:
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');end_x=0.;skills=[]
    for i,c in enumerate(c for c in cases if c['origin_id']==root):
        spec=json.loads(Path(c['config']).read_text());d=Path(spec['output']);skills+=spec['skills']
        summary=json.loads((d/'chain_summary.json').read_text());physical=json.loads((d/'results.json').read_text())
        z=np.load(d/'trajectory.npz');t=z['trace'][z['valid'][:,0],0];end_x=max(end_x,float(t[:,1].max()))
        line,=ax.plot(t[:,1],t[:,2],linestyle=['-','--',':'][i%3],label=f"{c['method']}: strict={int(summary['complete'])}")
        ax.plot(t[-1,1],t[-1,2],'o' if summary['complete'] else 'x',color=line.get_color())
        event=json.loads((d/'connection_decision.json').read_text()) if (d/'connection_decision.json').exists() else None
        error=None;prediction=None;actual=None
        if event:
            point=np.asarray(event['root'])[0,:2];ax.plot(*point,'s',color=line.get_color(),markersize=4)
        if event and 'selection' in event and (d/'handoff.json').exists():
            h=json.loads((d/'handoff.json').read_text());r=np.array(h['root'][0]);qx,qy,qz,qw=r[3:7]
            yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));sample=t[np.argmin(abs(t[:,0]-h['time']))]
            measured=np.array([r[1],yaw,np.cos(yaw)*r[7]+np.sin(yaw)*r[8],-np.sin(yaw)*r[7]+np.cos(yaw)*r[8],sample[5],sample[6]])
            chosen=event['candidate'];estimate=next((e for e in event['selection']['estimates'] if all(abs(e['candidate'][k]-chosen[k])<1e-6 for k in ('speed_m_s','alpha'))),None)
            # An unsupported current command may not belong to the candidate grid.
            if estimate is not None:
                prediction=np.array(estimate['predicted_entry']);actual=measured[:len(prediction)];error=prediction-actual;error[1]=np.arctan2(np.sin(error[1]),np.cos(error[1]))
                prediction=prediction.tolist();actual=actual.tolist();error=error.tolist()
        rows.append(dict(root=root,method=c['method'],complete=summary['complete'],checks=summary['checks'],stages=summary['stages'],decision=event,
                         contact_filtered_protocol=summary.get('contact_filtered_protocol'),physical_result=physical,
                         predicted_entry=prediction,actual_entry=actual,entry_error=error,trajectory=str(d/'trajectory.npz')))
    ref=np.load(d/'ground_route.npz')['xy'];ax.plot(*ref[ref[:,0]<=max(end_x,spec['completion']['final_goal_xy'][0])+2].T,'k:',label='Original reference')
    ax.axvspan(spec['platform_front_x'],spec['platform_back_x'],color='gray',alpha=.1,label='Platform X range')
    ax.set_aspect('equal',adjustable='box');ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=root+'\nDEV; skills: '+', '.join(dict.fromkeys(skills)));fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=8)
    name=root+'_online_xy.png';fig.savefig(a.run/name,dpi=140);plt.close(fig);figures.append(name)
scores={}
for method in methods:
    subset=[r for r in rows if r['method']==method];pairs=[(r,next(b for b in rows if b['root']==r['root'] and b['method']=='fixed')) for r in subset]
    errors=[r['entry_error'] for r in subset if r['entry_error'] is not None]
    filtered=[r for r in subset if r['contact_filtered_protocol'] is not None]
    scores[method]=dict(complete=sum(r['complete'] for r in subset),denominator=len(subset),
        reached=sum(r['decision'] is not None for r in subset),new_successes=[r['root'] for r,b in pairs if r['complete'] and not b['complete']],
        lost_successes=[r['root'] for r,b in pairs if b['complete'] and not r['complete']],
        contact_filtered_complete=sum(r['contact_filtered_protocol']['complete'] for r in filtered),contact_filtered_denominator=len(filtered),
        endpoint_rmse=np.sqrt(np.mean(np.square(errors),axis=0)).tolist() if errors else None,endpoint_error_sample_count=len(errors))
result=dict(role=m.get('role','DEV engineering validation; not independent TEST'),scores=scores,rows=rows,figures=figures,
            endpoint_units=['m','rad','m/s','m/s','rad','rad/s'],decision='Integration evidence only; no automatic adoption or robustness claim')
(a.run/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Online endpoint planning DEV','',*[f'![{r}]({f})' for r,f in zip(roots,figures)],'',result['decision'],'','| Method | Strict complete | Strict added / lost vs fixed | Contact-filter complete |','|---|---:|---:|---:|']
for name,s in scores.items():lines.append(f"| {name} | {s['complete']}/{s['denominator']} | {len(s['new_successes'])} / {len(s['lost_successes'])} | {s['contact_filtered_complete']}/{s['contact_filtered_denominator']} |")
lines+=['',f'All {len(roots)} declared origins and {len(rows)} trajectories retained, including failures before the decision. Square markers show actual decisions; crosses show observed endpoints.',
        'Endpoint errors cover only executed actions that reached release and had a corresponding prediction; other candidates are not validated. Original strict and declared contact-filter labels remain separate.',
        'Actual choices, stage results, predictions and raw physical records: comparison.json. No LLM was used.','','| Origin | Method | Actual exit command (m/s, alpha) | Selection status | Ground return | Continued |','|---|---|---|---|---|---|']
for row in rows:
    e=row['decision'];command=str(e['candidate']) if e else 'Not reached';status=e.get('selection',{}).get('status','fixed') if e else 'Not reached'
    lines.append(f"| {row['root']} | {row['method']} | {command} | {status} | {row['stages']['ground_return']} | {row['stages']['continued']} |")
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(scores))
