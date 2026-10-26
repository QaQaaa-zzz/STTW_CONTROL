"""All120 nominal XY traces, unchanged world frame, in12 matched panels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


def render(root,analysis_name):
    root=Path(root).resolve(); analysis=root/analysis_name
    if json.loads((analysis/'status.json').read_text())['phase']!='completed':raise ValueError('completed analysis required')
    rows=json.loads((analysis/'results.json').read_text()); plan=json.loads((root/'plan.json').read_text())
    if len(rows)!=120:raise ValueError('all120 rows required')
    roles=['goal_zero_feedback','lineage_dual','lineage_u325','lineage_h75','goal_full_feedback']
    colors=['#777777','#D55E00','#CC79A7','#009E73','#0072B2']
    labels=['Zero neural','Dual u200','Distilled u325','H75','Current free_GAE64']
    contexts=list(plan['contexts']); fig,axes=plt.subplots(4,3,figsize=(17,15),layout='constrained')
    audit=[]
    for ax,context in zip(axes.flat,contexts):
        rr=[r for r in rows if r['context']==context]
        if len(rr)!=10:raise ValueError('five families x two alpha required per context')
        with np.load(plan['contexts'][context]['path']) as p:
            s=p['s']; xy=p['xy']; goal=float(p['goal']);point=np.array([np.interp(goal,s,xy[:,i]) for i in [0,1]])
            ax.plot(xy[s<=goal,0],xy[s<=goal,1],color='black',lw=1,ls=':',zorder=0)
        ax.add_patch(Circle(point,.5,fill=False,ec='black',lw=.75));ax.scatter(*point,marker='*',c='black',s=27,zorder=5)
        for role,color in zip(roles,colors):
            for alpha,style in [(0.,'-'),(1.,'--')]:
                row=next(r for r in rr if r['role']==role and r['alpha']==alpha)
                with np.load(row['raw_file']) as raw:
                    active=raw['active_tick'].astype(bool);actual=raw['actual_xy'][active]
                if not len(actual):raise ValueError('empty physical trace')
                ax.plot(actual[:,0],actual[:,1],color=color,ls=style,lw=1.05,alpha=.9)
                ax.scatter(*actual[-1],color=color,s=12,marker='x' if row['physical_outcome']=='contact' else ('o' if alpha==0 else 's'))
                audit.append(dict(case_id=row['case_id'],raw_file=row['raw_file'],raw_sha256=row['raw_sha256'],plotted_active_samples=len(actual),alpha=alpha,physical_outcome=row['physical_outcome']))
        prefix,geometry=context.split('__');ax.set_title(prefix+'\n'+geometry,fontsize=8)
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.set_xlabel('World X (m)',fontsize=8);ax.set_ylabel('World Y (m)',fontsize=8);ax.tick_params(labelsize=7)
    handles=[Line2D([],[],color=c,label=l) for c,l in zip(colors,labels)]
    handles += [Line2D([],[],color='black',ls='-',marker='o',label='Executed alpha0'),Line2D([],[],color='black',ls='--',marker='s',label='Executed alpha1'),Line2D([],[],color='black',ls=':',marker='*',label='Original path / target (0.5m disk)')]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Frozen middle-policy lineage: all12 nominal contexts /120 actual trajectories\nShared current interface; related DEV weight transfer, not native historical reproduction',fontsize=13)
    out=analysis/'lineage_nominal_review'; out.mkdir(exist_ok=True)
    path=out/'all_xy_lineage_overview.png';fig.savefig(path,dpi=160);plt.close(fig)
    (out/'xy_figure_provenance.json').write_text(json.dumps(dict(cases=audit,contexts=12,trajectories=120,transform='None: world XY, all active samples, no smoothing or selection; each panel equal metre aspect. Original reference ends at original goal; actual traces stop at logged physical endpoint.',style='family=color; alpha=line style plus endpoint circle/square; contact endpoint=x',destination='Provisional research review; exact Transactions journal figure requirements pending, no compliance claim.',alt_text='Twelve panels compare the original world path and target with both alpha endpoints from zero neural control, dual u200, u325, H75 and current free_GAE64. All120 traces survive; separate counts and actual goal distances are in review.json.'),indent=2)+'\n')
    print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--analysis-name',default='analysis');a=p.parse_args();render(a.run,a.analysis_name)
