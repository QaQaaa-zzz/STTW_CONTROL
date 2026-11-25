"""Read the frozen online-brake matrix after completion and render every matched group."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_matched_controls import read_case, check_pair
from scipy.stats import binomtest


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_history(reference, candidate, cutoff):
    """Compare the observed lower-rate records through the pre-action boundary."""
    x, t = reference
    y, u = candidate
    nx = np.searchsorted(t, cutoff + 1e-7, side='right')
    ny = np.searchsorted(u, cutoff + 1e-7, side='right')
    n = min(nx, ny)
    same_clock = nx == ny and np.array_equal(t[:n], u[:n])
    # Pose, speed, path error, peak lean, valid/stop tick, delivered references, wheel speed.
    fields = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    return dict(reference_samples=int(nx), candidate_samples=int(ny), clock_equal=same_clock,
                observed_history_equal=bool(n and same_clock and np.array_equal(x[:n, fields], y[:n, fields])),
                max_shared_xy_difference_m=float(np.linalg.norm(x[:n, :2]-y[:n, :2], axis=1).max()) if n else np.nan)


def analyze(root):
    if json.loads((root/'status.json').read_text())['stage']!='physics_complete':
        raise ValueError('This analysis requires all declared physical results')
    with (root/'episodes.csv').open() as stream:
        cells=defaultdict(list)
        for r in csv.DictReader(stream):
            cells[r['condition'],int(r['group'])].append(r)
    out=root/'analysis'
    figures=out/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    rows, pairs, links, history_checks=[],[],[],[]
    for (condition,group),cell in cells.items():
        anchor=None
        outcomes, histories={},{}
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        xy,rew,cum,speed=axes.flat
        for r in cell:
            raw=Path(r['raw'])
            c,[(summary,trace,t,progress,_)]=read_case(raw,[r['method']])
            if anchor is None:
                anchor=c
                with np.load(Path(c['bundle'])/'route.npz') as z:
                    arc=z['s']; route=z['xy']
            else:
                check_pair(c,anchor)
                np.testing.assert_array_equal(c['origin'],anchor['origin'])
            method=r['method']
            with np.load(raw/'trajectory.npz') as z:
                live=z['valid'][:,0].astype(bool)
                active=np.array([],dtype=int)
                if method.endswith('_guard'):
                    enabled=z['guard_active'][:,0]
                    assert not np.any(enabled & (~live | ~z['guard_valid'][:,0]))
                    active=np.flatnonzero(enabled)
                    # The guard overrides the pre-filter speed target on every active lower tick.
                    lower_live=z['lower_trace'][:,:,0,6].reshape(-1).astype(bool)
                    applied_ticks=np.repeat(enabled,4) & lower_live
                    interface=json.loads((Path(c['bundle'])/'config.json').read_text())['residual_interface']
                    np.testing.assert_allclose(z['target'][applied_ticks,0,0],-interface['speed_negative_scale_m_s'],rtol=0,atol=1e-6)
                rt=np.flatnonzero(live)*.02
                rewards=z['reward'][live,0]
                post_time=rt+.02
                post_time[-1]=summary['time_s']
            first=active[0]*.02 if len(active) else np.nan
            seconds=np.maximum(0.,np.minimum((active+1)*.02,summary['time_s'])-active*.02).sum()
            row=dict(condition=condition,group=group,alpha=r['alpha'],method=method,slot=int(r['slot']),
                     first_trigger_s=first,triggered_s=seconds,roll=int('roll_limit' in summary['terminal']),
                     **{k:v for k,v in summary.items() if k!='method'})
            rows.append(row);outcomes[method]=row
            histories[method]=(trace,t)
            line,=xy.plot(trace[:,0],trace[:,1],label=method,lw=1.3)
            color=line.get_color()
            for ax,xx,yy in [(rew,post_time,rewards),(cum,post_time,rewards.cumsum()),(speed,t,trace[:,2]-c['settings']['user_speed_m_s'])]:
                ax.plot(xx,yy,color=color,lw=1.15)
            if len(active):
                first_idx=t.searchsorted(first)
                xy.scatter(*trace[min(first_idx,len(t)-1),:2],color=color,s=20,marker='v')
                speed.axvline(first,color=color,ls=':',lw=.8)
            if not summary['full']:
                xy.plot(*trace[-1,:2],marker='x',color=color)
        # Crop the long straight extension to the region visited in this trial.
        bounds=xy.dataLim.get_points().copy()
        crop=np.all((route>=bounds[0]-.7)&(route<=bounds[1]+.7),axis=1)
        xy.plot(*route[crop].T,'k:',label='Original route')
        if r['center_m']:
            sc=float(r['center_m'])+float(r['ds_cm'])/100
            xy.plot(*[np.interp(sc,arc,component) for component in route.T],marker='s',
                    markerfacecolor='none',markeredgecolor='k',ls='',label='Ridge center')
        else:
            for ax in (rew,cum,speed):
                ax.axvspan(2.8,2.95,color='grey',alpha=.16)
        xy.set(xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
        for ax,label in [(rew,'Reward per policy step'),(cum,'Cumulative reward'),(speed,'Forward speed error (m/s)')]:
            ax.set(xlabel='Time (s)',ylabel=label)
        xy.legend(fontsize=8)
        for ax in axes.flat: ax.grid(alpha=.2)
        fig.suptitle(f"H75 and Local300 | {condition} | physical group {group} | exploratory DEV\n"
                     "Triangle: first live intervention; cross: termination. Rewards use each lane's declared preference.")
        name=f'{condition}_g{group}.png'
        fig.savefig(figures/name,dpi=120);plt.close(fig)
        links.append(f'[{condition}, group {group}](figures/{name})')
        for base in ('fixed','preselected'):
            label=base if base in outcomes else 'fixed'
            a,b=outcomes[label],outcomes[label+'_guard']
            first=b['first_trigger_s']
            prefix=compare_history(histories[label],histories[label+'_guard'],first if np.isfinite(first) else np.inf)
            history_checks.append(dict(condition=condition,group=group,comparison=base,
                                       reference_terminal=a['terminal'],candidate_terminal=b['terminal'],
                                       first_trigger_s=first,reused=base!=label,**prefix))
            for metric in ('full','hold','A','B','C','roll'):
                pairs.append(dict(condition=condition,group=group,comparison=base,metric=metric,
                                  baseline=a[metric],guard=b[metric],gain=int(b[metric]>a[metric]),
                                  loss=int(b[metric]<a[metric]),same_trajectory_reuse=(base!=label)))
        if r['center_m'] and float(r['center_m'])==4.2:
            for suffix in ('','_guard'):
                fixed='fixed'+suffix
                selected='preselected'+suffix
                history_checks.append(dict(condition=condition,group=group,comparison='repeat'+suffix,
                                           reference_terminal=outcomes[fixed]['terminal'],candidate_terminal=outcomes[selected]['terminal'],
                                           first_trigger_s=outcomes[selected]['first_trigger_s'],reused=False,
                                           **compare_history(histories[fixed],histories[selected],np.inf)))
    write_csv(out/'episodes.csv',rows)
    write_csv(out/'pairs.csv',pairs)
    write_csv(out/'histories.csv',history_checks)
    aggregates=[]
    for cohort in ['all',*sorted(set(x['condition'] for x in pairs))]:
        for base in ('fixed','preselected'):
            for metric in ('full','hold','A','B','C','roll'):
                g=[p for p in pairs if p['metric']==metric and p['comparison']==base and (cohort=='all' or p['condition']==cohort)]
                up=sum(x['gain'] for x in g);down=sum(x['loss'] for x in g)
                aggregates.append(dict(condition=cohort,comparison=base,metric=metric,n=len(g),
                                      baseline=sum(x['baseline'] for x in g),guard=sum(x['guard'] for x in g),
                                      gains=up,losses=down,
                                      descriptive_exact_p=binomtest(up,up+down,.5).pvalue if up+down else 1.))
    write_csv(out/'counts.csv',aggregates)
    physical_checks=[h for h in history_checks if not h['reused']]
    before=[h for h in physical_checks if h['comparison'] in ('fixed','preselected')]
    repeats=[h for h in physical_checks if h['comparison'].startswith('repeat')]
    (out/'README.md').write_text('\n'.join(['# Online envelope-triggered speed override','',
          'All 89 paired DEV cases. Triangles mark the first active guard and crosses mark early terminations. A decrease in the roll-event count is favorable. The same physical groups occur in multiple layouts, so pooled exact paired tests are descriptive. Trigger durations end at the recorded physical endpoint.',
          '', '[Full counts and paired comparisons](counts.csv) · [Casewise gains and losses](pairs.csv) · [Endpoints, trigger times and physical errors](episodes.csv) · [Pre-action histories and same-preference repeats](histories.csv)',
          '', 'Each history comparison ends immediately before the first live intervention. For untriggered pairs and the 4.2 m technical repeats it covers the whole observed trajectory. A changed endpoint on an untriggered pair, or divergence before the first override, prevents attribution of that change to the intervention. Each case records both endpoint types, including any change from a roll event to another early termination.',
          '', f'Exact recorded histories through the pre-action boundary: {sum(h["observed_history_equal"] for h in before)}/{len(before)} physical comparisons. Whole-trajectory agreement on the repeated 4.2 m arms: {sum(h["observed_history_equal"] and h["reference_terminal"]==h["candidate_terminal"] for h in repeats)}/{len(repeats)}.',
          '', '| Comparison | Metric | Baseline | Guard | Increases | Decreases | n |','|---|---|---:|---:|---:|---:|---:|',
          *[f"|{r['comparison']}|{r['metric']}|{r['baseline']}|{r['guard']}|{r['gains']}|{r['losses']}|{r['n']}|" for r in aggregates if r['condition']=='all'],
          '',*links,'']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
