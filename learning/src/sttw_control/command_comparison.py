"""Saved command rollouts: paired XY and step-reward comparisons, no simulation."""
from pathlib import Path
import hashlib
import json
import numpy as np


def reference_pose(schedule, times, initial):
    """Exact planar unicycle integration, splitting every command transition."""
    schedule = np.asarray(schedule, dtype=float)[:, :3]
    times = np.asarray(times, dtype=float)
    if (schedule[0, 0] != 0 or np.any(np.diff(schedule[:, 0]) <= 0)
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Schedule and sample times must increase from zero')
    pose = np.asarray(initial, dtype=float).copy()
    result = [pose.copy()]
    for left, right in zip(times[:-1], times[1:]):
        edges = [left, *schedule[(schedule[:, 0] > left) & (schedule[:, 0] < right), 0], right]
        for a, b in zip(edges[:-1], edges[1:]):
            _, v, r = schedule[np.searchsorted(schedule[:, 0], a, side='right') - 1]
            dt = b-a
            angle = r*dt
            distance = v*dt*np.sinc(angle/(2*np.pi))
            pose[:2] += distance*np.array([np.cos(pose[2]+angle/2), np.sin(pose[2]+angle/2)])
            pose[2] += angle
        result.append(pose.copy())
    return np.asarray(result)


def _load(path):
    with np.load(path/'trace.npz') as data:
        trace = {k: data[k] for k in ('time','pose','reward','priority_alpha','terminated')}
    config = json.loads((path/'declaration.json').read_text())['config']
    schedule = json.loads((path/'commands.json').read_text())['schedule']
    return trace, config, np.asarray(schedule)[:, :3]


def generate_comparison(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    runs = {k: Path(v).resolve() for k, v in runs.items()}
    panels = {label: {p.relative_to(root/'evaluation'): p for p in
                     (root/'evaluation').glob('alpha_*/seed_*/*/residual/trace.npz')}
              for label, root in runs.items()}
    keys = list(next(iter(panels.values())))
    if not keys or any(set(p) != set(keys) for p in panels.values()):
        raise ValueError('All methods must have the same nonempty case/alpha/seed coverage')
    groups = {}
    sources = []; entries = []
    for key in keys:
        alpha_dir, seed, case = key.parts[:3]
        groups.setdefault((case, seed), []).append(key)
    for (case, seed), group in sorted(groups.items()):
        columns = []
        for key in sorted(group):
            methods = {}; anchor = None
            for label, root in runs.items():
                path = panels[label][key].parent
                tr, cfg, schedule = _load(path)
                base, bc, bs = _load(path.parent/'baseline')
                if anchor is None:
                    anchor = (base, cfg, schedule)
                    methods['ECBC + ESO'] = base
                else:
                    ab, ac, ass = anchor
                    for field in ('time','pose','reward','terminated'):
                        if not np.array_equal(ab[field], base[field]):
                            raise ValueError(f'Baseline mismatch: {label} {key} {field}')
                    if cfg['horizon_seconds'] != ac['horizon_seconds'] or not np.allclose(schedule, ass):
                        raise ValueError(f'Command mismatch: {label} {key}')
                if not np.allclose(tr['pose'][0], base['pose'][0]) or not np.array_equal(tr['priority_alpha'], np.full_like(tr['priority_alpha'],base['priority_alpha'][0])):
                    raise ValueError(f'Initial pose or alpha mismatch: {label} {key}')
                if not np.allclose(bs,schedule):raise ValueError('Baseline command mismatch')
                methods[label] = tr
                for directory in (path, path.parent/'baseline'):
                    for filename in ('trace.npz','commands.json','declaration.json'):
                        src = directory/filename
                        sources.append({'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
            base, cfg, schedule = anchor
            times = np.linspace(0, cfg['horizon_seconds'], round(cfg['horizon_seconds']/cfg['controller']['dt'])+1)
            ref = reference_pose(schedule, times, base['pose'][0])
            columns.append((float(base['priority_alpha'][0]),methods,times,ref,schedule))
        columns.sort(key=lambda c:c[0])
        fig, axes = plt.subplots(1,len(columns),figsize=(6*len(columns),6),squeeze=False,layout='constrained')
        reward_fig, reward_axes = plt.subplots(2,len(columns),figsize=(6*len(columns),8),squeeze=False,layout='constrained')
        points = np.concatenate([p for _,methods,_,ref,_ in columns for p in [ref[:,:2],*[t['pose'][:,:2] for t in methods.values()]]])
        lo,hi = points.min(0),points.max(0); pad=max(float((hi-lo).max())*.06,.2)
        arrays = {}; endpoints=[]
        for j,(alpha,methods,times,ref,schedule) in enumerate(columns):
            ax=axes[0,j];ax.plot(ref[:,0],ref[:,1],'k--',label='Command reference',lw=2)
            arrays[f'alpha_{alpha:g}_reference']=ref;arrays[f'alpha_{alpha:g}_reference_time']=times
            for t in schedule[1:,0]:
                rp=reference_pose(schedule,np.array([0.,t]),ref[0])[-1]
                ax.plot(*rp[:2],'ko',ms=4);ax.annotate(f'{t:g}s',rp[:2],xytext=(4,5),textcoords='offset points')
                for rax in reward_axes[:,j]:rax.axvline(t/cfg['controller']['dt'],color='gray',ls=':',alpha=.5)
            for i,(label,tr) in enumerate(methods.items()):
                color=f'C{i}';failed=bool(tr['terminated'][-1]);end=float(tr['time'][-1])
                ax.plot(tr['pose'][:,0],tr['pose'][:,1],color=color,label=label)
                ax.plot(*tr['pose'][-1,:2],marker='x' if failed else 's',color=color,ms=7)
                if failed:ax.annotate(f'{label}: {end:.3f}s',tr['pose'][-1,:2],xytext=(.02,.96-i*.045),textcoords='axes fraction',fontsize=8,color=color)
                steps=np.arange(1,len(tr['time']));rewards=tr['reward'][1:]
                for rax,values in zip(reward_axes[:,j],(rewards,np.cumsum(rewards,dtype=float))):
                    rax.plot(steps,values,color=color,label=label)
                    if failed:rax.plot(steps[-1],values[-1],'x',color=color,ms=8)
                prefix=f'alpha_{alpha:g}_method_{i}'
                for field in ('time','pose','reward'):arrays[prefix+'_'+field]=tr[field]
                endpoints.append(dict(alpha=alpha,method=label,array_prefix=prefix,end_seconds=end,failed=failed,observed_return=float(rewards.sum(dtype=float))))
            ax.set(xlim=(lo[0]-pad,hi[0]+pad),ylim=(lo[1]-pad,hi[1]+pad),xlabel='World x [m]',ylabel='World y [m]',title=f'alpha = {alpha:g}')
            ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=9)
            for row,rax in enumerate(reward_axes[:,j]):
                rax.set(xlabel='Control step',ylabel='Reward per step' if row==0 else 'Cumulative return',title=f'alpha = {alpha:g}',xlim=(0,len(times)-1))
                rax.set_yscale('symlog',linthresh=.01 if row==0 else 1);rax.grid(alpha=.2);rax.legend(fontsize=9)
        fig.suptitle(f'{case} | {seed}\nDashed: ideal command integration; x: failure, square: observed end. Reference is not a feasibility guarantee.')
        reward_fig.suptitle(f'{case} | {seed}\nActual total reward per step (top), cumulative return (bottom); symlog axes; dotted: command switches; x: failure.')
        stem=f'{case}_{seed}'
        for name,f in [('trajectory',fig),('reward_steps',reward_fig)]:
            for ext in ('png','pdf'):f.savefig(output/f'{stem}_{name}.{ext}',dpi=150)
            plt.close(f)
        np.savez_compressed(output/f'{stem}.npz',**arrays)
        entries.append(dict(case=case,seed=seed,stem=stem,endpoints=endpoints))
    (output/'manifest.json').write_text(json.dumps({'runs':{k:str(v) for k,v in runs.items()},'sources':sources,'cases':entries},indent=2)+'\n')
    (output/'INDEX.md').write_text('# 配对轨迹与总奖励对比\n\n参考轨迹由原始速度/世界偏航角速度指令积分，同初始位姿；不是可行性保证，也不是额外加入控制器的路径目标。实际轨迹在真实失败处结束，不补齐。总奖励为各分项之和，非单独新增奖励；上排逐步，下排累计，symlog坐标保留失败惩罚。全部数据为开发评估，不是独立测试。\n\n'+ '\n'.join(f"- {e['case']} / {e['seed']}: [轨迹]({e['stem']}_trajectory.png) · [每步总奖励]({e['stem']}_reward_steps.png) · [PDF]({e['stem']}_reward_steps.pdf) · [数据]({e['stem']}.npz)" for e in entries)+'\n')
    return entries
