"""Compare observed handoff branches to same-state repeats on their shared time grid."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


FIELDS = ('qpos', 'qvel', 'applied_targets', 'controller_state', 'rb_state')
EVENTS = ('rb_entry_s', 'first_far_contact_s', 'handoff_s', 'duration_s')


def load_case(root, row, tick):
    with np.load(root / row['case_dir'] / 'trajectory.npz') as z:
        steps = np.rint(z['trace'][:, 0] / .005).astype(int)
        start = np.flatnonzero(steps == tick).item()
        np.testing.assert_array_equal(steps[start:], np.arange(tick, steps[-1] + 1))
        data = {k: z[k][start:] for k in (*FIELDS, 'applied_target_valid')}
        data['phase'] = z['trace'][start:, 19]
        return data, z['reference_xy']


def compare(a, b):
    n = min(len(a['phase']), len(b['phase']))
    values, onset, initial = {}, {}, {}
    for field in FIELDS:
        d = np.abs(a[field][:n] - b[field][:n])
        valid = np.arange(n)
        if field == 'applied_targets':
            valid = valid[a['applied_target_valid'][:n] & b['applied_target_valid'][:n]]
        peak = d[valid].max(axis=1)
        changed = valid[peak > 0]
        values[field] = float(peak.max())
        onset[field] = int(changed[0]) if len(changed) else None
        initial[field] = float(peak[0]) if valid[0] == 0 else None
    values['xy_m'] = float(np.linalg.norm(a['qpos'][:n,:2] - b['qpos'][:n,:2], axis=1).max())
    values['phase_mismatch_steps'] = int(np.sum(a['phase'][:n] != b['phase'][:n]))
    return values, onset, initial


def evaluate(stage, parent, branch, anchor, ax):
    rows = json.loads((parent / 'status.json').read_text())['results']
    suffix = json.loads((branch / 'status.json').read_text())['results']
    if parent == branch:
        n = len(rows) // 5
        rows, suffix = rows[:n], suffix[n:]
    else:
        n = len(rows)
    assert len(suffix) == 4 * n
    tick = anchor['tick']
    results = []
    for i in anchor['lanes']:
        row = rows[i]
        repeats = [suffix[n*j+i] for j in range(4)]
        for r in repeats:
            assert (r['route'], r['variant']) == (row['route'], row['variant'])
        records = [row, *repeats]
        original, reference = load_case(parent, row, tick)
        samples = [original, *[load_case(branch, r, tick)[0] for r in repeats]]
        def match(x, y):
            vals, onset, initial = compare(samples[x], samples[y])
            for event in EVENTS:
                u, v = records[x][event], records[y][event]
                # Event presence is a separate categorical comparison.
                vals[event+'_presence'] = int((u is None) != (v is None))
                if u is not None and v is not None:
                    vals[event] = abs(u-v)
            vals['termination_changed'] = int(records[x]['termination'] != records[y]['termination'])
            return vals, onset, initial
        disk, onset, initial = match(0, 4)
        same = [match(x,y)[0] for x,y in itertools.combinations((1,2,3),2)]
        baseline = {k:max(b[k] for b in same if k in b) for k in disk if any(k in b for b in same)}
        outside = [k for k in disk if k not in baseline or disk[k] > baseline[k]]
        results.append(dict(stage=stage, case=row['case_dir'], lane=i, tick=tick,
                            disk_vs_continuous=disk, repeat_envelope=baseline,
                            two_memory_only=same[-1], outside=outside,
                            first_changed_after_ticks=onset, at_capture=initial,
                            terminations=[r['termination'] for r in records]))
        if ax is not None:
            ax.plot(reference[:,0], reference[:,1], ':', color='0.55', linewidth=1.1)
            for v,color,style in zip(samples,['#124c74','#929ea9','#929ea9','#929ea9','#c24016'],['-','-','-','-','--']):
                xy=v['qpos'][:,:2]
                ax.plot(*xy.T,style,color=color,linewidth=1.2)
            ax.plot(*original['qpos'][0,:2],'o',ms=3,color='#124c74')
    if ax is not None:
        xy = ax.lines
        points = np.concatenate([line.get_xydata() for j,line in enumerate(xy) if j%7 != 0])
        xmin, xmax = points[:,0].min()-.1, points[:,0].max()+.1
        visible_y = [points[:,1]]
        for ref_line in xy[::7]:
            ref = ref_line.get_xydata()
            visible_y.append(ref[(ref[:,0] >= xmin) & (ref[:,0] <= xmax),1])
        y = np.concatenate(visible_y)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y.min()-.12,y.max()+.12)
        ax.set_aspect('equal', adjustable='box')
        ax.set(title=stage + f'   t={tick*.005:.2f} s', xlabel='World x (m)', ylabel='World y (m)')
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest',type=Path,help='JSON list of parent/branch paths and observed stage anchors')
    ap.add_argument('output',type=Path)
    args = ap.parse_args()
    cases=json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots((len(cases)+1)//2,2,squeeze=False,figsize=(11,2.6*((len(cases)+1)//2)),layout='constrained')
    out=[]
    for ax,case in zip(axes.flat,cases):
        a={k:case[k] for k in ('tick','lanes')}
        out.extend(evaluate(case['stage'],Path(case['parent']),Path(case['branch']),a,ax))
    for ax in list(axes.flat)[len(cases):]:
        ax.remove()
    fig.suptitle('Known map · original reference (dotted), continuous (blue), same-state repeats (gray), disk restore (orange)')
    fig.savefig(args.output/'xy.png',dpi=145)
    (args.output/'continuity.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
