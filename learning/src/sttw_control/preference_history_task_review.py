"""Original-route shadow diagnostics from a genuine nonzero-progress snapshot."""
import numpy as np


def shadow_cross_track(xy, path, config, *, initial_xy, initial_progress):
    s = np.asarray(path['s'], float)
    route = np.asarray(path['xy'], float)
    heading = np.asarray(path['heading'], float)
    c = config['path']
    progress, previous_xy = float(initial_progress), np.asarray(initial_xy, float)
    errors, progress_trace = [], []
    for point in np.asarray(xy, float):
        distance = float(np.linalg.norm(point-previous_xy))
        lo = max(0., progress-c['projection_backward_m'])
        hi = min(s[-1], progress+max(c['projection_forward_floor_m'],
                                    c['projection_forward_motion_multiplier']*distance+c['projection_forward_padding_m']))
        first = max(0, int(np.searchsorted(s, lo, side='left'))-1)
        last = min(len(s)-2, int(np.searchsorted(s, hi, side='right'))-1)
        index = np.arange(first, last+1)
        a, v = route[index], route[index+1]-route[index]
        span = s[index+1]-s[index]
        u = np.sum((point-a)*v, axis=1)/np.sum(v*v, axis=1)
        u = np.clip(u, np.clip((lo-s[index])/span, 0, 1), np.clip((hi-s[index])/span, 0, 1))
        points, candidate = a+u[:, None]*v, s[index]+u*span
        d2 = np.sum((point-points)**2, axis=1)
        winner = int(np.argmin(np.where(d2 <= d2.min()+1e-10, abs(candidate-progress), np.inf)))
        h = float(np.interp(candidate[winner], s, heading))
        delta = point-points[winner]
        errors.append(-np.sin(h)*delta[0]+np.cos(h)*delta[1])
        progress = max(progress, float(candidate[winner]))
        progress_trace.append(progress)
        previous_xy = point
    return np.asarray(errors), np.asarray(progress_trace)


def paired_complete_metrics(records, candidate, baseline, metrics):
    """Keep incomplete/fatal pairs visible; compare common full survivors only."""
    keys = sorted({(r['state_id'], r['force_N']) for r in records})
    index = {(r['state_id'], r['force_N'], r['role']): r for r in records}
    included, excluded = [], []
    for state, force in keys:
        a, b = index[(state, force, candidate)], index[(state, force, baseline)]
        event = dict(state_id=state, force_N=force)
        if a['physical_label'] != 'PASS' or b['physical_label'] != 'PASS':
            excluded.append(dict(**event, candidate_outcome=a['physical_label'], baseline_outcome=b['physical_label']))
            continue
        included.append(dict(**event, differences={key: a[key]-b[key] for key in metrics}))
    return dict(candidate=candidate, baseline=baseline, common_complete=len(included),
                included=included, excluded=excluded,
                mean_differences={key: float(np.mean([r['differences'][key] for r in included]))
                                  if included else None for key in metrics},
                scope='Posthoc common survivors; pair-specific subset, not population dominance or Jalpha')
