"""Development-only candidate acceptance and recovery-first ordering."""
import numpy as np


def rank_candidate(candidate,baseline,*,speed_slack=.01,nominal_slack=.01):
    arrays=[np.asarray(candidate[k],dtype=float) for k in ('radial_rmse','speed_rmse','nominal_radial_rmse','post_event_peak','settling_seconds')]
    if not all(np.isfinite(x).all() for x in arrays):return None,'nonfinite validation'
    reference=[np.asarray(baseline[k],dtype=float) for k in ('speed_rmse','nominal_radial_rmse')]
    if not all(np.isfinite(x).all() for x in reference):return None,'nonfinite baseline reference'
    if any(candidate['failed']) or any(candidate['nominal_failed']):return None,'physical failure'
    if np.any(np.asarray(candidate['speed_rmse'])>np.minimum(.2,np.asarray(baseline['speed_rmse'])+speed_slack)):
        return None,'speed regression against paired baseline'
    if np.any(np.asarray(candidate['nominal_radial_rmse'])>np.asarray(baseline['nominal_radial_rmse'])+nominal_slack):
        return None,'undisturbed tracking regression'
    event=np.asarray(candidate['event_present'],bool)
    if not event.any():return (0.,0.,float(np.mean(candidate['radial_rmse']))),'eligible'
    complete=np.asarray(candidate['post_event_hold_complete'],bool)[event]
    # Missing recovery is right-censored at the observation budget, never zero.
    return (float(np.mean(~complete)),float(np.mean(np.asarray(candidate['settling_seconds'])[event])),
            float(np.mean(np.asarray(candidate['post_event_peak'])[event]))),'eligible'
