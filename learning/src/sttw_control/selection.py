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


def rank_command_candidate(candidate, baseline, *, speed_slack=.05, yaw_slack=.05, scope="initial"):
    """Development rank, not a safety certificate; failed-only banks remain labelled."""
    if scope=='full_episode':
        keys=('speed_rmse','yaw_rmse','episode_return','failed','terminal_tracking_hold')
        if not all(np.isfinite(np.asarray(source[k],float)).all() for source in (candidate,baseline) for k in keys):
            return None,'nonfinite command validation'
        nominal=~np.asarray(baseline['failed'],bool)
        speed=np.asarray(candidate['speed_rmse'])>np.asarray(baseline['speed_rmse'])+speed_slack
        yaw=np.asarray(candidate['yaw_rmse'])>np.asarray(baseline['yaw_rmse'])+yaw_slack
        delta=np.asarray(candidate['episode_return'])-np.asarray(baseline['episode_return'])
        return (float(np.sum(candidate['failed'])),float(np.sum(nominal&(speed|yaw))),float(np.sum(~np.asarray(candidate['terminal_tracking_hold'],bool))),-float(np.mean(delta))),'failure count, full-episode regression on surviving baselines, terminal tracking misses, negative paired return gain; candidate rank is not task acceptance'
    if scope!='initial':raise ValueError('invalid command ranking scope')
    keys=('initial_speed_rmse','initial_yaw_rmse','episode_return')
    if not all(np.isfinite(np.asarray(candidate[k],float)).all() for k in keys):
        return None,'nonfinite command validation'
    speed=np.asarray(candidate['initial_speed_rmse'])>np.asarray(baseline['initial_speed_rmse'])+speed_slack
    yaw=np.asarray(candidate['initial_yaw_rmse'])>np.asarray(baseline['initial_yaw_rmse'])+yaw_slack
    return (float(np.sum(candidate['failed'])),float(np.sum(speed|yaw)), -float(np.mean(candidate['episode_return']))),'failure count, initial tracking regression count, negative mean episode return'


def command_improved(score,best,min_delta):
    if score is None:return False
    if best is None:return True
    return score[:-1]<best[:-1] or (score[:-1]==best[:-1] and score[-1]<best[-1]-min_delta)


def command_should_stop(update,stale,min_updates,patience):
    return patience>0 and update>=min_updates and stale>=patience
