"""Offline amplitude, reversal events, and spectral power: distinct quantities."""
import numpy as np
from scipy.signal import welch

def signal_metrics(x,fs=50,rate_deadband=.05,excursion=.002):
    x=np.asarray(x,float);dx=np.diff(x);rate=dx*fs
    raw=int(np.sum(rate[1:]*rate[:-1]<0))
    # Adjacent nonzero direction runs: require both movements to be material.
    valid=np.flatnonzero(abs(rate)>rate_deadband)
    runs=[]
    for idx in valid:
        sign=np.sign(rate[idx])
        if runs and runs[-1][0]==sign and idx==runs[-1][2]+1:
            runs[-1][1]+=abs(dx[idx]);runs[-1][2]=idx
        else:runs.append([sign,abs(dx[idx]),idx])
    qualified=sum(a[0]!=b[0] and min(a[1],b[1])>=excursion for a,b in zip(runs,runs[1:]))
    centered=x-x.mean();f,p=welch(centered,fs=fs,window='hann',nperseg=min(128,len(x)),noverlap=min(128,len(x))//2,detrend=False)
    total=float(p.sum());band=(f>=10)&(f<=25)
    return dict(signal_sign_reversal_events=int(np.sum(x[1:]*x[:-1]<0)),derivative_reversal_definition='sign changes of first difference of this signal; for measured actual_steer_rate use signal_sign_reversal_events to count measured rate reversals',samples=len(x),duration_s=len(x)/fs,mean=float(x.mean()),peak_to_peak=float(np.ptp(x)),rms=float(np.sqrt(np.mean(x*x))),ac_rms=float(np.sqrt(np.mean(centered**2))),rate_rms=float(np.sqrt(np.mean(rate*rate))),raw_reversal_events=raw,raw_reversal_events_per_s=raw/(len(x)/fs),amplitude_reversal_events=int(qualified),amplitude_reversal_events_per_s=qualified/(len(x)/fs),peak_frequency_hz=float(f[np.argmax(p)]) if total>1e-25 else None,high_band_fraction=float(p[band].sum()/total) if total>1e-25 else None),f,p
