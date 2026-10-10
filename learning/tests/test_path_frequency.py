import numpy as np
from sttw_control.path_frequency import signal_metrics

def test_psd_and_reversals_are_different_units():
 t=np.arange(500)/50;x=.01*np.sin(2*np.pi*15*t)
 m,_,_=signal_metrics(x,50)
 assert abs(m['peak_frequency_hz']-15)<.3
 assert m['high_band_fraction']>.99
 assert m['raw_reversal_events_per_s']>25
 assert m['amplitude_reversal_events']>0
 m,_,_=signal_metrics(x*1e-4,50)
 assert m['raw_reversal_events']>0 and m['amplitude_reversal_events']==0
