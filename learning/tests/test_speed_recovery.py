import numpy as np
from sttw_control.speed_recovery import paired_speed


def test_nominal_bias_does_not_count_as_disturbance_rejection():
    t=np.arange(0,5.01,.01);base=np.full(len(t),2.05);dv=np.where((t>=1)&(t<2),-.1,0.)
    a=paired_speed(t,base+dv,t,base,start=1,end=2,target=2.1)
    b=paired_speed(t,base+.02+dv,t,base+.02,start=1,end=2,target=2.1)
    np.testing.assert_allclose(a['extra_speed'],b['extra_speed'])
    assert a['summary']['speed_excursion'] and a['summary']['speed_settling_after_event_s']>=.5-1e-8
    c=paired_speed(t,base,t,base,start=1,end=2,target=2.1)
    assert not c['summary']['speed_excursion'] and c['summary']['speed_settling_after_event_s'] is None
