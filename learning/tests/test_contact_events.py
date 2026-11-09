import numpy as np
from sttw_control.contact_events import extract_contact_events


def fixture():
 t=np.arange(8)*.1;p=np.zeros((8,2,3));p[:,:,0]=np.arange(8)[:,None]*.2
 f=np.zeros_like(p);f[:,:,2]=10;f[1:3,:,2]=0;f[3,1,2]=0
 return t,p,f,np.ones(8,dtype=bool)


def test_no_recovery_is_inferred_from_touchdown():
 result=extract_contact_events(*fixture(),.5,1.)
 assert result['recovery_status'].startswith('unlabelled')
 assert [e['time_s'] for e in result['events'] if e['kind']=='support_lost']==[.1]
 assert next(e for e in result['events'] if e['kind']=='first_far_bank_contact_proxy')['wheels']==[0]


def test_hold_and_failure_and_censoring_are_distinct():
 args=fixture();mask=np.ones(8,dtype=bool)
 assert extract_contact_events(*args,.5,1.,mask,.2)['recovery_status']=='observed'
 assert extract_contact_events(*args,.5,1.,mask,1.)['recovery_status']=='right_censored_at_observation_end'
 assert extract_contact_events(*args,.5,1.,mask,1.,True)['recovery_status']=='failed_before_recovery'


def test_invalid_interval_breaks_recovery_hold():
 t,p,f,v=fixture();v[5]=False
 assert extract_contact_events(t,p,f,v,.5,1.,np.ones(8,dtype=bool),.3)['recovery_status']=='right_censored_at_observation_end'


def test_starting_airborne_does_not_invent_observed_takeoff():
 t,p,f,v=fixture();f[0]=0
 result=extract_contact_events(t,p,f,v,.5,1.)
 assert not any(e['kind']=='support_lost' for e in result['events'])
