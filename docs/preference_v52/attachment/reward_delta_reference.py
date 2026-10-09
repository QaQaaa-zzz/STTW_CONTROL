#!/usr/bin/env python3
"""NumPy reference for ONLY the V5.2 deltas. Not a vehicle controller."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OLD_CAPS={'speed':100.,'steer':100.,'heading':300.,'roll':900.,'roll_rate':40.,'overspeed':50.,'low_speed':50.,'magnitude':10.,'upper_rate':10.,'upper_acceleration':20.,'yaw_recovery':20.,'reference_priority':40.,'command_compatibility':40.,'primary_excess':400.}
CAPS={**OLD_CAPS,'working_roll_excess':3000.}

def huber(x):
    x=np.asarray(x,dtype=float)
    return np.where(np.abs(x)<=1.,x*x,2*np.abs(x)-1.)

def working_roll_cost(roll_end, peak_abs_roll):
    risk=np.maximum(np.abs(roll_end),peak_abs_roll)
    raw=120.*huber(np.maximum(risk-.30,0.)/.02)
    return raw,np.minimum(raw,CAPS['working_roll_excess'])

def ref_priority(alpha,chi,g,dv,dd):
    if alpha not in (0,1): raise ValueError('only alpha 0/1')
    s=huber(np.maximum(abs(dd)-.01,0.)/.05)
    # Positive speed reference compensation is NOT speed sacrifice.
    v=huber(np.maximum(-dv-.05,0.)/.20)
    return 4.*chi*(1.-g)*((1-alpha)*s+alpha*v)

def normal_or_recovery_speed(ev):
    return 8.*huber(np.maximum(np.abs(ev)-.01,0.)/.1)

def failure_reward(remaining_including_current,gamma=.997):
    n=int(remaining_including_current)
    if n<1: raise ValueError('remaining horizon must include failed interval')
    return -5.-.1*.02*sum(CAPS.values())*(1-gamma**n)/(1-gamma)

def main():
    assert sum(OLD_CAPS.values())==2080 and sum(CAPS.values())==5080
    for a in (0,1):
        assert ref_priority(a,0,0,-.4,.1)==0
        assert ref_priority(a,1,1,-.4,.1)==0
    assert ref_priority(1,1,0,.10,0)==0
    assert ref_priority(1,1,0,-.10,0)>0
    assert ref_priority(0,1,0,-.4,0)==0
    assert ref_priority(0,1,0,0,-.08)>0
    assert working_roll_cost(.2,.299)[0]==0
    assert working_roll_cost(.20,.315)[0]>0  # substep excursion not hidden
    samples=[]
    for phi in [.2895,.30,.302,.3154,.3676,.4131]:
        old=40.*huber(max(abs(phi)-.26,0.)/.04)
        raw,eff=working_roll_cost(phi,abs(phi))
        samples.append({'abs_roll':phi,'retained_roll_cost':float(old),
                        'new_working_roll_cost':float(eff),'combined_roll_cost':float(old+eff)})
    report={'tests':'passed','vehicle_simulation_executed':False,'old_cap_sum':sum(OLD_CAPS.values()),'new_cap_sum':sum(CAPS.values()),'roll_examples':samples,'failure_first_interval_16s':float(failure_reward(800))}
    Path(__file__).with_name('reward_delta_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
