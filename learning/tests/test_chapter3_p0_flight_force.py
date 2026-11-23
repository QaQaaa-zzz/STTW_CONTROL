"""Align physical-force receipts with the observed control and contact clocks."""
from pathlib import Path

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize('onset,steps,force,air,n,hand,expected', [
    (0, 20, 5, True, 5, .49, ('partial_applied_overlap', 73, 73, False)),
    (6, 20, -5, True, 5, .49, ('full_applied_overlap', 100, 100, False)),
    (90, 9, 5, True, 5, .49, ('no_overlap', 0, 0, False)),
    (0, 0, 0, True, 5, .49, ('zero_force_marker', 0, 73, False)),
    (-1, 0, 5, True, 5, .49, ('no_trigger', 0, None, False)),
    (0, 20, 5, False, 5, None, ('no_crossing_flight', None, None, None)),
    (0, 20, 5, True, 1, .49, ('partial_applied_overlap', 75, 75, False)),
    (0, 20, 5, True, 5, .20, ('partial_applied_overlap', 73, 73, True)),
    (30, 20, 5, True, 5, .20, ('overlap_censored', 50, 50, True)),
    (6, 20, 5, True, 5, .42, ('full_applied_overlap', 100, 100, True)),
])
def test_prefix_flight_force(tmp_path,monkeypatch,onset,steps,force,air,n,hand,expected):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cli'))
    from analyze_chapter3_continuous_p0_branches import prefix_flight_force
    from report_platform_controls import flight
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><freejoint/>
      <geom name="frontwheel_collision" type="ellipsoid" pos=".14 0 0" size=".07 .02 .07"/>
      <geom name="rearwheel_collision" type="ellipsoid" pos="-.14 0 0" size=".07 .02 .07"/>
    </body></worldbody></mujoco>""")
    t = np.arange(100)*.005
    q = np.tile(model.qpos0,(len(t),1))
    q[:,0] = -.30 + 2.5*t
    q[:,2] = .07 + .24*np.sin(np.pi*np.clip((t-.025)/.4,0,1))
    loads = np.zeros((len(t), n, 4))
    loads[:6,:,1] = 20
    if n == 5:
        loads[6,:2,1] = 20
    loads[85:,:,:2] = 20
    if not air:
        loads[:,:,1] = 20
    flags = np.zeros((len(t),3),bool)
    flags[85:,2] = True
    p = tmp_path/'trajectory.npz'
    np.savez(p,trace=t[:,None].astype('f'),qpos=q,contact_substep=loads,
             wheel_contact_flags=flags,contact_physics_dt_s=.005/n)
    row = dict(variant={'force_y_N':force},air_event_tick=onset,applied_force_ticks=steps,
               rb_entry_s=0.,handoff_s=hand)
    s = prefix_flight_force(row,p,model,{'front_x_m':0.,'far_bank_height_m':0.})
    assert (s['pulse_flight_status'],s['pulse_actual_overlap_ms'],s['pulse_marker_overlap_ms'],
            s['flight_end_censored']) == expected
    assert s['flight_contact_grid_ms'] == 5/n
    assert s['verified'] == bool(air and hand == .49)
    # Compare against the previously frozen +2 row scorer, including a landing one tick late.
    old_verified = False
    if hand is not None:
        end = round(hand/.005)+2
        supported = flags[:end,2] & (loads[:end,-1,:2]>1e-5).all(1)
        _, _, landing = flight(model,q[:end],loads[:end],supported,t[:end].astype('f'),0.,0.,0.,0.)
        old_verified = landing is not None and landing <= hand+1e-5
    assert s['verified'] == old_verified
    assert s['pulse_applied_end_ms'] == ((onset+steps)*5 if steps else None)
    assert s['zero_force_marker'] == (force==0)
    assert s['pulse_overlap_upper_ms'] == (100 if expected[0]=='overlap_censored' else expected[1])
