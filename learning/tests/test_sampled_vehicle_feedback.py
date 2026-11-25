import pytest
import torch

from sttw_control.sampled_vehicle_feedback import SampledVehicleFeedback
from sttw_control.torch_path_reference import TorchPathReference


def test_multirate_delay_clean_route_and_reset():
    cfg={'path':dict(projection_backward_m=.05,projection_forward_floor_m=.3,
                     projection_forward_motion_multiplier=2.,projection_forward_padding_m=.1),
         'follower':dict(lookahead_time_s=.2,lookahead_min_speed_m_s=.2,lookahead_min_m=.1,lookahead_max_m=1.,
                         wheelbase_m_expected=.408,caster_deg_expected=0,steer_reference_max_rad=.8,preview_body_x_min_m=.05)}
    s=torch.arange(0,6,.1);z=torch.zeros_like(s)
    ref=TorchPathReference(dict(s=s,xy=torch.stack((s,z),dim=1),heading=z,curvature=z),cfg)
    f=SampledVehicleFeedback(dict(location_hz=20,availability_s=.01,noise_std=[0]*10,seed=4053),ref,torch.ones(1),.1)
    measurement=torch.zeros(1,7);pose=torch.zeros(1,3);v=torch.tensor([1.])
    f.reset(measurement,pose,v)
    for k in range(1,26):
        pose[0]=torch.tensor([k*.02,k*.01,0]);measurement[0,5]=k
        f.advance(measurement,pose,v)
        stamp=max(k-2,0); fix=stamp-stamp%10
        assert (f.imu_tick,f.location_tick)==(stamp,fix)
        assert float(f.speed)==pytest.approx(.1*stamp)
        assert float(f.pose[0,0])==pytest.approx(.02*fix)
        assert float(f.projection['cross_track'])==pytest.approx(k*.01)
    f.reset(torch.zeros_like(measurement),torch.zeros_like(pose),v)
    assert float(f.pose[0,0])==float(f.speed)==0 and len(f.frames)==1
    f.std[:]=.1
    f.reset(measurement,pose,v)
    first=f.measurement.clone()
    f.advance(measurement,pose,v)
    f.reset(measurement,pose,v)
    torch.testing.assert_close(f.measurement,first,rtol=0,atol=0)
