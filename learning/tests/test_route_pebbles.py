import unittest
import numpy as np
from sttw_control.route_pebbles import cap_height,stone_layout,project_route

class PebbleGeometryTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(seed=20261107,s_range_m=[3.,7.],half_width_m=.8,density_per_m2=24,radius_range_m=[.055,.08],height_m=.01,blend_length_m=.3)
    def test_overlap_uses_max_and_compact_support(self):
        stones=np.array([[5.,0.,.1,.1],[5.,0.,.1,.1]])
        np.testing.assert_allclose(cap_height(np.array([5.,5.2]),np.zeros(2),stones,self.c),[.01,0.])
    def test_height_pair_preserves_centers_and_radii(self):
        a=stone_layout(self.c);b=stone_layout(dict(self.c,height_m=.02))
        np.testing.assert_array_equal(a,b);self.assertEqual(len(a),154)
    def test_route_projection_sign_and_progress(self):
        route=dict(xy=np.column_stack((np.arange(8.),np.zeros(8))),s=np.arange(8.))
        s,n=project_route(np.array([[3.4,.2],[5.1,-.3]]),route)
        np.testing.assert_allclose(s,[3.4,5.1]);np.testing.assert_allclose(n,[.2,-.3])

if __name__=='__main__':unittest.main()
