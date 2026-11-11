import unittest
import numpy as np
from sttw_control.contact_momentum import roll_momentum

class ContactMomentumTest(unittest.TestCase):
    def test_orbital_momentum_and_translation_invariance(self):
        body=np.zeros((2,13));body[:,6]=1;body[:,1]=[-1,1];body[:,9]=[-2,2]
        mass=np.array([3.,3.]);com=np.zeros((2,3));inertia=np.zeros((2,3,3))
        a=roll_momentum(body,mass,com,inertia)
        np.testing.assert_allclose(a['angular_momentum_world'],[12,0,0]);self.assertEqual(a['roll_momentum'],-12)
        body[:,:3]+=[4,6,8];body[:,7:10]+=[3,2,1]
        b=roll_momentum(body,mass,com,inertia)
        np.testing.assert_allclose(a['angular_momentum_world'],b['angular_momentum_world'])

    def test_link_velocity_converts_to_com_velocity(self):
        body=np.zeros((2,13));body[:,6]=1;body[:,10]=2;body[:,1]=[-1,1]
        offsets=np.array([[0,.5,0],[0,-.5,0]])
        # Both link frames rotate, their COM velocities are +1 and -1 in z.
        got=roll_momentum(body,np.ones(2),offsets,np.tile(np.eye(3),(2,1,1)))
        np.testing.assert_allclose(got['angular_momentum_world'],[3,0,0])

    def test_rotating_axis_transport_is_not_external_torque(self):
        # One anisotropic body: L and omega need not be parallel.
        body=np.zeros((1,13));body[:,6]=1;body[:,10:13]=[1,2,3]
        got=roll_momentum(body,np.ones(1),np.zeros((1,3)),np.array([np.diag([2,3,4])]))
        np.testing.assert_allclose(got['angular_momentum_world'],[2,6,12])
        self.assertAlmostEqual(got['axis_transport_rate'],6.)

if __name__=='__main__':unittest.main()
