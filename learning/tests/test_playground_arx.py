"""ARX integration and reference dependence without future-state input."""
import unittest
import numpy as np
from sttw_control.playground_arx import integrate,fit,predict,geometry_error_bound


class ArxTest(unittest.TestCase):
    def test_body_velocity_is_rotated_into_world(self):
        pose=np.array([[0.,0.,.2,np.sqrt(.5),0.,0.,np.sqrt(.5),.1,0.,0.,0.,0.]])
        xi=np.zeros((1,9));xi[:,0]=2;xi[:,6]=.5
        result=integrate(pose,xi,xi)
        np.testing.assert_allclose(result[0,:3],[0,.04,.2],atol=1e-12)
        self.assertAlmostEqual(result[0,7],.11)

    def test_geometry_bound_covers_pure_translation(self):
        a=np.zeros((2,12));a[:,3]=1;b=a.copy();b[:,0]=.2
        np.testing.assert_allclose(geometry_error_bound(a,b,.5,np.ones(5)*.5),[.2,.2])

    def test_ridge_recovers_stable_linear_response(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(1000,22));y=.7*x[:,:9]+.2*x[:,9:18]
        model=fit(x,y,ridge=1e-8);actual=predict(model,x[:,:9],x[:,9:18],x[:,18:])
        np.testing.assert_allclose(actual,y,atol=1e-6)


if __name__=='__main__':unittest.main()
