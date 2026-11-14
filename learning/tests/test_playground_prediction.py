"""Prediction is causal and matches the nominal CPU model and MuJoCo geometry."""
import unittest
import numpy as np
import torch
import mujoco
from sttw_control.playground_prediction import NominalRollout,BodyEnvelope
from sttw_control.playground_arx import Reference,fit,rollout

class PredictionTest(unittest.TestCase):
    def test_batched_free_rollout_matches_reference(self):
        s=np.linspace(0,10,501);xy=np.stack((s,.1*np.sin(s)),1);h=np.arctan2(np.gradient(xy[:,1]),np.gradient(xy[:,0]));bank=dict(s=s,xy=xy[None],heading=h[None],curvature=np.gradient(h,s)[None])
        follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=.3,lookahead_min_m=.3,lookahead_max_m=2.)
        rng=np.random.default_rng(7);x=rng.normal(size=(100,33));model=fit(x,.9*x[:,:9]+.1*x[:,9:18])
        pose=np.zeros((2,12));pose[:,0]=[1,3];pose[:,2]=.2;pose[:,3]=1
        xi=np.zeros((2,9));xi[:,0]=1.8;xi[:,5]=[.1,-.1]
        pred=NominalRollout(model,follower,'cpu',torch.float64)
        t=lambda x:torch.tensor(x,dtype=torch.float64)
        actual,_=pred(t(pose),t(xi),t(xi),t(s),t(np.repeat(bank['xy'],2,0)),t(np.repeat(bank['heading'],2,0)),t(np.repeat(bank['curvature'],2,0)),20)
        expected,_=rollout(model,Reference(bank,0,follower),pose,xi,xi,20)
        np.testing.assert_allclose(actual.numpy(),expected,atol=1e-9)
        poses,velocities,domain=pred(t(pose),t(xi),t(xi),t(s),t(np.repeat(bank['xy'],2,0)),t(np.repeat(bank['heading'],2,0)),t(np.repeat(bank['curvature'],2,0)),20,return_domain=True)
        for step in [0,10,20]:
            features=Reference(bank,0,follower).features(poses[:,step].numpy(),velocities[:,step].numpy())
            np.testing.assert_allclose(domain[:,step].numpy(),features[:,9:12],atol=1e-9)


    def test_body_support_matches_mujoco_transforms(self):
        # Native topology with nonzero hinge anchors and rotated bodies/geoms.
        names=['rearwheel_joint','steering_joint','frontwheel_joint','hip_joint','knee_joint']
        children=''
        for i,name in enumerate(reversed(names)):
            children=f'<body pos=".1 .02 .03" euler=".1 .2 .3"><joint name="{name}" pos=".02 0 .01" axis="0 1 0"/><geom type="box" size=".02 .03 .04" pos=".03 .01 0" euler=".2 .1 .3"/>{children}</body>'
        xml='<mujoco><compiler angle="radian"/><worldbody><body><freejoint/><geom type="box" size=".1 .05 .03"/>'+children+'</body></worldbody></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.qpos[:7]=[1,2,.3,.98,.1,.05,.1];d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7]);d.qpos[7:]=[.1,.2,-.1,.3,-.4];mujoco.mj_forward(m,d)
        order=['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint'];p=np.r_[d.qpos[:7],[d.qpos[m.jnt_qposadr[m.joint(name).id]] for name in order]]
        lo,hi=BodyEnvelope(m,'cpu',torch.float64)(torch.tensor(p[None]));rot=d.geom_xmat.reshape(-1,3,3);extent=(abs(rot[:,:2,:])*m.geom_size[:,None,:]).sum(-1)
        np.testing.assert_allclose(lo[0].numpy(),d.geom_xpos[:,:2]-extent,atol=1e-12);np.testing.assert_allclose(hi[0].numpy(),d.geom_xpos[:,:2]+extent,atol=1e-12)

        lo3,hi3=BodyEnvelope(m,'cpu',torch.float64,dimensions=3)(torch.tensor(p[None]))
        extent3=(abs(rot)*m.geom_size[:,None,:]).sum(-1)
        np.testing.assert_allclose(lo3[0].numpy(),d.geom_xpos-extent3,atol=1e-12)
        np.testing.assert_allclose(hi3[0].numpy(),d.geom_xpos+extent3,atol=1e-12)

if __name__=='__main__':unittest.main()
