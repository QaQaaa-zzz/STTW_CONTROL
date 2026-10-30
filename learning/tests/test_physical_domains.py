"""Compile real isolated model domains; no physics integration or training."""
import numpy as np
import mujoco
from sttw_control.model import DEFAULT_XML,load_model,export_domain_model
from sttw_control.teleop_env import TeleopEnv


def test_physical_domain_reaches_runtime_without_changing_nominal(tmp_path):
    original=DEFAULT_XML.read_bytes();base=load_model();path=tmp_path/'domain.xml'
    receipt=export_domain_model(path,dict(wheel_sliding_friction=.6,chassis_mass_scale=1.1))
    env=TeleopEnv(backend='cpu',model_path=path);m=env.model
    np.testing.assert_allclose(m.body_mass[base.chassis],base.model.body_mass[base.chassis]*1.1)
    np.testing.assert_allclose(m.body_inertia[base.chassis],base.model.body_inertia[base.chassis]*1.1)
    for name in ['frontwheel','rearwheel']:
        body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
        ids=np.flatnonzero((m.geom_bodyid==body)&(m.geom_contype!=0))
        assert len(ids)==1
        np.testing.assert_allclose(m.geom_friction[ids[0],0],.6)
        floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
        assert m.geom_priority[ids[0]]>m.geom_priority[floor]
    np.testing.assert_array_equal(m.actuator_ctrlrange,base.model.actuator_ctrlrange)
    np.testing.assert_array_equal(m.actuator_forcerange,base.model.actuator_forcerange)
    assert m.opt.timestep==base.model.opt.timestep
    assert receipt['model']==env.bundle.identity
    assert DEFAULT_XML.read_bytes()==original
