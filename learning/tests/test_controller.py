from pathlib import Path
import subprocess
import numpy as np
import pytest

from sttw_control.controller import ControllerConfig, initial_controller, controller_step

ROOT = Path(__file__).resolve().parents[2]

def test_cpp_reset_clears_diagnostic_filter(tmp_path):
    pkg=ROOT/'mujoco_ros_ws20260313/mujoco_ros'
    binary=tmp_path/'reset'
    subprocess.run(['g++','-std=c++17','-I'+str(pkg/'include/mujoco_ros'),
                    str(Path(__file__).with_name('controller_reset.cpp')),
                    str(pkg/'src/ECBC/eso_based_controller.cpp'),str(pkg/'src/ECBC/pid_controller.cpp'),
                    '-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_cpp_sequence_parity(tmp_path):
    """Catches sign, update ordering, speed scheduling and ESO-state drift."""
    pkg = ROOT / 'mujoco_ros_ws20260313/mujoco_ros'
    binary = tmp_path / 'controller'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(pkg/'include/mujoco_ros'),
                    str(Path(__file__).with_name('controller_reference.cpp')),
                    str(pkg/'src/ECBC/eso_based_controller.cpp'),
                    str(pkg/'src/ECBC/pid_controller.cpp'), '-o', str(binary)], check=True)
    t = np.arange(900)*0.005
    speed = 1.6 + 1.2*np.sin(t*2.3)
    rows = np.column_stack([speed, .1*np.sin(t), .1*np.cos(t),
                            .08*np.sin(t*3), .24*np.cos(t*3), .08*np.cos(t*.5), t>1])
    result = subprocess.run([str(binary)], input='\n'.join(' '.join(map(str,r)) for r in rows),
                            text=True, capture_output=True, check=True)
    expected = np.loadtxt(result.stdout.splitlines())
    import jax
    import jax.numpy as jnp
    cfg = ControllerConfig()
    def scan_step(state, row):
        state, out = controller_step(state, row[:6], row[6]>0, cfg)
        return state, jnp.array([out.steer_rate,out.disturbance,out.equilibrium_shift])
    _, actual = jax.jit(lambda x: jax.lax.scan(scan_step, initial_controller(cfg), x))(jnp.asarray(rows))
    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)

def test_reset_has_no_previous_episode_disturbance():
    cfg=ControllerConfig()
    state=initial_controller(cfg)
    for _ in range(8):
        state,_=controller_step(state,np.array([2.,0.,0.,.2,0.,0.]),True,cfg)
    fresh=initial_controller(cfg)
    _,output=controller_step(fresh,np.array([2.,0.,0.,0.,0.,0.]),True,cfg)
    assert float(output.steer_rate)==0
    assert float(output.disturbance)==0
