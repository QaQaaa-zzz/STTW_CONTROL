from pathlib import Path
import subprocess

def test_cpp_residual_contract(tmp_path):
    root=Path(__file__).resolve().parents[2]
    subprocess.run(['g++','-std=c++17','-I'+str(root/'mujoco_ros_ws20260313/mujoco_ros/include/mujoco_ros'),
                    str(Path(__file__).with_name('command_reference.cpp')),'-o',str(tmp_path/'command')],check=True)
    subprocess.run([str(tmp_path/'command')],check=True)
