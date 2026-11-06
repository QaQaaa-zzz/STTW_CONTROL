# 当前 STTW 双轮车物理模型

这是当前 Isaac Lab 越野评测实际使用的小车模型，对应原 H75 与用户选定橙色 Local300。不是 Go2，也不是后续带摆双轮车。

- [vehicle.urdf](vehicle.urdf)：从 `runs/isaac_urdf_drive/vehicle.urdf` 原字节复制。
- [轮子 OBJ](ellipsoid_0.1_0.1_0.025.obj)：URDF 引用的完整网格依赖，原字节复制。
- [contract.json](contract.json)：质量、完整惯量、关节、初始姿态、源执行器参数及转换差异。
- [runtime.json](runtime.json)：当前评测的物理选项及可移植资产路径；[controller.json](controller.json) 为实际 ECBC/ESO 配置。
- [manifest.json](manifest.json)：文件身份及对应评测入口。

## 原 MuJoCo 模型与转换

[原始 MJCF](../../mujoco_ros_ws20260313/model/scalebike_scene_matlab.xml)及[六个 STL](../../mujoco_ros_ws20260313/model/meshes/)已经随同一分支版本管理；实读哈希均与 URDF 导出契约一致，不以同名文件替代原模型。MuJoCo 学习侧使用[model.py](../../learning/src/sttw_control/model.py)处理未使用的 torus 资源和启动伺服，按该入口加载。原始 MJCF、学习侧 portable XML 与 Isaac URDF 是三个不同表示，不能混称字节相同。

转换代码：[isaac_urdf.py](../../learning/src/sttw_control/isaac_urdf.py)、[isaac_asset.py](../../learning/src/sttw_control/isaac_asset.py)。Isaac 运行时通过[isaaclab_vector_env.py](../../learning/src/sttw_control/isaaclab_vector_env.py)转换为本机 USD 并施加物理参数；USD 缓存不作为模型源上传。

## 必须与模型一起使用的运行参数

物理步长0.0002s，底层控制0.005s，H75推理0.02s。当前轮子采用SDF碰撞（分辨率256），摩擦0.8，friction_offset_threshold=0。前轮柔顺接触刚度5324.4083、阻尼112.3858；后轮7833.3812、173.1104。具体数值以runtime.json及已提交运行代码为准。只导入URDF使用默认材质，不能声称复现了这些结果。

URDF的effort/velocity=1000是导入元数据，不是执行器真实能力。实际伺服和被动关节力使用[TorchVelocityServos](../../learning/src/sttw_control/torch_controller.py)及contract.json中的源参数。转向接口为角速度，后轮接口为轮轴转速。

此目录提供物理模型与ECBC配置；H75、Local300策略权重不属于本次“小车模型”打包。已有证据包中的策略身份不变。训练/评测仍需要相应权重、初态、地形和Isaac Lab环境；此模型包不代表完整训练环境已经独立安装。
