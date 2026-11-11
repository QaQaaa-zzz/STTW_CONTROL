# H75单脊/石子公式与接触数据包

用途：对照同一回合实际侧倾、名义模型预测、速度/转向及轮地接触响应。12工况×18车道=216条；两地形、两速度、三物理组，五alpha、完整/仅速度/仅转向、Zero及重复全部保留。

## 文件
- H75_trajectory_formula.zip：全部原始trajectory.npz、contract、指标、XY/奖励图，公式/控制器源码、参数、车辆URDF、参考路径、读取示例。无训练权重；本包用于分析已保存时序，不是从头启动仿真的软件包。
- H75_raw_contacts.zip：共享batch全部1841块5kHz原始接触、关节和刚体状态，0.8–10s，46001个物理时刻。与主包解压到同一目录。
- cases.json：工况本地lane到共享batch lane的映射。不要按lane切片扁平接触缓存！count/start才是每步索引。

## 要的数据与对应字段
1. 实际侧倾phi：trajectory measurement[...,0]；局部IMU角速度为[...,1]，不是一般姿态下的欧拉侧倾导数。measurement其余为转角rad、转角速度rad/s、IMU z角速度、正向后/前轮轴速度rad/s。符号和安装旋转见source/torch_observation.py。
2. 实际前向速度v：lower_trace最后维第2项，reshape(2000,18,11)；实际delta/delta_dot为measurement[...,2:4]。参考见nominal/target/filtered及lower_trace[...,8:10]，不可用参考替代实测。
3. 高频轮地接触：frontwheel/rearwheel_normal_force(N)、normal_direction、normal_point(m)、friction_force(N)、friction_point(m)，各自count/start。只读计数范围，法向标量乘方向得到力矢量；摩擦已是矢量。世界坐标；配套body_state为xyz、xyzw四元数、世界线速度、世界角速度。
4. 转向力矩：motor_torque按index joint_order（steering_joint）为实际施加执行器项，motor_requested_torque两列为转向/后轮请求（限幅前），不含被动弹簧阻尼。关节角/角速度同一物理步记录。
5. 事件与恢复：alive_before、trajectory valid/failure/domain_exit等保留真实有效窗口；接触事件需由接触点/力和地形定义提取。没有预先保存统一的“恢复时刻”标签。最终保持/任务结论见condition_metrics和analysis_summary。
6. 模型与参数：controller_dynamic.json、vehicle.urdf、源码、contract内随机化、惯量/质量、模型身份；主包含原始route.npz。名义方程phi_ddot=a1(v)*delta_dot+a2(v)*delta+a4*phi+d。系数按torch_controller._system及实际配置计算，不硬套固定数值；固定速度推导不等于变速全过程模型。

## 时间与坐标
主trajectory状态为每5ms后步记录。正常完整步时间=(i+1)*0.005；lower_trace[...,6]是该下层步开始时有效标记，[...,7]是该步实际执行物理子步数，末次可能不足25。用示例生成真实末步时间并截断。controller_terms为该下层步前控制计算，状态为后步；作比较要保持这一区别。
高频时间严格取physics_step*0.0002；alive_before保留发生失败的当步。高频与低频不能仅按数组下标对齐。body_state/接触点含全局环境平移，case pose已减origin；画在同一坐标系前须转换。index的body_order/joint_order和cases映射是准绳。

## 当前能算什么
可由原始接触重算前后轮合力、关于指定原点的接触力矩，已有contact_torque_geometry源码算前轮绕转向轴力矩。其结果不等于侧倾力矩；总接触也不等于“地形额外扰动”。
等效d未直接测量：用实测phi二阶导数减名义项得到的是模型总残差（含接触、耦合、建模/差分误差）。不能再把同一残差代回方程吻合原phi当独立验证。独立接触投影需要明确惯量、广义坐标及雅可比映射；本包有原始状态/几何/力，尚未生成该投影或预测曲线。

## 快速读取
解压主包后运行：python read_example.py --case single10_v2.3_seed101 --lane 6
读取高频示例：加 --contacts（须解压原始接触包）。lane6=full_0.5；lane7=speed_only_0.5；lane8=steer_only_0.5；lane12=full_1；以cases.json实际标签为准。
依赖：numpy；附带Torch几何/控制源码仅作公式来源，调用时另需torch与对应相对模块。

## 交付位置

[轨迹与公式包](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_contact_data_delivery/H75_trajectory_formula.zip) · [5kHz原始接触包](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_contact_data_delivery/H75_raw_contacts.zip) · [校验结果](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_contact_data_delivery/CHECKS.json)。大文件仅保留在外置数据盘，Git保存说明和读取脚本；本轮不重跑仿真。

读取示例已实际通过：单脊2.3m/s、seed101、full_0.5提取2000行状态，同车道首帧0.8s前轮法向力约19.98N。原始接触包完成及CRC/SHA256以CHECKS.json为准；该文件未出现时仍在打包校验。
