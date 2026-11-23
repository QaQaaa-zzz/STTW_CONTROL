# H75 首次策略更新的实测字段

[逐车道逐字段原值、差值和单位](first_policy_step.csv) · [三个工况与两组物理参数的 XY 图](README.md)

原 672 批与先前 36 批的第 1 次回放按相同物理组、参考、α、Zero 身份及绝对车道坐标匹配。两批轨迹均缺少实际 0 ms H75 observation（351 分量）、policy_measurement 及未压缩的原始 latent。配置显式省略前两项，评测写盘列表排除了 result.action；Zero 在策略输出后被掩蔽，所以其 target 无法反推该车道网络原始输出。相同初态文件约束初值，但不能据此证明策略输入在 GPU 上逐位相同。

0 ms 首次执行 middle.before → 生成 latent → tanh 并按通道缩放成 target → 过滤、限速和限幅，接着生成底层命令，然后进行 25 个 0.2 ms 物理子步。第 1 条 5 ms 诊断记录持有该周期开始前的 target、修正和命令，与 5 ms 的 post_pose、measurement 同行保存。实测两批 target 在第 1 个 20 ms 策略周期的四个 5 ms 记录里各自保持相同。日志 nominal 在 middle.after 后保存，不能作为 0 ms 完整输入。

|字段|变化车道数 / 36|最大绝对差|单位|
|---|---:|---:|---|
|target_speed|18/36|1.49011612e-08|m/s|
|target_steer|25/36|3.7252903e-09|rad|
|filtered_speed|15/36|6.98491931e-10|m/s|
|filtered_steer|25/36|2.32830644e-10|rad|
|delivered_offset_speed|0/36|0|m/s|
|delivered_offset_steer|25/36|2.32830644e-10|rad|
|governed_speed|0/36|0|m/s|
|governed_steer|25/36|2.32830644e-10|rad|
|lower_residual_steer_rate|28/36|3.35276127e-08|rad/s|
|lower_residual_rear_axis_speed|17/36|3.57627869e-07|rad/s|
|command_steer_rate|28/36|3.35276127e-08|rad/s|
|command_rear_axis_speed|9/36|1.90734863e-06|rad/s|
|post_x|0/36|0|m|
|post_y|0/36|0|m|
|post_yaw|0/36|0|rad|

target 两通道任一发生数值变化的为 25/30 条 H75，Zero 的两路目标均为零（6/6）。两路 target 本身早于这一周期的物理积分；现有字段不能把最初的数值差异进一步定位到输入构造或网络矩阵运算。

冻结代码（3d90f818）

- `learning/src/sttw_control/torch_middle_control.py:32`
- `learning/src/sttw_control/torch_middle_control.py:41`
- `learning/src/sttw_control/torch_middle_control.py:92`
- `learning/src/sttw_control/torch_path_policy.py:50`
- `learning/src/sttw_control/isaac_path_rollout.py:100`
- `learning/src/sttw_control/isaac_path_rollout.py:133`
- `learning/cli/evaluate_isaac_gpu.py:133`
