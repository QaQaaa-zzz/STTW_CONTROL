# 同状态同动作采样对照

基准源码c422c63；本轮只改短回合reset/观测与诊断累计，物理、种子、奖励不变。生产训练进程未切换。

`gpu_equivalence_and_timing.json`为8环境、固定动作、共享GPU的单步微基准，分别检查普通步和4/8环境片段超时的混合重置。每种情况各版计时2次，不能外推512×128整轮速度。241个返回叶均检查有限性，离散叶完全一致；所有浮点叶并非逐位相同。

先前两次验证失败保存在`preserved_benchmark_attempts.json`，包括fixture类身份错误与过严全树精度阈值，不覆盖原始运行目录。成功尝试按位姿2e-5、速度2e-4、观测2e-4、奖励2e-5、冲量1e-5绝对差检查；这些界限仅为本次工程核对，不是新增任务成功门槛。全树混合单位最大差0.007019，控制相关量单列。

复现脚本为实际执行的`benchmark_sampling.py`。在原worktree、相同venv与bank下，先执行 `git show c422c63:learning/src/sttw_control/short_recovery.py > /tmp/sttw_reference_short_recovery.py`。脚本输出目录是不可变尝试名，重新执行需换新目录名。设置 `JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=learning/src`；无PPO更新，脚本复用Bark-only注册。不会启动/恢复生产训练。

计时约0.79s/步，两版没有可辨别加速；此实验不含完整Torch actor/rollout学习流程，设备统计的主机回传收益仍需完整采样测量。优化未改变物理条件，但图重排存在数值尾差，不承诺完整随机训练逐位重放。
