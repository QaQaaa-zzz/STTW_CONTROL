# H75进场偏好交叉实验（运行中）

目的：在相同物理参数下分开进场偏好与后续偏好的贡献。原H75+Local300冻结，沿用原72记录的两地形、两速度和101–103物理参数。每个来源以alpha=.5或1行驶至s2.5m，分叉比较未恢复继续、恢复原档、恢复.5、恢复1和Zero。切换在障碍前，后续接触按实际事件对齐。

24组五分支，每回合最多10秒，600万physics lane steps、0优化。状态恢复覆盖ESO、滤波、网络历史与动作/传感延迟；保留未恢复对照检查PhysX缓存影响。原生失败保留，恢复前失败单列，不补样。工程错误停机。源模型/资产身份随每组contract.json保存。

运行目录：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_alpha_crossover`；入口：`learning/cli/run_alpha_mechanism.py --run <运行目录>`。本目录protocol/jobs/逐组配置为冻结输入副本，包含本机资产路径；运行前须映射真实资产位置。结果PENDING，XY和奖励索引由运行脚本生成RESULTS.md。

旧20组机制数据另外用`learning/cli/analyze_alpha_mechanism.py --run <h75_alpha_mechanism>`做CPU流式复算，输出mechanism_analysis目录的逐分支/接触段JSON及0.2ms时序。法向冲量含承重，不称净扰动；入带延迟按下次接触前连续200ms路径误差≤0.3m统计。

验证：两个入口及分析脚本语法检查通过；安装环境env_isaaclab下snapshot延迟历史/未触及分支行为测试1 passed。首组启动后实际物理进度由日志核验。mjp环境缺少pytest和torch，测试及CPU分析改用已有env_isaaclab，未安装新依赖。

后续：先回收交叉结果和机制量，再冻结带Zero强度扫描及独立选档TEST配置。不会把本轮熟悉布局DEV称独立TEST。
