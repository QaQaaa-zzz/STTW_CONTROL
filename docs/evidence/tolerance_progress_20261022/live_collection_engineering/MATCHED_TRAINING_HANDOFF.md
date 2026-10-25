# 同合同普通PPO：执行与固定终点交接

本文件只保存工程入口；当前事实、采用与下一步以共享PROJECT_STATE.md为准。

训练代码commit：a410111。原任务设计见 `docs/superpowers/plans/2026-10-25-matched-live-ppo-design.md`；真实run为 `runs/matched_live_ppo_v1`，完整参数在其plan及effective_training_contract内。source_at_launch保留全部启动源码，后续新增评价准备器不修改训练封存文件。

已实际核验第一轮120回合/两臂各60、无未知；两臂各4次Critic更新且Actor0更新。真实reward scalar通过HTTP加载在 http://localhost:34359 ，收据 `tensorboard_verified.json`。这只是训练可见性与on-policy执行证据，不是最终方法收益。

进度只读命令：

```bash
cat /home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/matched_live_ppo_v1/status.json
```

记录的主进程PID2778238、exec session21600；实际监测先核/proc身份与当前status，不能仅凭旧PID/文件判断仍运行。`monitoring.json`为900秒轮询或真实终止；不凭观察超时重启。Bark沿既有项目登记，email/autofix关闭。

## 固定评价准备

新 `learning/src/sttw_control/live_ppo_evaluation.py`只在整个64轮配对训练完成后，核对固定终点/评价表SHA并准备两个路线子run，使用原确定性参考执行器。接口为：

```python
from sttw_control.live_ppo_evaluation import prepare
prepare('runs/matched_live_ppo_v1', 'runs/matched_live_ppo_v1_fixed64_evaluation')
```

不能现在执行，不接受中间checkpoint/best替代。此prepare函数尚未针对真实终点运行，因为权重仍未训练完成。case适配测试通过，不等价于评价GPU验证。

准备后每路100条：原Actor/profile0、原Actor/profile7、新plain/profile0、新support/profile7。使用原封存200条DEV表，alpha、力方向、时长及起扰时刻在四角色之间一致；不同alpha各自抽取时刻，不能据此单独作严格的alpha抗扰因果对比。无扰每route/alpha/role仅一条。训练与评价来自旧几何/相关源初态，不称最终独立泛化。

每路先运行 `evaluate_anticipatory_reference.py --phase shape --stage reentry`，完成后才 `--phase execute --stage reentry`；固定stage不能重复执行，错误保存并停。全评价最大400000实际5ms步、0optimizer，启动前仍须独立台账及Bark核对。当前没有评价执行。

## 待完成分析

复用 `geometric_task_trace`、`task_metrics`、`cost_accounting`及原alpha奖励重评分，输出全部四角色共同XY、有符号逐20ms奖励和累计量，保留接触真实终点、未知及任务失败。旧 `anticipatory_analysis.analyze` 的XY只支持两alpha，不能盲目调用后宣称五alpha交付完成；需复用已有434分析的五alpha绘图结构。

主要比较：trained_support 对 parent_support（支持适应），trained_plain 对 parent_plain（普通续训），trained_support 对 trained_plain（共同训练后的支持取舍）。同时报告确定性名义成本和受扰物理/联合任务；原task64/source保留。一个配对训练重复，不能以200相关回合替代多个独立训练种子。
