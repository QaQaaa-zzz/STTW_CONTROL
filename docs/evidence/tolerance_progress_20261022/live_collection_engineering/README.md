# 12回合实时采样工程证据快照

[研究裁决](DECISION.md)；原始轨迹与全部六张共同XY图保存在工作树的 `runs/live_preference_collector_v1_attempt2/`，不复制大型数组或权重。

`audit_control_authority.py`是原run内脚本的归档副本，其相对路径依赖原run目录，不能直接在本目录执行。原命令为：

```bash
cd /home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration
/home/qy/mujoco_playground/.venv/bin/python runs/live_preference_collector_v1_attempt2/audit_control_authority.py
```

该脚本拒绝覆盖既有结果；本次结果已存在，无需重跑。两个JSON只读复核结果各含全部12例；完整原始NPZ的SHA见权限审计JSON。它们不是训练结果或独立泛化证据。
