# Path-feedback 主线推进执行包

执行入口：`CODEX_Path_Coverage_Mainline_V2.md`。
`path_coverage_iteration_v2.json`是明确的执行规格，不是可直接传给旧CLI的原生训练配置；需按指令生成版本化resolved run/config。
`geometry_budget_check.py`和`arithmetic_check.json`仅用于代数、预算与名义几何覆盖核验，不是物理仿真或已验证训练效果。
`review_summary.json`给出已核对证据范围。

新训练不是已执行工作。保留local300、原物理、网络、奖励和20秒任务，只改变采样覆盖，从各自已验证best分叉，旧best不覆盖。
