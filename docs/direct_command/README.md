# STTW直接指令网络 V3 执行包

交付内容：
- STTW_Codex_Direct_Command_V3.md：完整实施规格（首读）。
- STTW_Direct_Command_V3.json：参数、输入字段、工况与预算。
- STTW_Direct_Command_V3_reference.py：纯数学/映射/奖励/预算校验，不是环境或完整控制器。
- numerical_check_results.json：本次实际执行的纯数学校验结果。

默认只实施一个输入alpha、输出速度/转向参考修正的共享网络；不实现参数网络、解析alpha分配器、候选搜索、世界模型或扩散。20更新pilot后停止，不自动扩大训练。

校验命令：
```bash
python STTW_Direct_Command_V3_reference.py
```

已完成的是文档/配置与纯函数数值检查。尚未导入用户仓库、运行车辆仿真、训练或验证效果。参数均为首轮设计值，没有保证必然收敛、不摔或泛化。
