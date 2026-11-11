# S4 新布局验证

22工况×2新布局×4新物理组×12车道=2112条，10s/条。五固定alpha、Zero、三个任务的机制选档、三个任务的已有简单规则。固定原H75/Local300和共同起点，0训练。

两类选择器在运行前确定alpha，回合内固定。选择只用地形类型/高度、请求速度、任务容差；不使用新布局结果，当前状态与路径几何保持共同配置。

[frozen_decisions.json](frozen_decisions.json) · [protocol.json](protocol.json) · [实际XY（运行后）](RESULTS.md)
