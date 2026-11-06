# E5前向绕行地图：未物理执行

复用项目已固定版本的PythonRobotics Dubins适配器，分流点(-2,0)经过(4,-2)、(12.6,-2)、(16.6,0)到共同目标(21.6,0)，半径1.8m。各段位置/航向连续、曲率不连续，仍由原参考滤波执行。保存原路前缀＋支路地图、实际下发支路参考、0.6m半宽标记线及各段规划元数据。

采样中心线x单调，平台x=7.6..11.6区间的中心线到平台侧边净距为1m。该几何检查不是车体包络、动态可跟踪或安全认证。摆车模型、轮子、摩擦、地形不改；地图来自已有scene_L9.6实际网格平台边界。

E5共同终点改到x=21.6，使最后一弯结束后保留5m直行；原J4全任务22秒/终点14.6m的实验与成功标签保持原样。拟先两条强制路线试验（原跳跃/绕行），每条30秒、400Hz物理/200Hz底层/50Hz策略，最多24000物理步，0优化；尚未启动。两路线通过后才比较规则/能力决策，不能把尚未执行的支路当已验证备选。

若任务只要求共同目的地，分别记目标完成；若必须platform_jump，绕行至目的地仅记safe_exit，不能完成必需跳跃。后续使用共用available_forward_routes的分流截止和驻留规则，不加入回头绕圈。

## 执行入口已接通，2026-11-06 18:11 UTC启动

`forward_branch`使用共同原路progress、`available_forward_routes`分流截止和原`begin_connection`接口。切换不重置ESO/滤波/历史；bypass禁用跳跃触发，错过口则保留原路线。事件记录请求路线、实际路线、截止原因和耗时。新E5地图仅把原路目标位置改为共享21.6m，上游和原路径点列不变。

分析新增`chain_summary.json.forward_task`，独立报告共同目标到达、地图走廊、满窗、是否完成平台技能、optional/mandatory任务完成及绕行退出。原J4标签继续单列，绘图同时显示原路、支路和共同目标。两条强制路线已按manifest顺序启动，尚无物理结果；不能称在线能力规划已验证。

### Remaining-task revision preserves required skills

`ContinuousTask.revise_remaining` previously preserved goal IDs but could accept
a rewritten goal that omitted its `required_skills`. It now rejects removal of
an outstanding skill requirement before mutating task state. Route/order changes
remain possible; execution reference and completion evidence remain unchanged.
This closes a concrete task-semantics gap: a mandatory platform goal cannot become
an optional destination merely by revising the remaining program. Optional-route
experiments and their separate mandatory/optional outcome labels are unchanged.
Fourteen task-manager/route-selector CPU tests pass, including a rejected revision
that keeps the goal ID but removes `platform_jump`. No new physical run; this is
an interface fix, not additional evidence of closed-loop task performance.

## 多根在线规则／能力比较入口

`learning/cli/prepare_online_route_comparison.py` 复用现有 `ForwardRouteSelector`、`ContinuousTask` 和异步执行器。输入 forced original/bypass 的 manifest，只取每个配对中的 original 配置，各生成规则和能力两个在线版本；共享初态、扰动、决策点、地图、.5s查询期限及400/200/50Hz时序。能力模型固定现有核参数，复制证据后按整根排除，不调参。主任务允许绕行；必须跳台的结果另行统计，绕行不能计为该任务完成。

当前6训练根×D0/Fy±2N×2方法为36条，最多432000物理步、0优化。当前仅完成入口与CPU接线检查：18对配置除规划器/输出外一致，实际历史12条可用记录能通过真实查询接口返回支路；14项既有任务/路由测试通过。临时配置 `/tmp/ch3_online_route_wiring` 仅用于接线，使用历史单根数据，不是最终冻结证据，也没有运行仿真。

正式比较须在当前 `chapter3_forward_route_roots` 36条完整采集后，先用 `export_forward_route_records.py` 导出完整标签，再运行下列准备命令；不把未完成的六根采集或临时配置称为在线结果：

```bash
PYTHONPATH=learning/src python learning/cli/prepare_online_route_comparison.py \
  --manifest "$DATA/chapter3_forward_route_roots/manifest.json" \
  --records "$DATA/chapter3_forward_route_roots/route_records/records.jsonl" \
  --rule-config "$DATA/chapter3_async_route_probe/planner.json" \
  --output "$DATA/chapter3_online_route_comparison"
```

`DATA=/media/qy/6A7E3DB07E3D75C3/STTW_runs`。冻结后用既有 `run_jump_approach_pilot.py` 执行生成的 manifest，复用释放出的GPU名额；此次没有新增并发、没有启动该36条。最终必须按根比较原路完成、绕行完成、必须跳台完成、误放行/未知、查询延迟及实际XY，不以本入口测试代替任务收益。
