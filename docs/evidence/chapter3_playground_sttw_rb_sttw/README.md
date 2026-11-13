# STTW → RB → STTW：GPU串联测试

![三条路线实际与参考XY](xy.png)

**地图更正：** 旧批次实际碎石区侵入沟前两米缓冲0.2m、沟后缓冲0.6m，顶层平地说明未落实到生成参数。当前串联配置已改为沟前x=8.3–10.3m、沟后x=10.6–12.6m全宽平地，碰撞网格检查通过；下文历史结果不属于修正地图验收。新批次：`flat_buffers`。

使用冻结RB Run3 transition_128000（actor ffe98cc0c497），原任务250回合复测246/250。该历史成功率不代表本地形串联成功率。

本次三条路线60/61/62同批运行；GPU MJX-Warp物理、GPU H75/ECBC/ESO、GPU原生RB推理，DLPack共享张量，循环内不下载物理状态、观测、动作、接触或轨迹。阶段掩码、失败、回交及末端保持在GPU处理；结束后统一保存200Hz轨迹并按真实失败终点截断。失败lane状态冻结，固定10秒批次窗口；无训练。

## 实测耗时

GPU批次总时间36.107秒，初始化1.201秒，循环34.820秒；三个世界、每世界固定2000个5ms控制tick。旧逐条且接触已修正版循环55.053秒、含初始化/输出的阶段时间56.360秒。新批次不会因某条提前失败而缩短其他lane。两次接触动力学轨迹不同，不能当作逐位等价的严格性能基准，也不将三环境结果外推4096环境吞吐；未声称全GPU模式已经改善落地能力。

| 路线 | RB接管/s | 首次真实远岸接触/s | 失败/s | 回交STTW | 最终目标 |
|---|---:|---:|---:|---|---|
|60|4.060|5.090|5.115|否|失败|
|61|4.060|5.105|5.460|否|失败|
|62|4.060|5.095|5.480|否|失败|

![阶段、速度、侧倾和摆角](stages_tracking.png)

采用已验收的STTW：1.8m/s、alpha0、锁髋膝、implicitfast1ms×5；RB释放髋膝、原生位置执行器、RK4 5ms×1、20ms策略；真实远岸轮地接触后延迟0.5秒才回交。原图最高脊坡4cm，其余沟/桥/碎石布局保持；质量、惯量、力矩上限未改。三条均在0.5秒回交等待完成之前触发物理失败。后续问题在落地与回交阶段；本次不更改等待、控制或成功阈值来制造通过。

## 接触记录更正

安装的MJX-Warp `_get_data_into_warp` 明确跳过contact复制。此前evaluation及run3_native_rb_retry调用mjx.get_data读取CPU接触，故其“无远岸接触、无回交”字段无效，也不能作为正确回交闭环的能力结论。原始状态保留。现在直接使用GPU contact__geom/dist/pos/worldid与nacon有效掩码判断，排除轮自接触；已完成的compact_corrected阶段35处独立NumPy归约一致，后续入口已删除这个重复校验选项。

## 复现

```bash
PYTHONPATH=learning/src:/home/qy/DVGC/JIT/src /home/qy/mujoco_playground/.venv/bin/python learning/cli/evaluate_playground_handoff.py --config learning/configs/playground_sttw_rb_sttw.json --output /path/to/new-output --skills rb_run3 --modes h75 --routes 60 61 62 --seconds 10 --physics-substeps 5
```

重型任务仍使用项目共享GPU锁。原始目录：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/`；当前批次gpu_batch_eval，保留此前失败启动及原始对照。当前源码清除串联CPU入口，复用既有Torch控制和逐lane独立时钟/掩码重置；只有初始化文件读取和最终导出在CPU。
