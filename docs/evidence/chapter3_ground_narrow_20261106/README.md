# 窄路开发挑战：准备完成，尚未执行

8 个预先固定根，复用 G4 的低/高速度、两斜向及两组对角初偏组合。每根同一实际 D0 出口接左右弯各六候选，共 96 后缀，最多 480 万物理 lane 步，0 优化。不是全部 24 根的均匀样本或独立 TEST。

半宽 0.6 m 为任务书建议的开发起点，原 2 m 宽路证据保留。各目录保存原中心线和左右标记边界，地图为标记道路，不添加墙体/接触障碍。控制器、动力学、地形、摩擦与时序均不因窄路评分修改。5000 Hz 物理、200 Hz 底层、50 Hz H75。

中心线任务标签与车轮/车体碰撞顶点边界分别记分。记录 200 Hz 各刚体实际姿态，离线使用 URDF 碰撞几何；该离散顶点证据不等于连续整车体积或 PhysX 实际 SDF 接触面的严格净空保证，正式宽度尚未宣称锁定。

执行顺序：名义 D0 完成且吞吐比较通过后再执行，source_failed 根保留、不替换、不生成虚假快照。已有真实快照由 source_snapshot 显式传入，避免重复产生源轨迹。调用 run_connection_collection.py --manifest 数据盘本目录/manifest.json --group cases --start 指定起点 --count 指定根数 --parallel-tasks --analysis-python /home/qy/miniconda3/envs/mjp/bin/python。该文档不代表已启动。
