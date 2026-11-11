# eta=0.533来源复算

旧2cm/4cm、2.6m/s等权平均RMSE端点：alpha0的(Ey,Ev)=(0.06361200660467148m,0.1724538877606392m/s)，alpha1为(0.24631942808628082m,0.07506746798753738m/s)。

eta=(Ev0-Ev1)/(Ey1-Ey0)=0.0973864197731018/0.18270742148160934=0.5330184126259171 s^-1。

该值与原task_cost_check.json的eta_endpoint_normalization完全一致。来源：sttw-offroad/docs/evidence/sttw_offroad_20261106/curve/tradeoff_shape_analysis.json及physics_checks/task_cost_check.json。它是这两个已见工况端点变化范围的归一化，不是车辆常数或独立任务偏好；历史脚本将其舍入为默认.533。数值来源已复算，不能用同批数据的权重吻合当独立验证。
