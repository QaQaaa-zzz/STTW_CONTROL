# 整车支持误差与滚动历史复测

固定已选存档；M0同数据ARX。各集合条件数量见表格，均为既有DEV身份。所有预测只使用窗口入口真实历史，之后自由展开；无新物理成功率。

|集合/模型|时域s|窗/例|均值XY m|P95 XY m|最差XY m|最差整车XYZ支持m|
|---|---:|---|---:|---:|---:|---:|
|rolling_M0|0.1|36/2|0.00274|0.00594|0.00846|0.01324|
|rolling_M0|0.3|34/2|0.01260|0.02369|0.02675|0.02983|
|rolling_M0|1.0|28/2|0.05997|0.09400|0.11556|0.11588|
|rolling_M0|2.0|18/2|0.11167|0.17751|0.18689|0.18856|
|rolling_M0|3.0|8/2|0.17028|0.20002|0.20136|0.19410|
|rolling_direct_seed0|0.1|36/2|0.00181|0.00615|0.00668|0.01565|
|rolling_direct_seed0|0.3|34/2|0.00603|0.01572|0.02013|0.02884|
|rolling_direct_seed0|1.0|28/2|0.01721|0.03148|0.03590|0.03767|
|rolling_direct_seed0|2.0|18/2|0.02731|0.04130|0.04578|0.04412|
|rolling_direct_seed0|3.0|8/2|0.03334|0.05132|0.05431|0.05211|
|rolling_direct_seed1|0.1|36/2|0.00182|0.00512|0.00643|0.01488|
|rolling_direct_seed1|0.3|34/2|0.00654|0.01559|0.02006|0.02956|
|rolling_direct_seed1|1.0|28/2|0.02056|0.03538|0.03911|0.04523|
|rolling_direct_seed1|2.0|18/2|0.03239|0.05047|0.05620|0.05865|
|rolling_direct_seed1|3.0|8/2|0.03818|0.05426|0.05655|0.05469|
|rolling_direct_seed2|0.1|36/2|0.00184|0.00684|0.00719|0.01467|
|rolling_direct_seed2|0.3|34/2|0.00689|0.01511|0.01664|0.02458|
|rolling_direct_seed2|1.0|28/2|0.01742|0.03822|0.04344|0.04580|
|rolling_direct_seed2|2.0|18/2|0.03321|0.05145|0.05551|0.05149|
|rolling_direct_seed2|3.0|8/2|0.03598|0.05710|0.06335|0.05115|
|rolling_residual_seed0|0.1|36/2|0.00184|0.00517|0.00807|0.01166|
|rolling_residual_seed0|0.3|34/2|0.00598|0.01517|0.02123|0.02266|
|rolling_residual_seed0|1.0|28/2|0.01339|0.02074|0.02676|0.02646|
|rolling_residual_seed0|2.0|18/2|0.02063|0.03482|0.05040|0.03976|
|rolling_residual_seed0|3.0|8/2|0.02484|0.04352|0.05100|0.04914|
|rolling_residual_seed1|0.1|36/2|0.00196|0.00492|0.00712|0.01211|
|rolling_residual_seed1|0.3|34/2|0.00674|0.01455|0.01871|0.02061|
|rolling_residual_seed1|1.0|28/2|0.01485|0.02218|0.02582|0.02938|
|rolling_residual_seed1|2.0|18/2|0.02183|0.03716|0.04167|0.03731|
|rolling_residual_seed1|3.0|8/2|0.02583|0.04072|0.04307|0.04407|
|rolling_residual_seed2|0.1|36/2|0.00189|0.00473|0.00769|0.01047|
|rolling_residual_seed2|0.3|34/2|0.00615|0.01395|0.01849|0.01969|
|rolling_residual_seed2|1.0|28/2|0.01363|0.02132|0.03367|0.04026|
|rolling_residual_seed2|2.0|18/2|0.02533|0.03951|0.05432|0.04656|
|rolling_residual_seed2|3.0|8/2|0.02956|0.04841|0.05508|0.05087|
|static18_M0|0.1|331/18|0.00303|0.00894|0.01671|0.01903|
|static18_M0|0.3|313/18|0.01367|0.03597|0.06440|0.07236|
|static18_M0|1.0|247/18|0.07038|0.15836|0.40415|0.40332|
|static18_M0|2.0|157/18|0.17488|0.34080|0.62235|0.51162|
|static18_M0|3.0|67/18|0.27121|0.51137|0.62465|0.56383|
|static18_direct_seed0|0.1|331/18|0.00255|0.00723|0.01112|0.02045|
|static18_direct_seed0|0.3|313/18|0.00835|0.02013|0.04457|0.06269|
|static18_direct_seed0|1.0|247/18|0.02730|0.05727|0.28742|0.30631|
|static18_direct_seed0|2.0|157/18|0.04486|0.11659|0.42041|0.37224|
|static18_direct_seed0|3.0|67/18|0.05962|0.22918|0.42041|0.37224|
|static18_direct_seed1|0.1|331/18|0.00258|0.00721|0.01185|0.01988|
|static18_direct_seed1|0.3|313/18|0.00851|0.02236|0.03863|0.06410|
|static18_direct_seed1|1.0|247/18|0.02810|0.06577|0.24161|0.25674|
|static18_direct_seed1|2.0|157/18|0.04316|0.09935|0.33760|0.29447|
|static18_direct_seed1|3.0|67/18|0.05833|0.20650|0.33760|0.31350|
|static18_direct_seed2|0.1|331/18|0.00264|0.00753|0.01146|0.01947|
|static18_direct_seed2|0.3|313/18|0.00887|0.02269|0.04247|0.07228|
|static18_direct_seed2|1.0|247/18|0.02864|0.06783|0.33996|0.34776|
|static18_direct_seed2|2.0|157/18|0.04877|0.13507|0.47001|0.35728|
|static18_direct_seed2|3.0|67/18|0.07339|0.25279|0.47734|0.48478|
|static18_residual_seed0|0.1|331/18|0.00247|0.00658|0.01203|0.01420|
|static18_residual_seed0|0.3|313/18|0.00787|0.02080|0.04969|0.07767|
|static18_residual_seed0|1.0|247/18|0.02180|0.05109|0.23668|0.26359|
|static18_residual_seed0|2.0|157/18|0.03780|0.08255|0.43156|0.37984|
|static18_residual_seed0|3.0|67/18|0.05605|0.32705|0.43156|0.37984|
|static18_residual_seed1|0.1|331/18|0.00246|0.00681|0.01310|0.01347|
|static18_residual_seed1|0.3|313/18|0.00818|0.02185|0.04910|0.07681|
|static18_residual_seed1|1.0|247/18|0.02235|0.04786|0.17187|0.18331|
|static18_residual_seed1|2.0|157/18|0.03784|0.07658|0.35680|0.34364|
|static18_residual_seed1|3.0|67/18|0.05288|0.28219|0.38512|0.35624|
|static18_residual_seed2|0.1|331/18|0.00247|0.00673|0.01280|0.01384|
|static18_residual_seed2|0.3|313/18|0.00796|0.02009|0.04810|0.07526|
|static18_residual_seed2|1.0|247/18|0.02086|0.04567|0.20987|0.22741|
|static18_residual_seed2|2.0|157/18|0.03741|0.07793|0.36288|0.33604|
|static18_residual_seed2|3.0|67/18|0.05438|0.31256|0.39070|0.36152|
|static19_M0|0.1|314/18|0.00311|0.00981|0.01312|0.01859|
|static19_M0|0.3|296/18|0.01449|0.03202|0.05482|0.05583|
|static19_M0|1.0|235/18|0.07236|0.15250|0.22524|0.24230|
|static19_M0|2.0|145/18|0.17110|0.30716|0.35102|0.37234|
|static19_M0|3.0|55/18|0.22828|0.38902|0.39643|0.40074|
|static19_direct_seed0|0.1|314/18|0.00258|0.00718|0.01932|0.02798|
|static19_direct_seed0|0.3|296/18|0.00810|0.02004|0.03107|0.03943|
|static19_direct_seed0|1.0|235/18|0.02170|0.03938|0.09097|0.08513|
|static19_direct_seed0|2.0|145/18|0.03128|0.06189|0.11226|0.10560|
|static19_direct_seed0|3.0|55/18|0.03740|0.06928|0.11226|0.10560|
|static19_direct_seed1|0.1|314/18|0.00267|0.00760|0.02076|0.02857|
|static19_direct_seed1|0.3|296/18|0.00832|0.02300|0.03659|0.03900|
|static19_direct_seed1|1.0|235/18|0.02223|0.04698|0.11247|0.11496|
|static19_direct_seed1|2.0|145/18|0.03188|0.06654|0.12034|0.12407|
|static19_direct_seed1|3.0|55/18|0.04185|0.07484|0.12650|0.12407|
|static19_direct_seed2|0.1|314/18|0.00276|0.00787|0.02061|0.02798|
|static19_direct_seed2|0.3|296/18|0.00880|0.02208|0.03439|0.04167|
|static19_direct_seed2|1.0|235/18|0.02319|0.04787|0.12202|0.12458|
|static19_direct_seed2|2.0|145/18|0.03675|0.08258|0.16608|0.16238|
|static19_direct_seed2|3.0|55/18|0.05381|0.10100|0.19060|0.19526|
|static19_residual_seed0|0.1|314/18|0.00232|0.00618|0.01054|0.01522|
|static19_residual_seed0|0.3|296/18|0.00695|0.01623|0.02870|0.03035|
|static19_residual_seed0|1.0|235/18|0.01578|0.03039|0.05491|0.05077|
|static19_residual_seed0|2.0|145/18|0.02467|0.04755|0.09127|0.08209|
|static19_residual_seed0|3.0|55/18|0.03050|0.05997|0.09127|0.08209|
|static19_residual_seed1|0.1|314/18|0.00229|0.00602|0.00988|0.01439|
|static19_residual_seed1|0.3|296/18|0.00690|0.01677|0.02878|0.03568|
|static19_residual_seed1|1.0|235/18|0.01595|0.03081|0.05197|0.04525|
|static19_residual_seed1|2.0|145/18|0.02329|0.04403|0.06860|0.06199|
|static19_residual_seed1|3.0|55/18|0.02804|0.04922|0.06860|0.06584|
|static19_residual_seed2|0.1|314/18|0.00230|0.00627|0.01018|0.01446|
|static19_residual_seed2|0.3|296/18|0.00698|0.01646|0.03104|0.03422|
|static19_residual_seed2|1.0|235/18|0.01549|0.02994|0.05495|0.04870|
|static19_residual_seed2|2.0|145/18|0.02518|0.04492|0.08385|0.06968|
|static19_residual_seed2|3.0|55/18|0.03176|0.05858|0.08385|0.07029|
|switch18to19_M0|0.1|314/18|0.00317|0.00857|0.01433|0.01491|
|switch18to19_M0|0.3|296/18|0.01489|0.03149|0.06100|0.06179|
|switch18to19_M0|1.0|238/18|0.07327|0.14807|0.21415|0.23233|
|switch18to19_M0|2.0|148/18|0.17108|0.30189|0.34236|0.37189|
|switch18to19_M0|3.0|58/18|0.23303|0.38310|0.38933|0.39688|
|switch18to19_direct_seed0|0.1|314/18|0.00251|0.00687|0.01636|0.02499|
|switch18to19_direct_seed0|0.3|296/18|0.00834|0.02029|0.03850|0.04169|
|switch18to19_direct_seed0|1.0|238/18|0.02570|0.05427|0.09200|0.09814|
|switch18to19_direct_seed0|2.0|148/18|0.04054|0.09210|0.10619|0.11161|
|switch18to19_direct_seed0|3.0|58/18|0.04723|0.09654|0.10332|0.10731|
|switch18to19_direct_seed1|0.1|314/18|0.00259|0.00680|0.01767|0.02522|
|switch18to19_direct_seed1|0.3|296/18|0.00869|0.02211|0.04552|0.04743|
|switch18to19_direct_seed1|1.0|238/18|0.02475|0.05205|0.10788|0.11059|
|switch18to19_direct_seed1|2.0|148/18|0.03657|0.07819|0.11649|0.12613|
|switch18to19_direct_seed1|3.0|58/18|0.04739|0.10869|0.11649|0.12613|
|switch18to19_direct_seed2|0.1|314/18|0.00259|0.00675|0.01812|0.02531|
|switch18to19_direct_seed2|0.3|296/18|0.00874|0.02093|0.04240|0.04808|
|switch18to19_direct_seed2|1.0|238/18|0.02523|0.05326|0.12097|0.11205|
|switch18to19_direct_seed2|2.0|148/18|0.04324|0.09932|0.14436|0.14411|
|switch18to19_direct_seed2|3.0|58/18|0.06229|0.13079|0.16601|0.16679|
|switch18to19_residual_seed0|0.1|314/18|0.00236|0.00638|0.01333|0.01456|
|switch18to19_residual_seed0|0.3|296/18|0.00767|0.01967|0.03693|0.03929|
|switch18to19_residual_seed0|1.0|238/18|0.02011|0.04195|0.07591|0.08238|
|switch18to19_residual_seed0|2.0|148/18|0.03457|0.06913|0.10669|0.10748|
|switch18to19_residual_seed0|3.0|58/18|0.03959|0.07629|0.13292|0.11893|
|switch18to19_residual_seed1|0.1|314/18|0.00231|0.00615|0.01327|0.01377|
|switch18to19_residual_seed1|0.3|296/18|0.00759|0.01962|0.03702|0.04339|
|switch18to19_residual_seed1|1.0|238/18|0.02007|0.03959|0.08408|0.09120|
|switch18to19_residual_seed1|2.0|148/18|0.03389|0.05966|0.09497|0.09735|
|switch18to19_residual_seed1|3.0|58/18|0.03849|0.06934|0.11013|0.09748|
|switch18to19_residual_seed2|0.1|314/18|0.00237|0.00619|0.01261|0.01495|
|switch18to19_residual_seed2|0.3|296/18|0.00768|0.01860|0.03846|0.04435|
|switch18to19_residual_seed2|1.0|238/18|0.01991|0.04067|0.08492|0.09038|
|switch18to19_residual_seed2|2.0|148/18|0.03498|0.06713|0.09563|0.09406|
|switch18to19_residual_seed2|3.0|58/18|0.03976|0.07596|0.11080|0.09986|
|switch19to18_M0|0.1|331/18|0.00292|0.00836|0.01519|0.01565|
|switch19to18_M0|0.3|313/18|0.01347|0.03820|0.06115|0.06236|
|switch19to18_M0|1.0|246/18|0.06825|0.15990|0.20820|0.20009|
|switch19to18_M0|2.0|156/18|0.16184|0.30544|0.35108|0.33280|
|switch19to18_M0|3.0|66/18|0.24760|0.37128|0.39371|0.37780|
|switch19to18_direct_seed0|0.1|331/18|0.00229|0.00672|0.01076|0.02322|
|switch19to18_direct_seed0|0.3|313/18|0.00708|0.01739|0.03322|0.03580|
|switch19to18_direct_seed0|1.0|246/18|0.01933|0.03771|0.07074|0.06025|
|switch19to18_direct_seed0|2.0|156/18|0.03097|0.05514|0.12573|0.12088|
|switch19to18_direct_seed0|3.0|66/18|0.03528|0.06295|0.07912|0.07641|
|switch19to18_direct_seed1|0.1|331/18|0.00242|0.00702|0.01118|0.02381|
|switch19to18_direct_seed1|0.3|313/18|0.00765|0.01881|0.03933|0.04089|
|switch19to18_direct_seed1|1.0|246/18|0.02155|0.03945|0.08976|0.08093|
|switch19to18_direct_seed1|2.0|156/18|0.03292|0.05512|0.09280|0.08644|
|switch19to18_direct_seed1|3.0|66/18|0.03745|0.07078|0.09392|0.09809|
|switch19to18_direct_seed2|0.1|331/18|0.00245|0.00740|0.01180|0.02393|
|switch19to18_direct_seed2|0.3|313/18|0.00777|0.01984|0.03708|0.04035|
|switch19to18_direct_seed2|1.0|246/18|0.02261|0.04711|0.10510|0.09749|
|switch19to18_direct_seed2|2.0|156/18|0.03811|0.07840|0.15347|0.13536|
|switch19to18_direct_seed2|3.0|66/18|0.05613|0.12238|0.17768|0.18048|
|switch19to18_residual_seed0|0.1|331/18|0.00222|0.00672|0.01233|0.01347|
|switch19to18_residual_seed0|0.3|313/18|0.00667|0.01767|0.03264|0.03505|
|switch19to18_residual_seed0|1.0|246/18|0.01569|0.02826|0.05084|0.04879|
|switch19to18_residual_seed0|2.0|156/18|0.02338|0.04231|0.07438|0.06353|
|switch19to18_residual_seed0|3.0|66/18|0.02811|0.04650|0.08049|0.06798|
|switch19to18_residual_seed1|0.1|331/18|0.00222|0.00622|0.01217|0.01326|
|switch19to18_residual_seed1|0.3|313/18|0.00706|0.01856|0.03112|0.03401|
|switch19to18_residual_seed1|1.0|246/18|0.01688|0.03137|0.05512|0.04986|
|switch19to18_residual_seed1|2.0|156/18|0.02465|0.04272|0.07173|0.07196|
|switch19to18_residual_seed1|3.0|66/18|0.02856|0.05400|0.07539|0.06977|
|switch19to18_residual_seed2|0.1|331/18|0.00224|0.00619|0.01183|0.01303|
|switch19to18_residual_seed2|0.3|313/18|0.00680|0.01759|0.03254|0.03399|
|switch19to18_residual_seed2|1.0|246/18|0.01630|0.02908|0.05673|0.05058|
|switch19to18_residual_seed2|2.0|156/18|0.02777|0.05542|0.07857|0.07514|
|switch19to18_residual_seed2|3.0|66/18|0.03293|0.06021|0.07978|0.06883|

逐路线完整结果见per_route.csv；worst_windows.csv保留各模型各时域最差整车支持窗口的路线、初态和真实锚点。窗口重叠，不能把窗口数当独立样本。

平均误差和最差误差分别保留；不同模型/初始化可能排序不同。滚动少量实例不足以证明整个滚动域已覆盖，原覆盖失败仍保留。当前没有按结果自动放宽安全裕量或采用长时域规划。

全覆盖XY：每个模型/条件选择自己的最差1秒XY窗口，窗口锚点可不同，不能视为逐点配对误差图。原始完整轨迹及参考为底图，预测曲线仅覆盖所选有效窗口。

![rolling_M0 全路线实际与预测](rolling_M0_xy.png)

![rolling_direct_seed0 全路线实际与预测](rolling_direct_seed0_xy.png)

![rolling_direct_seed1 全路线实际与预测](rolling_direct_seed1_xy.png)

![rolling_direct_seed2 全路线实际与预测](rolling_direct_seed2_xy.png)

![rolling_residual_seed0 全路线实际与预测](rolling_residual_seed0_xy.png)

![rolling_residual_seed1 全路线实际与预测](rolling_residual_seed1_xy.png)

![rolling_residual_seed2 全路线实际与预测](rolling_residual_seed2_xy.png)

![static18_M0 全路线实际与预测](static18_M0_xy.png)

![static18_direct_seed0 全路线实际与预测](static18_direct_seed0_xy.png)

![static18_direct_seed1 全路线实际与预测](static18_direct_seed1_xy.png)

![static18_direct_seed2 全路线实际与预测](static18_direct_seed2_xy.png)

![static18_residual_seed0 全路线实际与预测](static18_residual_seed0_xy.png)

![static18_residual_seed1 全路线实际与预测](static18_residual_seed1_xy.png)

![static18_residual_seed2 全路线实际与预测](static18_residual_seed2_xy.png)

![static19_M0 全路线实际与预测](static19_M0_xy.png)

![static19_direct_seed0 全路线实际与预测](static19_direct_seed0_xy.png)

![static19_direct_seed1 全路线实际与预测](static19_direct_seed1_xy.png)

![static19_direct_seed2 全路线实际与预测](static19_direct_seed2_xy.png)

![static19_residual_seed0 全路线实际与预测](static19_residual_seed0_xy.png)

![static19_residual_seed1 全路线实际与预测](static19_residual_seed1_xy.png)

![static19_residual_seed2 全路线实际与预测](static19_residual_seed2_xy.png)

![switch18to19_M0 全路线实际与预测](switch18to19_M0_xy.png)

![switch18to19_direct_seed0 全路线实际与预测](switch18to19_direct_seed0_xy.png)

![switch18to19_direct_seed1 全路线实际与预测](switch18to19_direct_seed1_xy.png)

![switch18to19_direct_seed2 全路线实际与预测](switch18to19_direct_seed2_xy.png)

![switch18to19_residual_seed0 全路线实际与预测](switch18to19_residual_seed0_xy.png)

![switch18to19_residual_seed1 全路线实际与预测](switch18to19_residual_seed1_xy.png)

![switch18to19_residual_seed2 全路线实际与预测](switch18to19_residual_seed2_xy.png)

![switch19to18_M0 全路线实际与预测](switch19to18_M0_xy.png)

![switch19to18_direct_seed0 全路线实际与预测](switch19to18_direct_seed0_xy.png)

![switch19to18_direct_seed1 全路线实际与预测](switch19to18_direct_seed1_xy.png)

![switch19to18_direct_seed2 全路线实际与预测](switch19to18_direct_seed2_xy.png)

![switch19to18_residual_seed0 全路线实际与预测](switch19to18_residual_seed0_xy.png)

![switch19to18_residual_seed1 全路线实际与预测](switch19to18_residual_seed1_xy.png)

![switch19to18_residual_seed2 全路线实际与预测](switch19to18_residual_seed2_xy.png)

原始逐窗口/模型身份/数组：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/p3_multispeed_geometry`。
