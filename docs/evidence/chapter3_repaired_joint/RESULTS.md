# Measured-state downstream route selection

![All actual/reference XY](xy_all.png)

![Per-step and cumulative rewards](reward_all.png)

Development retest of the same 30 previously inspected roots; three methods and three crossed-slot repetitions, 90 episodes/method. All upstream failures remain in the denominator. Models and conditions frozen before execution.

|Method|Goal/N|Handoffs|Valid handoffs|Published|Road violations|Mean query ms|
|---|---:|---:|---:|---:|---:|---:|
|joint_feedback|23/90|88|87|88|11|0.872|
|joint_open|25/90|88|87|88|15|0.683|
|fixed_straight|34/90|81|77|81|10|0.642|

|Comparator|Mean root difference|Root bootstrap95%|Gains/losses|Equal query descriptors/both queried|
|---|---:|---|---:|---:|
|joint_open|-0.0222|[-0.15555555555555556, 0.1222222222222222]|13/15|0/86|
|fixed_straight|-0.1222|[-0.2666666666666666, 0.01111111111111111]|12/23|0/79|

![Successful non-straight selection, descriptive](selected_success.png)

Task-conditioned selected-route outcomes: [{'route': 60, 'selected': 48, 'goals': 17}, {'route': 61, 'selected': 33, 'goals': 5}, {'route': 62, 'selected': 7, 'goals': 1}]. These conditional counts are selection-biased, not route rankings. All 257 publications match the logged route and declared delay.

Actual 1068480 physical lane steps; summed simulation wall323.72s; 0 training.



Full query/publication records,model scores,actual contexts and endpoints remain in repeat0/1/2 and results.json. All failures and unpublished decisions are retained.

|Method|Approach choices|Planned downstream|Handoff revisions|Successful revised|Mean upstream query ms|
|---|---|---|---:|---:|---:|
|joint_feedback|{5: 38, 2: 50, 8: 1, 0: 1}|{60: 78, 61: 12}|38|6|373.332|
|joint_open|{5: 38, 2: 50, 8: 1, 0: 1}|{60: 78, 61: 12}|0|0|369.720|
|fixed_straight|{7: 90}|{60: 90}|0|0|0.710|

Joint panel: at x=3 m, frozen upstream predictions score 15 approach speed/preference pairs and three downstream routes. Joint-open retains its planned downstream route; joint-feedback queries again at measured handoff. Fixed uses 2 m/s, alpha .5 and straight continuation. The planned road runs during handoff inference delay. Route publication does not reset physical/controller state. This extension excludes upstream S/90-degree/bypass candidates; earlier broader panels remain separate.

No certified margin, independent-map generalization, superiority or equivalence claim. Query descriptors differ and hidden solver state is not serialized. All failures remain in the primary denominator.
