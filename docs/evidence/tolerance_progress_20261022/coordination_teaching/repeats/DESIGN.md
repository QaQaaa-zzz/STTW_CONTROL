# Two fixed optimization-seed repeats

Only minibatch sampling seed changes:918502,918503 versus original918501. Same verified dataset, parentu325, actor architecture, four groups,400 steps,Adam3e-5,clip1. Same physical controller/reward. Fresh optimizer each time; fixed final400, no best seed selection.

Two actors evaluated together on all original DEV panels:moderate400,strong400,low_speed120,straight20=940 fresh trajectories,maximum1880000 lower ticks,zero new PPO. Each Actor branches only within its own policy/context. No pooling seeds as independent physics trials. Training data and snapshot are shared, so this checks optimization reproducibility only; it cannot establish robustness to dataset/reset variation.

Primary outcomes: per-seed target low-speed8, all60 low-speed tasks, standard strong200, unique nominal30 and all-alpha speed/path behavior. Retain work-roll continuous costs and domain exits. Report all seeds, not the strongest. No automatic deployment, reward change, training extension or retries. Full numerical panels precede figures.
