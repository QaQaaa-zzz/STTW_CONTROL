# Model-reserve reward integration smoke

Completed1600/1600 lower ticks in204.27s, zero PPO updates. Same genuine left90_R2/2.3m/s/t1 snapshot per frozen Coverage u200 endpoint, same local300, four reward modes, vmap2/scan50 and shared compilation cache. Source b2dd78e. 29 related CPU tests passed before the GPU run.

- Zero-weight mode matches baseline exactly in final qpos, qvel and every reward sample in this smoke. Existing ten-component schema retained.
- Static mode has eleven components but reserve cost is zero on this segment. This is a measured property, not a failed logger and not evidence of equivalence throughout training.
- Dynamic cost rate ranges0..8.68075848, below cap10, and integral is7.50069666 over both endpoint trajectories. Reward difference relative to baseline is about-0.75007224, consistent with scale0.1 plus small numerical differences in the original costs.
- Independently reconstructed each tick reward from all effective components times0.1*0.005. Checks passed for every mode. No policy/lower engineering faults, physical failures or domain exits.
- Static max qpos/qvel deviations1.52588e-5/3.10913e-5; dynamic7.62939e-6/2.28882e-5. Retained as accepted GPU numerical variation; no exact replay gate imposed.

Original source checkpoint audits also passed for both endpoints: update200,17 policy tensors,17 parent Adam slots, terminal_contract_v2; original actor export matches serialized actor. New reward training must copy actor/critic/log_std with fresh optimizer consistently across arms, not silently reuse parent optimizer or call it exact resume. That fork and production manifests are still pending.

This verifies integration/accounting on two short trajectories. It does not prove model-margin validity or learned recovery improvement. No new model is adopted; full objective remains incomplete.
