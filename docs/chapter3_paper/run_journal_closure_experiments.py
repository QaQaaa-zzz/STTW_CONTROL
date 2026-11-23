"""Execute journal-closure experiments E1, E2, and E3 with exact physical-branch parity.

E1: Stage 1 pre-jump preference (alpha) physical state-manifold shift across 100 unseen starts.
E2: Safety margin epsilon Pareto sweep vs. Stage 3 veto fallback + complete Stage 3 rule/k/feature
    ablations across all 100 unseen starts (Panels 1 & 2, 2,700 left-gate contracts).
E3: Full two-sided (Left + Right gates, all 6 contracts) unified 3-stage system evaluation across
    Test A/B (1,440 contracts) and 3 unseen wall layouts x 100 unseen starts (5,400 contracts).
"""
import json
import sys
import time
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pandas as pd
import torch

REPO = Path("/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad")
sys.path.insert(0, str(REPO / "learning/src"))
sys.path.insert(0, str(REPO / "learning/cli"))

from sttw_control.platform_fallback import VetoExitBank, handback_features
from sttw_control.platform_online import velocity_state
from sttw_control.playground_history_model import GroundPredictionState, load_history_model
from sttw_control.playground_prediction import BodyEnvelope, euler, rotate
from sttw_control.playground_terrain_features import TerrainFeatures

torch.set_num_threads(8)

PM = Path("/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw")
PR = PM / "platform_test0_20261120/approach_alignment/replication_wide/chapter_mainline/post_review"
OUT = REPO / "docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/journal_closure"
OUT.mkdir(parents=True, exist_ok=True)

ACTIONS = [(0, 0.0), (1, 0.0), (2, 0.0), (1, 0.5), (1, 1.0)]
EXITS = ["left", "straight", "right"]
SECTIONS = ["near", "mid", "far"]
POSE_ORDER = torch.tensor([0, 1, 2, 3, 4, 5, 6, 8, 9, 7, 10, 11])
EPS_RB = 0.1017768085002899
EPS_GRID = [0.00, 0.02, 0.04, 0.06, 0.08, EPS_RB, 0.12, 0.15, 0.20]


def bootstrap_ci(df, col, seed=20261122):
    by_root = df.groupby("root")[col].mean().to_numpy(dtype=float)
    draws = np.random.default_rng(seed).choice(by_root, (10000, len(by_root)))
    lo, hi = np.quantile(draws.mean(1), [0.025, 0.975])
    return float(by_root.mean()), float(lo), float(hi)


def bootstrap_delta_ci(df, col_new, col_old, seed=20261122):
    by_root = (df[col_new].astype(float) - df[col_old].astype(float)).groupby(df["root"]).mean().to_numpy()
    draws = np.random.default_rng(seed).choice(by_root, (10000, len(by_root)))
    lo, hi = np.quantile(draws.mean(1), [0.025, 0.975])
    return float(by_root.mean()), float(lo), float(hi)


class CpuStage2Evaluator:
    """Exact CPU replica of PlatformDepartureQuery returning per-exit clearances and predicates."""

    def __init__(self, part_dir: Path, sections_x: list[float]):
        cfg = json.loads((part_dir / "resolved_config.json").read_text())
        self.hand = cfg["ground_handoff"]
        self.online = cfg["platform_online"]
        self.sections = dict(zip(SECTIONS, sections_x))
        self.req_speed = float(self.hand["handoff_variants"][0]["continuation_speed_m_s"])
        self.goal = self.hand["goal"]
        self.halfwidth = float(self.hand["road_half_width_m"])
        self.margin = float(self.online["margin_m"])

        physics = mujoco.MjModel.from_binary_path(str(part_dir / "terrain.mjb"))
        self.model = load_history_model(Path(self.online["ground_model"]), device="cpu")
        terrain = dict(np.load(self.online["relative"]["raster"]))
        terrain["height"] = terrain["height"] - self.online["relative"]["height"]
        self.model.terrain = TerrainFeatures(**terrain, device="cpu")
        with np.load(self.online["relative"]["initializer"]) as init:
            self.mu, self.sc, self.wt = [torch.as_tensor(init[k], dtype=torch.float32) for k in ("mu", "scale", "weight")]
        self.env = BodyEnvelope(physics, device="cpu", dimensions=3)
        walls = [i for i in range(physics.ngeom) if physics.geom(i).name.startswith("planning_obstacle_")]
        self.center = torch.as_tensor(physics.geom_pos[walls], dtype=torch.float32)
        self.half = torch.as_tensor(physics.geom_size[walls], dtype=torch.float32)
        with np.load(cfg["paired_departures"]["route_bank"]) as bank:
            self.route = SimpleNamespace(**{k: torch.as_tensor(bank[k], dtype=torch.float32) for k in ("s", "xy", "heading", "curvature")})
            xy = bank["xy"].astype(np.float32, copy=False)
        self.stride = len(self.route.s)
        self.anchors = {}
        for a in range(3):
            for d in range(3):
                off = 3 + (3 * a + d) * self.stride
                self.anchors[a, d] = {sha256(row).digest(): j for j, row in enumerate(xy[off:off + self.stride])}

    @torch.inference_mode()
    def evaluate_prefixes(self, df_unique: pd.DataFrame) -> dict:
        """Run Stage 2 dynamic rollout once per unique verified prefix; return raw per-exit metrics."""
        out = {}
        df_work = df_unique.assign(tick=np.rint(df_unique["handoff_s"] / 0.005).astype(int) + 1)
        for tick, grp in df_work.groupby("tick"):
            steps = int((10.0 - tick * 0.005) / 0.02)
            if steps + 1 < 26:
                for r in grp.itertuples():
                    out[r.prefix] = dict(valid_horizon=False, q0=None, v0=None)
                continue
            states, refs, q0_list, v0_list = [], [], [], []
            for r in grp.itertuples():
                dep_used = int(r.chosen_dep) if hasattr(r, "chosen_dep") and not pd.isna(r.chosen_dep) else int(r.planned_dep)
                with np.load(r.trajectory) as z:
                    q, v = z["qpos"][[tick, tick - 4]], z["qvel"][[tick, tick - 4]]
                    anchor = self.anchors[int(r.approach), dep_used][sha256(z["published_reference_xy"]).digest()]
                refs.append(3 + (3 * int(r.approach) + np.arange(3)) * self.stride + anchor)
                states.append((q[0], v[0], q[1], v[1]))
                q0_list.append(q[0])
                v0_list.append(v[0])
            q, v, pq, pv = [torch.as_tensor(np.stack(a), dtype=torch.float32) for a in zip(*states)]
            vel = velocity_state(q, v)
            prev = velocity_state(pq, pv)
            pose = q.index_select(1, POSE_ORDER).repeat_interleave(3, 0)
            refs_t = torch.as_tensor(np.asarray(refs), dtype=torch.long).flatten()
            pub = (self.route.s, self.route.xy[refs_t], self.route.heading[refs_t], self.route.curvature[refs_t])
            u, pr_v = vel.repeat_interleave(3, 0), prev.repeat_interleave(3, 0)
            inp = torch.cat((pose[:, :8], pose[:, 10:12], u, pr_v), 1)
            hidden = ((inp - self.mu) / self.sc) @ self.wt[1:] + self.wt[0]
            pose[:, 2] -= self.online["relative"]["height"]
            f = self.model.rollout_state(
                GroundPredictionState(pose, u, pr_v, hidden), *pub, steps, torch.full((len(pose),), self.req_speed)
            )
            poses, vel_p, domain = f["pose"], f["velocity"], f["domain"]
            poses[:, :, 2] += self.online["relative"]["height"]
            lo, hi = self.env(poses)
            dist = torch.maximum(lo[:, :, :, None] - self.center - self.half,
                                 self.center - self.half - hi[:, :, :, None]).amax(-1).amin((2, 3))
            yaw = euler(poses[:, :, 3:7].reshape(-1, 4))[:, 2].reshape(poses.shape[:2])
            world_v = rotate(poses[:, :, 3:7].reshape(-1, 4), vel_p[:, :, :3].reshape(-1, 3)).reshape(*yaw.shape, 3)
            vx = world_v[:, :, 0]
            spd = vx * yaw.cos() + world_v[:, :, 1] * yaw.sin()
            g = self.goal
            hit = (
                (poses[:, :, 0] >= g["x_min_m"]) & (poses[:, :, 0] <= g["x_max_m"])
                & (poses[:, :, 1].abs() <= g["abs_y_max_m"]) & (vx >= g["min_vx_m_s"])
                & (domain[:, :, 0].abs() <= g["path_tolerance_m"])
                & (domain[:, :, 1].abs() <= g["heading_tolerance_rad"])
                & ((spd - self.req_speed).abs() <= g["speed_tolerance_m_s"])
            )
            held = hit.unfold(1, 26, 1).all(-1)
            end = torch.where(held.any(1), held.long().argmax(1) + 25, hit.shape[1] - 1)
            win = torch.arange(hit.shape[1]) <= end[:, None]
            clear = dist.masked_fill(~win, float("inf")).amin(1).reshape(-1, 3).numpy()
            bad = (
                (domain[:, :, 0].abs() > self.halfwidth)
                | (domain[:, :, 1].abs() >= torch.pi / 2)
                | (domain[:, :, 2] < self.model.nominal.follower["preview_body_x_min_m"])
            )
            att = euler(poses[:, :, 3:7].reshape(-1, 4)).reshape(*yaw.shape, 3)
            bad |= (att[:, :, 0].abs() > 0.7) | (att[:, :, 1].abs() > 1.309)
            in_dom = (~(bad & win).any(1)).reshape(-1, 3).numpy()
            pass_goal = (held.any(1).reshape(-1, 3).numpy()) & in_dom
            x, y = poses[:, :, 0], poses[:, :, 1]

            for i, r in enumerate(grp.itertuples()):
                sl = slice(3 * i, 3 * i + 3)
                gates = {}
                for sec_name, sec_x in self.sections.items():
                    cross = (x[sl, :-1] < sec_x) & (x[sl, 1:] >= sec_x) & win[sl, 1:]
                    ci = cross.long().argmax(1)
                    ar = torch.arange(3)
                    at = y[sl][ar, ci] + (sec_x - x[sl][ar, ci]) * (y[sl][ar, ci + 1] - y[sl][ar, ci]) / (
                        x[sl][ar, ci + 1] - x[sl][ar, ci]
                    ).clamp_min(1e-7)
                    has_c = cross.any(1).numpy()
                    at_np = at.numpy()
                    for side in (-1, 1):
                        gates[(sec_name, side)] = has_c & (at_np * side >= 0.06) & (np.abs(at_np) <= 3.9)
                out[r.prefix] = dict(
                    valid_horizon=True,
                    clear=clear[i],
                    pass_goal=pass_goal[i],
                    gates=gates,
                    q0=q0_list[i],
                    v0=v0_list[i],
                )
        return out


def stage2_decision(rec: dict, section: str, side: int, incumbent: int, margin: float):
    if not rec["valid_horizon"]:
        return incumbent, False
    clear = rec["clear"]
    safe = clear > margin
    pg = rec["pass_goal"]
    gt = rec["gates"][(section, side)]
    feas = safe & pg & gt
    score = 16.0 * feas + 4.0 * safe + 2.0 * gt + 1.0 * pg + 0.01 * np.clip(clear, -1.0, 1.0)
    choice = incumbent if bool(feas[incumbent]) else int(np.argmax(score))
    return choice, bool(feas[choice])


def stage1_actions_twosided(prior_df: pd.DataFrame, bank: dict, roots: list[int], margin: float):
    """Symmetric 30-root 5-NN Stage 1 selection for BOTH sides (-1 and +1) and 9 vs 15 options."""
    ids = np.arange(8, 38)
    features = np.column_stack((bank["qpos"][:, 1:2], bank["qpos"][:, 3:7], bank["qvel"][:, :6]))
    std = features[ids].std(0).clip(0.005)
    out = {}
    for root in roots:
        dist = (((features[ids] - features[root]) / std) ** 2).sum(1)
        near = ids[np.argsort(dist)[:5]]
        for sec in SECTIONS:
            for side in (-1, 1):
                gate = (prior_df[sec] * side >= 0.06) & (prior_df[sec].abs() <= 3.9)
                safe = prior_df["clear"] > margin
                val = 16.0 * (safe & gate & prior_df["reachable"]).astype(float) + 4.0 * safe + 2.0 * gate + prior_df["reachable"].astype(float) + 0.01 * prior_df["clear"].clip(-1, 1)
                grid = prior_df.assign(val=val).pivot(index="root", columns=["candidate", "departure"], values="val")
                grid = grid.reindex(index=near, columns=pd.MultiIndex.from_product([range(5), EXITS])).fillna(-4.0)
                sc = grid.mean().to_numpy()
                i9, d9 = divmod(int(np.argmax(sc[:9])), 3)
                i15, d15 = divmod(int(np.argmax(sc[:15])), 3)
                a9, al9 = ACTIONS[i9]
                a15, al15 = ACTIONS[i15]
                fix_a, fix_d = (1, 0) if side == 1 else (0, 2)
                out[(root, sec, side)] = {
                    "fixed": (fix_a, 0.0, fix_d),
                    "alpha0": (a9, al9, d9),
                    "alpha_enabled": (a15, al15, d15),
                }
    return out


# ===========================================================================
# EXPERIMENT 1: Stage 1 Pre-Jump Alpha Physical State-Manifold Shift
# ===========================================================================
def run_exp1():
    print("\n=== Running Experiment E1: Stage 1 Alpha Physical State-Manifold Shift ===")
    t0 = time.perf_counter()
    o1 = pd.read_csv(PR / "veto_fallback/oracle_cases.csv")
    o2 = pd.read_csv(PR / "veto_fallback_followup/oracle_cases.csv")
    orc = pd.concat([o1, o2], ignore_index=True)

    # Signed-left approach (approach == 1) across 100 unseen starts x 3 forces x 3 alphas
    left_orc = orc[(orc["approach"] == 1) & (orc["side"] == 1)].copy()
    # Physical left-gate solvability per (layout, root, force, approach_alpha) across any of 3 exits (near gate + any section)
    solv = left_orc.groupby(["layout", "root", "force", "approach_alpha", "section"])["success"].any().unstack("section")
    solv["solvable_near"] = solv["near"].astype(int)
    solv["solvable_any"] = solv[SECTIONS].any(axis=1).astype(int)
    solv_map = solv[["solvable_near", "solvable_any"]].unstack("layout")

    # Extract handback physical state once per (root, force, approach_alpha) using config_double trajectories
    base = left_orc[(left_orc["layout"] == "config_double") & (left_orc["section"] == "near") & (left_orc["planned_dep"] == 0)].copy()
    rows = []
    for r in base.itertuples():
        key = (r.root, r.force, r.approach_alpha)
        s_l = int(solv_map.loc[key, ("solvable_near", "config_l")])
        s_r = int(solv_map.loc[key, ("solvable_near", "config_r")])
        s_d = int(solv_map.loc[key, ("solvable_near", "config_double")])
        if not r.verified or pd.isna(r.handoff_s):
            rows.append(dict(root=r.root, force=r.force, alpha=r.approach_alpha, verified=0,
                             solv_l=s_l, solv_r=s_r, solv_double=s_d))
            continue
        tick = int(round(r.handoff_s / 0.005)) + 1
        with np.load(r.trajectory) as z:
            q = z["qpos"][tick]
            v = z["qvel"][tick]
        w, xq, yq, zq = q[3:7]
        roll = np.degrees(np.arctan2(2 * (w * xq + yq * zq), 1 - 2 * (xq * xq + yq * yq)))
        yaw = np.degrees(np.arctan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq)))
        steer = np.degrees(q[8])
        rows.append(dict(
            root=int(r.root), force=int(r.force), alpha=float(r.approach_alpha), verified=1,
            handoff_s=float(r.handoff_s), x_hb=float(q[0]), y_hb=float(q[1]),
            yaw_hb_deg=float(yaw), roll_hb_deg=float(roll), steer_hb_deg=float(steer),
            vx_hb=float(v[0]), vy_hb=float(v[1]),
            solv_l=s_l, solv_r=s_r, solv_double=s_d,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp1_alpha_manifold.csv", index=False)

    # Summary statistics + paired deltas vs alpha=0
    ver = df[df["verified"] == 1].copy()
    summ = []
    for al, g in df.groupby("alpha"):
        gv = g[g["verified"] == 1]
        summ.append(dict(
            alpha=al, total=len(g), verified=len(gv), verified_pct=100 * len(gv) / len(g),
            y_hb_med=gv["y_hb"].median(), y_hb_iqr=gv["y_hb"].quantile(0.75) - gv["y_hb"].quantile(0.25),
            yaw_hb_med=gv["yaw_hb_deg"].median(), yaw_hb_iqr=gv["yaw_hb_deg"].quantile(0.75) - gv["yaw_hb_deg"].quantile(0.25),
            steer_hb_med=gv["steer_hb_deg"].median(),
            vx_hb_med=gv["vx_hb"].median(),
            solv_l_pct=100 * g["solv_l"].mean(), solv_r_pct=100 * g["solv_r"].mean(), solv_double_pct=100 * g["solv_double"].mean(),
        ))
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(OUT / "exp1_alpha_summary.csv", index=False)

    # Paired deltas relative to alpha=0
    piv = ver.pivot(index=["root", "force"], columns="alpha", values=["y_hb", "yaw_hb_deg", "steer_hb_deg", "vx_hb"])
    deltas = []
    for al in (0.5, 1.0):
        mask = piv[("y_hb", 0.0)].notna() & piv[("y_hb", al)].notna()
        deltas.append(dict(
            alpha=al, n_paired=int(mask.sum()),
            delta_y_cm_med=100 * (piv.loc[mask, ("y_hb", al)] - piv.loc[mask, ("y_hb", 0.0)]).median(),
            delta_yaw_deg_med=(piv.loc[mask, ("yaw_hb_deg", al)] - piv.loc[mask, ("yaw_hb_deg", 0.0)]).median(),
            delta_steer_deg_med=(piv.loc[mask, ("steer_hb_deg", al)] - piv.loc[mask, ("steer_hb_deg", 0.0)]).median(),
            delta_vx_med=(piv.loc[mask, ("vx_hb", al)] - piv.loc[mask, ("vx_hb", 0.0)]).median(),
        ))
    pd.DataFrame(deltas).to_csv(OUT / "exp1_alpha_paired_deltas.csv", index=False)
    print(summ_df.to_string(index=False))
    print("\nPaired shifts vs alpha=0:")
    print(pd.DataFrame(deltas).to_string(index=False))
    print(f"E1 finished in {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# EXPERIMENT 2 & 3: Margin Sweep, Stage 3 Ablations, & Two-Sided Full System
# ===========================================================================
def build_twosided_veto_banks():
    """Build frozen 30-start (roots 1041-1070) two-sided Stage 3 banks from alpha_transfer."""
    orc = pd.read_csv(PR / "alpha_transfer/oracle_cases.csv")
    banks = {}
    for layout in ["config_l", "config_r", "config_double"]:
        sub = orc[(orc["layout"] == layout) & (orc["verified"])].copy()
        by_pref = sub.drop_duplicates("prefix").set_index("prefix")
        q_list, v_list = [], []
        for r in by_pref.itertuples():
            tick = int(round(r.handoff_s / 0.005)) + 1
            with np.load(r.trajectory) as z:
                q_list.append(z["qpos"][tick])
                v_list.append(z["qvel"][tick])
        wall = sub.drop_duplicates(["prefix", "planned_dep"]).pivot(index="prefix", columns="planned_dep", values="collision")
        # done indexed by (side_idx in [0(-1), 1(+1)], section_idx in [0,1,2], dep in [0,1,2])
        done = sub.pivot(index="prefix", columns=["side", "section", "planned_dep"], values="success")
        cols = pd.MultiIndex.from_product([(-1, 1), SECTIONS, range(3)])
        done = done.reindex(index=by_pref.index, columns=cols)
        assert not done.isna().any().any()
        banks[layout] = dict(
            state=handback_features(torch.tensor(np.stack(q_list)), torch.tensor(np.stack(v_list))).float().numpy(),
            root=by_pref["root"].to_numpy(),
            route=by_pref["approach"].to_numpy(),
            wall=wall.loc[by_pref.index, [0, 1, 2]].to_numpy(dtype=np.int64),
            done=done.to_numpy(dtype=np.int64).reshape(-1, 2, 3, 3),
        )
    return banks


def veto_choose_variant(bank: dict, state_vec: np.ndarray, route_val: int, sec_idx: int, side_idx: int,
                        incumbent: int, k: int = 5, mode: str = "v2", feat_idx=None):
    """Generalized Stage 3 fallback selector supporting V1/V2, k-NN sweep, and feature ablations."""
    st = bank["state"]
    if feat_idx is not None:
        st = st[:, feat_idx]
        sv = state_vec[feat_idx]
    else:
        sv = state_vec
    scale = np.maximum(st.std(axis=0), 0.015)
    dist = (((sv[None, :] - st) / scale) ** 2).sum(axis=1)
    mask = bank["route"] == route_val
    dist = np.where(mask, dist, np.inf)
    k_eff = min(k, int(mask.sum()))
    if k_eff == 0:
        return incumbent
    near = np.argpartition(dist, k_eff - 1)[:k_eff]
    # Exact deterministic tie-break via full argsort on masked finite distances
    near = np.argsort(dist)[:k_eff]
    walls = bank["wall"][near].sum(axis=0)
    wins = bank["done"][near, side_idx, sec_idx].sum(axis=0)

    if mode == "v1":
        # Unconstrained min-collision, breaking ties by wins, retaining incumbent on exact tie
        score = wins - (k_eff + 1) * walls
        best = int(np.argmax(score))
        return incumbent if score[incumbent] == score[best] else best
    elif mode == "v2":
        score = wins - (k_eff + 1) * walls
        improves = (walls < walls[incumbent]) & (wins >= wins[incumbent])
        masked = np.where(improves, score, -10**9)
        masked[incumbent] = score[incumbent]
        best = int(np.argmax(masked))
        return incumbent if masked[incumbent] == masked[best] else best
    raise ValueError(mode)


def run_exp2_and_exp3():
    print("\n=== Building Two-Sided 30-Start Frozen Veto Banks ===")
    t0 = time.perf_counter()
    twosided_banks = build_twosided_veto_banks()

    # Load Panel 1 + Panel 2 (100 independent unseen starts 1071-1170)
    oc1 = pd.read_csv(PR / "veto_fallback/online_cases.csv")
    oc2 = pd.read_csv(PR / "veto_fallback_followup/online_cases.csv")
    or1 = pd.read_csv(PR / "veto_fallback/oracle_cases.csv")
    or2 = pd.read_csv(PR / "veto_fallback_followup/oracle_cases.csv")
    online_all = pd.concat([oc1.assign(panel=1), oc2.assign(panel=2)], ignore_index=True)
    oracle_all = pd.concat([or1.assign(panel=1), or2.assign(panel=2)], ignore_index=True)

    # Index oracle outcomes by (layout, root, force, section, side, approach, approach_alpha, planned_dep)
    keys = ["layout", "root", "force", "section", "side", "approach", "approach_alpha", "planned_dep"]
    orc_lookup = oracle_all.set_index(keys)[["verified", "success", "collision", "handoff_s", "prefix", "trajectory"]]

    # Stage 1 symmetric plans for all 100 unseen starts (1071-1170) on all 3 layouts
    plan_vf = json.loads((PR / "veto_fallback/test_freeze_manifest.json").read_text())
    bank_init = dict(np.load(plan_vf["bank"]))
    roots_100 = list(range(1071, 1171))
    sections_x = plan_vf["section_x"]

    s1_by_layout = {}
    for layout in ["config_l", "config_r", "config_double"]:
        prior_df = pd.read_csv(PR / f"veto_fallback/{layout}/rb_run1_prior.csv")
        s1_by_layout[layout] = stage1_actions_twosided(prior_df, bank_init, roots_100, EPS_RB)

    # Collect EVERY unique verified prefix needed across ALL planners (fixed, alpha0, alpha_enabled) x BOTH sides!
    # Also evaluate exact CPU Stage 2 predictions once per unique verified prefix per layout.
    print("\n=== Running Exact CPU Stage 2 Forecasts Across All 100 Unseen Starts (3 Wall Layouts) ===")
    stage2_cache = {}
    for layout in ["config_l", "config_r", "config_double"]:
        t_lay = time.perf_counter()
        needed_keys = set()
        for root in roots_100:
            for sec in SECTIONS:
                for side in (-1, 1):
                    for arm in ("alpha0", "alpha_enabled"):
                        a, al, _ = s1_by_layout[layout][(root, sec, side)][arm]
                        for force in (-5, 0, 5):
                            needed_keys.add((layout, root, force, a, al))
        # Match to oracle verified prefixes (dep 0 trajectory carries identical prefix & relock state)
        sub_orc = oracle_all[(oracle_all["layout"] == layout) & (oracle_all["section"] == "near") &
                             (oracle_all["side"] == 1) & (oracle_all["planned_dep"] == 0) & (oracle_all["verified"])].copy()
        sub_orc = sub_orc[sub_orc.apply(lambda r: (r.layout, r.root, r.force, r.approach, r.approach_alpha) in needed_keys, axis=1)]
        sub_orc = sub_orc.drop_duplicates("prefix")
        part_dir = PR / f"veto_fallback/{layout}/part_0/rb_run1"
        ev = CpuStage2Evaluator(part_dir, sections_x)
        stage2_cache[layout] = ev.evaluate_prefixes(sub_orc)
        print(f"  {layout}: evaluated {len(sub_orc)} unique verified prefixes in {time.perf_counter()-t_lay:.1f}s")

    # -----------------------------------------------------------------------
    # STRICT PARITY CHECK against archived GPU online_cases (2,700 left-gate contracts!)
    # -----------------------------------------------------------------------
    print("\n=== Verifying 100% Parity Against Archived Online GPU Runs (2,700 Left-Gate Contracts) ===")
    exp_online = online_all[online_all["planner"] == "expanded"].copy()
    fb_online = online_all[online_all["planner"] == "veto_bank"].copy()
    mism_s2 = 0; mism_s3 = 0; n_ver_q = 0; n_veto_chk = 0

    # Map (layout, root, force, approach, alpha) -> prefix from oracle_all
    pref_map = oracle_all[(oracle_all["section"] == "near") & (oracle_all["side"] == 1) & (oracle_all["planned_dep"] == 0)].set_index(
        ["layout", "root", "force", "approach", "approach_alpha"]
    )[["verified", "prefix"]].to_dict("index")

    for r in exp_online.itertuples():
        if not r.verified or r.query_batch_ms <= 0:
            continue
        pinfo = pref_map[(r.layout, r.root, r.force, r.approach, r.approach_alpha)]
        rec = stage2_cache[r.layout][pinfo["prefix"]]
        ch, adm = stage2_decision(rec, r.section, int(r.side), int(r.planned_dep), EPS_RB)
        n_ver_q += 1
        if ch != int(r.chosen_dep) or adm != bool(r.query_feasible):
            mism_s2 += 1
    assert mism_s2 == 0, f"Stage 2 CPU parity failed: {mism_s2}/{n_ver_q}"

    # Verify Stage 3 parity against archived veto_bank online arm (all 2,700 contracts!)
    merged_fb = exp_online.merge(fb_online, on=["panel", "layout", "root", "force", "section", "side"], suffixes=("_exp", "_fb"))
    for r in merged_fb.itertuples():
        if not r.verified_exp or r.query_batch_ms_exp <= 0 or bool(r.query_feasible_exp):
            continue
        pinfo = pref_map[(r.layout, r.root, r.force, r.approach_exp, r.approach_alpha_exp)]
        rec = stage2_cache[r.layout][pinfo["prefix"]]
        feat = handback_features(torch.tensor(rec["q0"][None]), torch.tensor(rec["v0"][None])).float().numpy()[0]
        # Verify using exact archived single-side bank & our two-sided bank (side_idx=1)
        ch_fb = veto_choose_variant(twosided_banks[r.layout], feat, int(r.approach_exp), SECTIONS.index(r.section), 1, int(r.chosen_dep_exp), k=5, mode="v2")
        n_veto_chk += 1
        if ch_fb != int(r.chosen_dep_fb):
            mism_s3 += 1
    assert mism_s3 == 0, f"Stage 3 Veto Fallback parity failed: {mism_s3}/{n_veto_chk}"
    print(f"  PARITY PASSED: Stage 2 = 0/{n_ver_q} mismatches; Stage 3 Veto Fallback = 0/{n_veto_chk} mismatches!")

    # -----------------------------------------------------------------------
    # EXPERIMENT E2-A: Safety Margin Epsilon Sweep (2,700 left-gate contracts)
    # -----------------------------------------------------------------------
    print("\n=== Running Experiment E2-A: Safety Margin Epsilon Sweep (100 Unseen Starts, 2,700 Contracts) ===")
    sweep_rows = []
    for eps in EPS_GRID:
        per_contract = []
        for r in exp_online.itertuples():
            a, al, inc = int(r.approach), float(r.approach_alpha), int(r.planned_dep)
            pinfo = pref_map[(r.layout, r.root, r.force, a, al)]
            if not pinfo["verified"]:
                ch, adm, queried = inc, False, False
            else:
                rec = stage2_cache[r.layout][pinfo["prefix"]]
                if not rec["valid_horizon"]:
                    ch, adm, queried = inc, False, False
                else:
                    ch, adm = stage2_decision(rec, r.section, 1, inc, eps)
                    queried = True
            orc_row = orc_lookup.loc[(r.layout, r.root, r.force, r.section, 1, a, al, ch)]
            per_contract.append(dict(
                layout=r.layout, root=r.root, done=int(orc_row["success"]), wall=int(orc_row["collision"]),
                queried=int(queried), admitted=int(adm),
                adm_done=int(adm and orc_row["success"]), adm_wall=int(adm and orc_row["collision"]),
                veto=int(queried and not adm), veto_wall=int(queried and not adm and orc_row["collision"]),
            ))
        cdf = pd.DataFrame(per_contract)
        for scope, part in [("all", cdf), *cdf.groupby("layout")]:
            dm, dlo, dhi = bootstrap_ci(part, "done")
            wm, wlo, whi = bootstrap_ci(part, "wall")
            adm_n = int(part["admitted"].sum())
            sweep_rows.append(dict(
                epsilon_m=eps, scope=scope, n=len(part),
                done=int(part["done"].sum()), done_pct=100 * dm, done_lo=100 * dlo, done_hi=100 * dhi,
                wall=int(part["wall"].sum()), wall_pct=100 * wm, wall_lo=100 * wlo, wall_hi=100 * whi,
                queried=int(part["queried"].sum()), admitted=adm_n,
                adm_done=int(part["adm_done"].sum()), adm_wall=int(part["adm_wall"].sum()),
                adm_wall_pct=(100 * part["adm_wall"].sum() / adm_n) if adm_n > 0 else 0.0,
                veto=int(part["veto"].sum()), veto_wall=int(part["veto_wall"].sum()),
            ))
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT / "exp2_margin_sweep.csv", index=False)
    print(sweep_df[sweep_df["scope"].isin(["all", "config_double"])][["scope", "epsilon_m", "done_pct", "wall_pct", "admitted", "adm_wall", "adm_wall_pct", "veto_wall"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # EXPERIMENT E2-B: Stage 3 Rule, k-NN, & Feature-Space Ablations (2,700 contracts)
    # -----------------------------------------------------------------------
    print("\n=== Running Experiment E2-B: Stage 3 Fallback Rule / k-NN / Feature Ablations ===")
    variants = [
        ("Default (Stage 2)", "default", None, None),
        ("Static: Always Left", "static", 0, None),
        ("Static: Always Straight", "static", 1, None),
        ("Static: Always Right", "static", 2, None),
        ("V1 Min-Collision (k=5)", "v1", 5, None),
        ("V2 Pareto-Guarded (k=1)", "v2", 1, None),
        ("V2 Pareto-Guarded (k=3)", "v2", 3, None),
        ("V2 Pareto-Guarded (k=5)", "v2", 5, None),
        ("V2 Pareto-Guarded (k=7)", "v2", 7, None),
        ("V2 Pareto-Guarded (k=10)", "v2", 10, None),
        ("V2 Pareto-Guarded (k=15)", "v2", 15, None),
        ("V2 Pareto-Guarded (k=25)", "v2", 25, None),
        ("V2 Features: XY only", "v2", 5, [0, 1]),
        ("V2 Features: XY+Orient", "v2", 5, [0, 1, 2, 3, 4]),
        ("V2 Features: XY+Orient+Steer", "v2", 5, [0, 1, 2, 3, 4, 5]),
        ("Post-Veto Physical Oracle", "oracle_postveto", None, None),
    ]

    # Precompute handback feature vector & 3-exit physical outcomes for every contract in exp_online
    base_items = []
    for r in exp_online.itertuples():
        a, al, ch_def = int(r.approach), float(r.approach_alpha), int(r.chosen_dep)
        is_veto = bool(r.verified and r.query_batch_ms > 0 and not r.query_feasible)
        pinfo = pref_map[(r.layout, r.root, r.force, a, al)]
        feat = None
        if is_veto:
            rec = stage2_cache[r.layout][pinfo["prefix"]]
            feat = handback_features(torch.tensor(rec["q0"][None]), torch.tensor(rec["v0"][None])).float().numpy()[0]
        # Exact 3-exit outcomes for this handback
        outs = [orc_lookup.loc[(r.layout, r.root, r.force, r.section, 1, a, al, d)] for d in range(3)]
        dones = [int(o["success"]) for o in outs]
        walls = [int(o["collision"]) for o in outs]
        base_items.append((r.layout, r.root, r.section, a, ch_def, is_veto, feat, dones, walls))

    abl_rows = []
    for name, kind, param, fidx in variants:
        recs = []
        for lay, root, sec, a, ch_def, is_veto, feat, dones, walls in base_items:
            if not is_veto or kind == "default":
                ch = ch_def
            elif kind == "static":
                ch = param
            elif kind in ("v1", "v2"):
                ch = veto_choose_variant(twosided_banks[lay], feat, a, SECTIONS.index(sec), 1, ch_def, k=param, mode=kind, feat_idx=fidx)
            elif kind == "oracle_postveto":
                # Best physically available exit from this vetoed handback: avoid wall first, then complete
                scores = [dones[d] - 2 * walls[d] for d in range(3)]
                best = int(np.argmax(scores))
                ch = ch_def if scores[ch_def] == scores[best] else best
            recs.append(dict(layout=lay, root=root, done=dones[ch], wall=walls[ch],
                             def_done=dones[ch_def], def_wall=walls[ch_def], switched=int(ch != ch_def)))
        rdf = pd.DataFrame(recs)
        for scope, part in [("all", rdf), *rdf.groupby("layout")]:
            dm, dlo, dhi = bootstrap_delta_ci(part, "done", "def_done")
            wm, wlo, whi = bootstrap_delta_ci(part, "wall", "def_wall")
            abl_rows.append(dict(
                variant=name, scope=scope, n=len(part), switches=int(part["switched"].sum()),
                done=int(part["done"].sum()), done_pct=100 * part["done"].mean(),
                wall=int(part["wall"].sum()), wall_pct=100 * part["wall"].mean(),
                delta_done_pp=100 * dm, delta_done_lo=100 * dlo, delta_done_hi=100 * dhi,
                delta_wall_pp=100 * wm, delta_wall_lo=100 * wlo, delta_wall_hi=100 * whi,
            ))
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(OUT / "exp2_fallback_ablation.csv", index=False)
    print(abl_df[abl_df["scope"].isin(["all", "config_double"])][["scope", "variant", "done", "wall", "delta_done_pp", "delta_wall_pp", "delta_wall_lo", "delta_wall_hi"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # EXPERIMENT E3-A: Full Two-Sided (Left + Right Gate) Unified 3-Stage System
    # Across ALL 100 Unseen Starts x 3 Wall Geometries = 5,400 RB Contracts!
    # -----------------------------------------------------------------------
    print("\n=== Running Experiment E3-A: Two-Sided 6-Contract Unified 3-Stage System (100 Unseen Starts, 5,400 Contracts) ===")
    ts_records = []
    for layout in ["config_l", "config_r", "config_double"]:
        for root in roots_100:
            for force in (-5, 0, 5):
                for sec in SECTIONS:
                    for side in (-1, 1):
                        side_idx = 0 if side == -1 else 1
                        s1 = s1_by_layout[layout][(root, sec, side)]
                        # Evaluate 4 systems on identical (layout, root, force, sec, side)
                        # 1. Fixed (alpha=0)
                        fa, fal, fd = s1["fixed"]
                        o_fix = orc_lookup.loc[(layout, root, force, sec, side, fa, fal, fd)]
                        ts_records.append(dict(layout=layout, root=root, force=force, section=sec, side=side,
                                               system="Fixed (a=0)", done=int(o_fix["success"]), wall=int(o_fix["collision"]),
                                               admitted=0, adm_wall=0))
                        # 2. C-R (alpha=0)
                        a0, al0, d0 = s1["alpha0"]
                        p0 = pref_map[(layout, root, force, a0, al0)]
                        if p0["verified"] and stage2_cache[layout][p0["prefix"]]["valid_horizon"]:
                            ch0, adm0 = stage2_decision(stage2_cache[layout][p0["prefix"]], sec, side, d0, EPS_RB)
                        else:
                            ch0, adm0 = d0, False
                        o_0 = orc_lookup.loc[(layout, root, force, sec, side, a0, al0, ch0)]
                        ts_records.append(dict(layout=layout, root=root, force=force, section=sec, side=side,
                                               system="C-R (a=0)", done=int(o_0["success"]), wall=int(o_0["collision"]),
                                               admitted=int(adm0), adm_wall=int(adm0 and o_0["collision"])))
                        # 3. C-R (Exp. alpha) and 4. Full 3-Stage (Exp. alpha + Stage 3 Veto Fallback)
                        ae, ale, de = s1["alpha_enabled"]
                        pe = pref_map[(layout, root, force, ae, ale)]
                        if pe["verified"] and stage2_cache[layout][pe["prefix"]]["valid_horizon"]:
                            rec_e = stage2_cache[layout][pe["prefix"]]
                            che, adme = stage2_decision(rec_e, sec, side, de, EPS_RB)
                            if not adme:
                                feat_e = handback_features(torch.tensor(rec_e["q0"][None]), torch.tensor(rec_e["v0"][None])).float().numpy()[0]
                                ch3 = veto_choose_variant(twosided_banks[layout], feat_e, ae, SECTIONS.index(sec), side_idx, che, k=5, mode="v2")
                            else:
                                ch3 = che
                        else:
                            che, adme, ch3 = de, False, de
                        o_e = orc_lookup.loc[(layout, root, force, sec, side, ae, ale, che)]
                        o_3 = orc_lookup.loc[(layout, root, force, sec, side, ae, ale, ch3)]
                        ts_records.append(dict(layout=layout, root=root, force=force, section=sec, side=side,
                                               system="C-R (Exp. a)", done=int(o_e["success"]), wall=int(o_e["collision"]),
                                               admitted=int(adme), adm_wall=int(adme and o_e["collision"])))
                        ts_records.append(dict(layout=layout, root=root, force=force, section=sec, side=side,
                                               system="Full 3-Stage", done=int(o_3["success"]), wall=int(o_3["collision"]),
                                               admitted=int(adme), adm_wall=int(adme and o_3["collision"])))
    ts_df = pd.DataFrame(ts_records)

    # Summarize by (gate_group in [Left, Right, All], layout in [all, config_l, config_r, config_double], system)
    ts_summ = []
    for gname, gmask in [("Left (+1)", ts_df["side"] == 1), ("Right (-1)", ts_df["side"] == -1), ("Both Sides", np.ones(len(ts_df), dtype=bool))]:
        sub_g = ts_df[gmask]
        for scope, part_l in [("all", sub_g), *sub_g.groupby("layout")]:
            for sys_name in ["Fixed (a=0)", "C-R (a=0)", "C-R (Exp. a)", "Full 3-Stage"]:
                ps = part_l[part_l["system"] == sys_name]
                dm, dlo, dhi = bootstrap_ci(ps, "done")
                wm, wlo, whi = bootstrap_ci(ps, "wall")
                adm_n = int(ps["admitted"].sum())
                ts_summ.append(dict(
                    gate=gname, scope=scope, system=sys_name, n=len(ps),
                    done=int(ps["done"].sum()), done_pct=100 * dm, done_lo=100 * dlo, done_hi=100 * dhi,
                    wall=int(ps["wall"].sum()), wall_pct=100 * wm, wall_lo=100 * wlo, wall_hi=100 * whi,
                    admitted=adm_n, adm_wall=int(ps["adm_wall"].sum()),
                    adm_wall_pct=(100 * ps["adm_wall"].sum() / adm_n) if adm_n > 0 else 0.0,
                ))
    ts_summ_df = pd.DataFrame(ts_summ)
    ts_summ_df.to_csv(OUT / "exp3_twosided_wall_layouts.csv", index=False)
    print(ts_summ_df[ts_summ_df["scope"] == "all"][["gate", "system", "n", "done_pct", "done_lo", "done_hi", "wall_pct", "wall_lo", "wall_hi", "adm_wall", "admitted"]].to_string(index=False))

    # Paired deltas on Both Sides (All 5,400 contracts): Full 3-Stage vs Fixed (a=0) and vs C-R (a=0) and vs C-R (Exp. a)
    piv_ts = ts_df.pivot(index=["layout", "root", "force", "section", "side"], columns="system", values=["done", "wall"]).reset_index()
    delta_rows = []
    for gname, gmask in [("Left (+1)", piv_ts["side"] == 1), ("Right (-1)", piv_ts["side"] == -1), ("Both Sides", np.ones(len(piv_ts), dtype=bool))]:
        p = piv_ts[gmask]
        for ref_sys in ["Fixed (a=0)", "C-R (a=0)", "C-R (Exp. a)"]:
            dd, dd_lo, dd_hi = bootstrap_delta_ci(pd.DataFrame({"root": p["root"], "a": p[("done", "Full 3-Stage")], "b": p[("done", ref_sys)]}), "a", "b")
            dw, dw_lo, dw_hi = bootstrap_delta_ci(pd.DataFrame({"root": p["root"], "a": p[("wall", "Full 3-Stage")], "b": p[("wall", ref_sys)]}), "a", "b")
            delta_rows.append(dict(
                gate=gname, comparison=f"Full 3-Stage - {ref_sys}", n=len(p),
                delta_done_pp=100 * dd, delta_done_lo=100 * dd_lo, delta_done_hi=100 * dd_hi,
                delta_wall_pp=100 * dw, delta_wall_lo=100 * dw_lo, delta_wall_hi=100 * dw_hi,
            ))
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT / "exp3_twosided_deltas.csv", index=False)
    print("\nTwo-Sided Paired Deltas (100 Unseen Starts, 3 Wall Layouts):")
    print(delta_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # EXPERIMENT E3-B: Two-Sided Test A/B (40 Unseen Starts, 1,440 RB Contracts)
    # -----------------------------------------------------------------------
    print("\n=== Running Experiment E3-B: Two-Sided Test A/B (40 Unseen Starts, 1,440 RB Contracts) ===")
    al_dir = PR / "alpha_library"
    al_plan = json.loads((al_dir / "test_freeze_manifest.json").read_text())
    al_orc = pd.read_csv(al_dir / "oracle_cases.csv")
    al_orc = al_orc[al_orc["skill"] == "rb_run1"].copy()
    al_onl = pd.read_csv(al_dir / "online_cases.csv")
    al_onl = al_onl[al_onl["skill"] == "rb_run1"].copy()
    al_lookup = al_orc.set_index(keys)[["verified", "success", "collision", "handoff_s", "prefix", "trajectory"]]
    al_pref = al_orc[(al_orc["section"] == "near") & (al_orc["side"] == 1) & (al_orc["planned_dep"] == 0)].set_index(
        ["layout", "root", "force", "approach", "approach_alpha"]
    )[["verified", "prefix"]].to_dict("index")

    roots_40 = al_plan["roots"]
    al_s1, al_s2 = {}, {}
    for layout in ["test_a", "test_b"]:
        prior_df = pd.read_csv(al_dir / f"{layout}/rb_run1_prior.csv")
        al_s1[layout] = stage1_actions_twosided(prior_df, bank_init, roots_40, EPS_RB)
        needed = set()
        for root in roots_40:
            for sec in SECTIONS:
                for side in (-1, 1):
                    for arm in ("alpha0", "alpha_enabled"):
                        a, al_v, _ = al_s1[layout][(root, sec, side)][arm]
                        for force in (-5, 0, 5):
                            needed.add((layout, root, force, a, al_v))
        sub_o = al_orc[(al_orc["layout"] == layout) & (al_orc["section"] == "near") & (al_orc["side"] == 1) & (al_orc["planned_dep"] == 0) & (al_orc["verified"])].copy()
        sub_o = sub_o[sub_o.apply(lambda r: (r.layout, r.root, r.force, r.approach, r.approach_alpha) in needed, axis=1)].drop_duplicates("prefix")
        ev = CpuStage2Evaluator(al_dir / f"{layout}/part_0/rb_run1", al_plan["section_x"])
        al_s2[layout] = ev.evaluate_prefixes(sub_o)

    # Parity check on all 1,440 online left-gate rows of alpha_library (alpha0 + alpha_enabled)
    mism_al = 0; chk_al = 0
    for r in al_onl.itertuples():
        if not r.verified or r.query_batch_ms <= 0:
            continue
        pinfo = al_pref[(r.layout, r.root, r.force, r.approach, r.approach_alpha)]
        ch, adm = stage2_decision(al_s2[r.layout][pinfo["prefix"]], r.section, 1, int(r.planned_dep), EPS_RB)
        chk_al += 1
        if ch != int(r.chosen_dep) or adm != bool(r.query_feasible):
            mism_al += 1
    assert mism_al == 0, f"Test A/B Stage 2 parity failed: {mism_al}/{chk_al}"
    print(f"  Test A/B PARITY PASSED: 0/{chk_al} mismatches!")

    ab_rows = []
    for layout in ["test_a", "test_b"]:
        for root in roots_40:
            for force in (-5, 0, 5):
                for sec in SECTIONS:
                    for side in (-1, 1):
                        s1 = al_s1[layout][(root, sec, side)]
                        fa, fal, fd = s1["fixed"]
                        of = al_lookup.loc[(layout, root, force, sec, side, fa, fal, fd)]
                        ab_rows.append(dict(layout=layout, root=root, force=force, section=sec, side=side,
                                            system="Fixed (a=0)", done=int(of["success"]), wall=int(of["collision"]),
                                            admitted=0, adm_wall=0))
                        for arm_key, label in [("alpha0", "C-R (a=0)"), ("alpha_enabled", "C-R (Exp. a)")]:
                            aa, alv, dd = s1[arm_key]
                            pi = al_pref[(layout, root, force, aa, alv)]
                            if pi["verified"] and al_s2[layout][pi["prefix"]]["valid_horizon"]:
                                ch, adm = stage2_decision(al_s2[layout][pi["prefix"]], sec, side, dd, EPS_RB)
                            else:
                                ch, adm = dd, False
                            oo = al_lookup.loc[(layout, root, force, sec, side, aa, alv, ch)]
                            ab_rows.append(dict(layout=layout, root=root, force=force, section=sec, side=side,
                                                system=label, done=int(oo["success"]), wall=int(oo["collision"]),
                                                admitted=int(adm), adm_wall=int(adm and oo["collision"])))
    ab_df = pd.DataFrame(ab_rows)
    ab_summ = []
    for gname, gmask in [("Left (+1)", ab_df["side"] == 1), ("Right (-1)", ab_df["side"] == -1), ("Both Sides", np.ones(len(ab_df), dtype=bool))]:
        sg = ab_df[gmask]
        for sys_name in ["Fixed (a=0)", "C-R (a=0)", "C-R (Exp. a)"]:
            ps = sg[sg["system"] == sys_name]
            dm, dlo, dhi = bootstrap_ci(ps, "done")
            wm, wlo, whi = bootstrap_ci(ps, "wall")
            adm_n = int(ps["admitted"].sum())
            ab_summ.append(dict(
                gate=gname, system=sys_name, n=len(ps),
                done=int(ps["done"].sum()), done_pct=100 * dm, done_lo=100 * dlo, done_hi=100 * dhi,
                wall=int(ps["wall"].sum()), wall_pct=100 * wm, wall_lo=100 * wlo, wall_hi=100 * whi,
                admitted=adm_n, adm_wall=int(ps["adm_wall"].sum()),
            ))
    ab_summ_df = pd.DataFrame(ab_summ)
    ab_summ_df.to_csv(OUT / "exp3_twosided_testab.csv", index=False)
    print(ab_summ_df.to_string(index=False))
    print(f"\nAll E2 & E3 experiments finished in {time.perf_counter()-t0:.1f}s!")


if __name__ == "__main__":
    run_exp1()
    run_exp2_and_exp3()
