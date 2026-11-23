"""Measure the Chapter 3 continuous exit planner from one saved DEV robot observation."""
import argparse
from copy import copy
import csv
import gc
import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from sttw_control.chapter3_continuous_exit import ContinuousExitPlanner, committed_progress
from sttw_control.platform_online import velocity_state
from sttw_control.playground_adapters import GroundAdapter
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_prediction import BodyEnvelope
from sttw_control.playground_terrain_features import TerrainFeatures
from sttw_control.torch_middle_control import TorchMiddleControl


def load_fixture(part, trajectory, device, index):
    cfg = json.loads((part / 'resolved_config.json').read_text())
    policy, hand = cfg['platform_online'], cfg['ground_handoff']
    rel = policy['relative']
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    model = load_history_model(Path(policy['ground_model']), device=device)
    with np.load(rel['raster']) as z:
        raster = dict(z)
    raster['height'] -= rel['height']
    model.terrain = TerrainFeatures(**raster, device=device)
    physics = mujoco.MjModel.from_binary_path(str(part / 'terrain.mjb'))
    walls = [i for i in range(physics.ngeom) if physics.geom(i).name.startswith('planning_obstacle_')]
    with np.load(rel['initializer']) as z:
        mu, scale, weights = (tensor(z[key]) for key in ('mu', 'scale', 'weight'))
    query = SimpleNamespace(model=model, relative=rel, goal=hand['goal'], margin=policy['margin_m'],
                            mean=mu, scale=scale, weights=weights, halfwidth=hand['road_half_width_m'],
                            envelope=BodyEnvelope(physics, device=device, dimensions=3),
                            center=tensor(physics.geom_pos[walls]), half=tensor(physics.geom_size[walls]),
                            pose_order=torch.tensor([0,1,2,3,4,5,6,8,9,7,10,11], device=device))
    with np.load(trajectory) as z:
        cols = z['columns'].tolist()
        trace = z['trace']
        if index is None:
            eligible = np.flatnonzero((trace[:, cols.index('phase')] == 2) & (np.arange(len(trace)) % 4 == 0))
            if not len(eligible):
                raise ValueError('The DEV trace never reaches a 20 ms post-relock boundary')
            index = int(eligible[0])
        if index < 4 or trace[index, cols.index('phase')] != 2 or index % 4:
            raise ValueError('Index must select a post-relock 20 ms boundary with previous velocity')
        q, v = tensor(z['qpos'][[index, index - 4]]), tensor(z['qvel'][[index, index - 4]])
        recorded_xy = z['qpos'][index:, :2]
        prior_x = z['qpos'][:index+1, 0].max()
        active_xy = z['published_reference_xy']
    with np.load(cfg['paired_departures']['route_bank']) as z:
        mapped = np.flatnonzero((z['xy'] == active_xy).all(axis=(1, 2)))
        if len(mapped) == 0:
            raise ValueError('The recorded published route does not belong to this configuration')
        row = int(mapped[0])
        route = SimpleNamespace(s=tensor(z['s']), xy=tensor(active_xy[None]), batched=True,
                                heading=tensor(z['heading'][row:row+1]), curvature=tensor(z['curvature'][row:row+1]),
                                route_ids=torch.zeros(1, device=device, dtype=torch.long))
    return query, route, q, v, recorded_xy, index, float(trace[index, cols.index('time')]), float(prior_x)


@torch.inference_mode()
def run(args):
    device = torch.device(args.device)
    query, route, q, v, xy, tick, sample_time, prior_x = load_fixture(args.part, args.trajectory, device, args.index)
    if prior_x >= min(args.section, query.goal['x_min_m']):
        raise ValueError('Choose a DEV snapshot before the first gate crossing and terminal hold')
    # Exercise the same prefix guard / physical-clock contract as a 20 ms publication,
    # without mutating a live controller or substituting replay for closed-loop evidence.
    middle = TorchMiddleControl.__new__(TorchMiddleControl)
    middle.tick = tick
    middle.v_user = q.new_full((1,), 2.)
    middle.route = route
    adapter = GroundAdapter(middle, None)
    follower = query.model.nominal.follower
    samples, stats = [], []
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    args.out.mkdir(exist_ok=True, parents=True)
    for count in args.batch:
        planner = ContinuousExitPlanner(query, route, candidates=count, iterations=args.iterations,
                                        horizon_steps=args.steps, fused=device.type == 'cuda')
        for i in range(args.warmup + args.repetitions):
            middle.route = route
            side = args.side or (1 if i % 2 == 0 else -1)
            start = perf_counter_ns()
            twists = velocity_state(q, v)
            idx = (route.xy[0] - q[0, :2]).square().sum(-1).argmin().view(1)
            progress = route.s[idx]
            look = (twists[:1, 0].clamp_min(follower['lookahead_min_speed_m_s']) * follower['lookahead_time_s'])
            look = look.clamp(follower['lookahead_min_m'], follower['lookahead_max_m']) * follower.get('lookahead_scale', 1.)
            committed = committed_progress(progress, look, middle.v_user, args.budget_ms / 1000).clamp_max(route.s[-1])
            middle.projection, middle.follower = {'progress': progress}, {'lookahead': look}
            result = planner.plan(q[:1], twists[:1], twists[1:], route, committed, args.section, side,
                                  10. - sample_time)
            admitted = bool(result.admitted[0])
            if admitted:
                replacement = copy(route)
                replacement.xy, replacement.heading, replacement.curvature = result.xy, result.heading, result.curvature
                adapter.update_plan(replacement, committed, middle.tick * .005, adapter.plan_version)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = (perf_counter_ns() - start) / 1e6
            if i < args.warmup:
                continue
            gpu_ms = None
            if device.type == 'cuda':
                # The captured graph measures one frozen rollout AND Stage 2 scoring pass;
                # candidate generation, CEM iterations and dispatch are in elapsed above.
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                planner.graph.replay()
                end.record()
                end.synchronize()
                gpu_ms = begin.elapsed_time(end)
            delay_ticks = max(4, int(np.ceil(elapsed / 20.)) * 4)
            samples.append(dict(candidates=count, iterations=args.iterations, repetition=i - args.warmup,
                                side=side, model_score_gpu_ms=gpu_ms, obs_to_publish_ms=elapsed, admitted=admitted,
                                deadline_ms=args.budget_ms, deadline_missed=elapsed > args.budget_ms,
                                schedule_delay_s=delay_ticks * .005,
                                recorded_old_route_travel_m=float(distance[delay_ticks]) if delay_ticks < len(distance) else None))
        batch = samples[-args.repetitions:]
        times = [v['obs_to_publish_ms'] for v in batch]
        quantiles = lambda values: dict(zip(('p50','p95','p99'), map(float, np.percentile(values, [50,95,99]))))
        stats.append(dict(candidates=count, obs_to_publish_ms=quantiles(times),
                          model_score_gpu_ms=quantiles([v['model_score_gpu_ms'] for v in batch]) if device.type=='cuda' else None,
                          deadline_missed=sum(v['deadline_missed'] for v in batch), n=len(batch)))
        with (args.out / 'latency_samples.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=samples[0]); writer.writeheader(); writer.writerows(samples)
        receipt = dict(device=str(device), part=str(args.part), trajectory=str(args.trajectory), tick=tick,
                       deadline_ms=args.budget_ms, budget_ms=args.budget_ms, steps=args.steps, iterations=args.iterations,
                       warmup=args.warmup, section=args.section, side=args.side, batches=stats,
                       scope='Replay observation-to-publication/rejection as measured on a saved DEV state. Physics observation extraction excluded. Recorded travel follows original ground path.')
        (args.out / 'summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(stats[-1], flush=True)
        del planner
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', type=Path, help='DEV run directory with resolved_config.json and terrain.mjb')
    p.add_argument('trajectory', type=Path, help='One verified DEV trajectory.npz from that same run')
    p.add_argument('out', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--batch', type=int, nargs='+', default=[16,64,256,512,1024])
    p.add_argument('--steps', type=int, default=255)
    p.add_argument('--iterations', type=int, default=2)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repetitions', type=int, default=100)
    p.add_argument('--budget-ms', type=float, default=100.)
    p.add_argument('--section', type=float, default=12.7)
    p.add_argument('--side', type=int, choices=(-1, 0, 1), default=0, help='0 alternates gate sides on the same observation')
    p.add_argument('--index', type=int)
    args = p.parse_args()
    if args.repetitions <= 0 or args.warmup < 0:
        p.error('Expected positive repetitions and nonnegative warmup')
    run(args)
