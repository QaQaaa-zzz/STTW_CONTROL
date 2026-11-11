"""Evaluate a frozen event predictor on held-out executed segments, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sttw_control.ditch_event_model import EventPlanner


def evaluate(model_directory, collection):
    started = time.monotonic()
    planner = EventPlanner(model_directory)
    cfg = json.loads((collection / 'config.json').read_text())
    protocol = json.loads((collection / 'protocol.json').read_text())
    summary = json.loads((collection / 'evaluation/summary.json').read_text())
    data = dict(np.load(collection / 'evaluation/trajectories.npz'))
    if any(c['role'] != 'TEST' for c in protocol['cases']):
        raise ValueError('This entry point accepts TEST-only collections')
    hidden = np.concatenate((data['observation'], data['controller_history']), -1)
    codec = planner.codec
    def state(lane, t, root, road):
        ix = min(int(np.floor((t + 1e-6) / .02)), len(hidden) - 1)
        latent = ((hidden[ix, lane] - codec['hidden_mean']) / codec['hidden_std']) @ codec['components'].T
        return np.r_[root, latent, float(road)].astype(np.float32)
    source = Path(cfg['source_run'])
    raw_road = np.load(source / 'evaluation/trajectories.npz')['road_violation_history']
    prior_road = np.array([raw_road[max(0, round(c['published_time_s'] / .02) - 1), c['source_lane']] for c in protocol['cases']])
    parameters = np.asarray(cfg['event_candidate_parameters'], np.float32)[cfg['ground_handoff']['reference_route_ids']]
    ground = np.asarray(cfg['ground_handoff']['ground_only_lanes'])
    stride = planner.meta.get('prefix_stride_policy_steps', 0)
    records, initial = [], []
    for lane, case in enumerate(summary['cases']):
        end_ix = np.flatnonzero(data['physics_valid'][:, lane])[-1]
        end_t = float(data['physics_time'][end_ix])
        terminal = state(lane, end_t, data['physics_root_xyzw'][end_ix, lane], data['road_violation'][lane])
        first = state(lane, 0., data['initial_root_xyzw'][lane], prior_road[lane])
        initial.append(first)
        if ground[lane]:
            hit = float(case['success'])
            t = float(data['event_time'][lane, 4]) if hit else end_t
            target = state(lane, t, data['event_root_xyzw'][lane, 4], False) if hit else terminal
            records.append((lane, 4, first, target, t, hit))
            if stride:
                for ix in range(stride, case['steps'], stride):
                    at = ix * .02
                    if at < t:
                        records.append((lane, 4, state(lane, at, data['root_xyzw'][ix-1, lane], data['road_violation_history'][ix-1, lane]), target, t-at, hit))
            continue
        points = [(0., first)]
        t = case['entry_s']
        if t is None:
            points.append(None)
        else:
            ix = max(0, min(round(t/.02)-1, len(hidden)-1))
            points.append((t, state(lane, t, data['root_xyzw'][ix, lane], data['road_violation_history'][ix, lane])))
        if stride:
            nt, ns = points[1] if points[1] is not None else (end_t, terminal)
            for ix in range(stride, case['steps'], stride):
                at = ix * .02
                if at < nt:
                    records.append((lane, 0, state(lane, at, data['root_xyzw'][ix-1, lane], data['road_violation_history'][ix-1, lane]), ns, nt-at, float(points[1] is not None)))
        for event in (0, 2, 3, 4):
            t = float(data['event_time'][lane, event])
            points.append((t, state(lane, t, data['event_root_xyzw'][lane, event], data['event_road_violation'][lane, event])) if np.isfinite(t) else None)
        for mode in range(5):
            if points[mode] is None:
                break
            t, s = points[mode]
            nxt = points[mode+1]
            nt, ns = nxt if nxt is not None else (end_t, terminal)
            records.append((lane, mode, s, ns, max(0., nt-t), float(nxt is not None)))
            if nxt is None:
                break
        if points[1] is not None:
            t, s = points[1]
            nt = float(data['event_time'][lane, 1])
            hit = float(np.isfinite(nt))
            ns = state(lane, nt, data['event_root_xyzw'][lane, 1], data['event_road_violation'][lane, 1]) if hit else terminal
            records.append((lane, 5, s, ns, max(0., (nt if hit else end_t)-t), hit))
    lanes = np.array([r[0] for r in records]); modes = np.array([r[1] for r in records])
    s = np.stack([r[2] for r in records]); ns = np.stack([r[3] for r in records])
    dt = np.array([r[4] for r in records]); hit = np.array([r[5] for r in records])
    model = planner.model
    with torch.no_grad():
        inp = torch.from_numpy(s); mode = torch.from_numpy(modes)
        mean, logvar, logtime, logit = model(inp, torch.from_numpy(planner.plan_features(s, parameters[lanes])), torch.from_numpy(planner.map.crop(s[:, :13])), mode)
        pred_t = torch.expm1(logtime.clamp(0, np.log(11.)))
        pred = (model.physical_baseline(inp, pred_t, mode) + mean*model.residual_std + model.residual_mean).numpy()
        probability = torch.sigmoid(logit).numpy(); times = pred_t.numpy()
    training = np.load(model_directory / 'transitions.npz')
    train = training['role'] == 'TRAIN'
    rates = {m: float(training['hit'][train & (training['mode'] == m)].mean()) for m in range(6)}
    roots = np.array([c['root'] for c in protocol['cases']])
    def metrics(keep):
        result = {}
        for m in range(6):
            mask = keep & (modes == m); positive = mask & (hit > 0)
            result[str(m)] = dict(n=int(mask.sum()), events=int(positive.sum()), train_event_rate=rates[m], brier=float(np.mean((probability[mask]-hit[mask])**2)) if mask.any() else None, constant_brier=float(np.mean((rates[m]-hit[mask])**2)) if mask.any() else None, time_mae_s=float(np.mean(abs(times[positive]-dt[positive]))) if positive.any() else None, xy_mae_m=float(np.linalg.norm(pred[positive, :2]-ns[positive, :2], axis=1).mean()) if positive.any() else None)
        return result
    rollout = planner.score(np.stack(initial), parameters, particles=32)
    success = np.array([c['success'] for c in summary['cases']], float)
    completion = np.asarray(rollout['predicted_completion'])
    train_summary = json.loads((Path(planner.meta['collection']) / 'evaluation/summary.json').read_text())
    train_protocol = json.loads((Path(planner.meta['collection']) / 'protocol.json').read_text())
    base_rate = float(np.mean([c['success'] for c, p in zip(train_summary['cases'], train_protocol['cases']) if p['role'] == 'TRAIN']))
    cases = [dict(root=int(roots[i]), candidate=protocol['cases'][i]['candidate'], success=int(success[i]), predicted_completion=float(completion[i]), predicted_time_s=rollout['predicted_time_s'][i], prior_road_violation=bool(prior_road[i])) for i in range(len(initial))]
    return dict(model=str(model_directory), checkpoint_sha256=hashlib.sha256((model_directory/'model.pt').read_bytes()).hexdigest(), codec_sha256=hashlib.sha256((model_directory/'codec.npz').read_bytes()).hexdigest(), parameter_count=sum(p.numel() for p in model.parameters()), selected_update=planner.meta['selected_update'], collection=str(collection), independent_roots=len(set(roots)), executed_segments=len(initial), positive_segments=int(success.sum()), prior_road_violation_segments=int(prior_road.sum()), prefix_stride_policy_steps=stride, event_metrics=metrics(np.ones(len(lanes),bool)), event_metrics_by_root={str(r):metrics(roots[lanes] == r) for r in sorted(set(roots))}, autoregressive=dict(brier=float(np.mean((completion-success)**2)), train_constant=base_rate, constant_brier=float(np.mean((base_rate-success)**2)), particles=32), cases=cases, elapsed_s=time.monotonic()-started, scope='Frozen evaluation on previously observed TEST roots; no refitting or model selection. Conditional event metrics use actual preceding event states; completion uses initial autoregressive futures. Prefix records are correlated within four roots. Not an online planning comparison.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    result = evaluate(args.model, args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','event_metrics_by_root')}, indent=2))


if __name__ == '__main__':
    main()
