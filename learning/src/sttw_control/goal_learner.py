"""Single-consumption ordinary goal-PPO lifecycle, not a safe-control guarantee."""
import copy
import hashlib
import json
import math
import numpy as np
import torch
from .goal_population import assemble_goal_population
from .goal_networks import goal_network_ppo_terms


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class GoalLearner:
    """Freeze full sampling identity, retain every declared episode, then update.

    Caller authenticates physics, raw files, original user/path and exact source
    inventory. IDs/recording rechecks cannot certify a deliberately false source.
    Objective weights are explicit; this is not lexicographic or constrained RL.
    """
    def __init__(self, bundle, *, objective_weights, epochs, actor_lr, value_lr,
                 clip, entropy_coef, grad_clip, actor_aggregation='scalarize_then_clip', cost_target_scale=1.):
        if (isinstance(cost_target_scale, bool) or not isinstance(cost_target_scale, (int, float)) or
                not math.isfinite(cost_target_scale) or cost_target_scale <= 0):
            raise ValueError('finite positive explicit cost target scale required')
        if actor_aggregation not in ('scalarize_then_clip', 'componentwise_clip'):
            raise ValueError('explicit supported Actor aggregation required')
        if (set(objective_weights) != {'goal', 'cost', 'survival'} or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                    not math.isfinite(x) or x < 0 for x in objective_weights.values()) or
                not any(objective_weights.values()) or isinstance(epochs, bool) or
                not isinstance(epochs, int) or epochs <= 0 or
                not all(math.isfinite(x) and x > 0 for x in (actor_lr, value_lr, grad_clip)) or
                not math.isfinite(clip) or not 0 < clip < 1 or
                not math.isfinite(entropy_coef) or entropy_coef < 0):
            raise ValueError('explicit valid ordinary PPO settings required')
        self.bundle = bundle
        self.settings = dict(objective_weights=copy.deepcopy(objective_weights), epochs=epochs,
            actor_lr=actor_lr, value_lr=value_lr, clip=clip, entropy_coef=entropy_coef, grad_clip=grad_clip,
            actor_aggregation=actor_aggregation, cost_target_scale=cost_target_scale)
        self.actor_parameters = list(bundle.actor.parameters()) + list(bundle.goal_branch.parameters()) + [bundle.log_std]
        self.value_parameters = list(bundle.values.parameters())
        self.optimizers = [torch.optim.Adam(self.actor_parameters, lr=actor_lr),
                           torch.optim.Adam(self.value_parameters, lr=value_lr)]
        self.optimizer_steps = [0, 0]
        self.collection_index = 0
        self._pending = None
        self._batch = None

    def fingerprint(self):
        h = hashlib.sha256(_json(dict(config=self.bundle.config, settings=self.settings)).encode())
        for name, value in sorted(self.bundle.state_dict().items()):
            a = value.detach().cpu().numpy()
            h.update((name + str(a.shape) + str(a.dtype)).encode()); h.update(a.tobytes())
        return h.hexdigest()

    def freeze_collection(self, manifest, *, horizon_s, radius_m):
        """manifest entries bind episode_id, original_goal_xy and original case."""
        if self._pending is not None:
            raise ValueError('previous collection is unresolved, unconsumed or failed')
        if not manifest or not all(set(e) == {'episode_id', 'original_goal_xy', 'case'} for e in manifest):
            raise ValueError('explicit original-goal and case inventory required')
        ids = [e['episode_id'] for e in manifest]
        if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('nonempty unique episode IDs required')
        if not all(math.isfinite(x) and x > 0 for x in (horizon_s, radius_m)):
            raise ValueError('positive original horizon and radius required')
        for e in manifest:
            goal = np.asarray(e['original_goal_xy'], float)
            if goal.shape != (2,) or not np.isfinite(goal).all() or not isinstance(e['case'], dict):
                raise ValueError('finite original world goal and explicit original case required')
        frozen_manifest = copy.deepcopy(manifest)
        inventory_sha = hashlib.sha256(_json(frozen_manifest).encode()).hexdigest()
        fingerprint = self.fingerprint()
        identity = hashlib.sha256(_json([fingerprint, inventory_sha, self.collection_index,
                                        horizon_s, radius_m]).encode()).hexdigest()
        self._pending = dict(phase='collecting', collection_id=identity,
            fingerprint=fingerprint, manifest=frozen_manifest, inventory_sha256=inventory_sha,
            horizon_s=horizon_s, radius_m=radius_m)
        return dict(collection_id=identity, parameter_fingerprint=fingerprint,
            inventory_sha256=inventory_sha, collection_index=self.collection_index,
            cost_target_scale=self.settings['cost_target_scale'],
            parameters=self.bundle.sampling_parameters())

    def prepare_population(self, episodes):
        """Bind exact case IDs/metadata; retain an internally owned tensor batch."""
        p = self._pending
        if p is None or p['phase'] != 'collecting':
            raise ValueError('one fresh collecting snapshot required')
        if self.fingerprint() != p['fingerprint']:
            raise ValueError('sampling parameters/config/settings changed during collection')
        by_id = {e['episode_id']: e for e in p['manifest']}
        ids = [e['episode_id'] for e in episodes]
        if len(ids) != len(by_id) or set(ids) != set(by_id) or len(set(ids)) != len(ids):
            raise ValueError('exact complete frozen episode inventory required, no omissions or replacements')
        for e in episodes:
            original = by_id[e['episode_id']]
            if _json(e['case']) != _json(original['case']) or _json(e['original_goal_xy']) != _json(original['original_goal_xy']):
                raise ValueError('original case/goal changed after collection was frozen')
        batch = assemble_goal_population(episodes, expected_episode_count=len(by_id),
            config=self.bundle.config, collection_id=p['collection_id'], learner_id=p['collection_id'],
            horizon_s=p['horizon_s'], radius_m=p['radius_m'],
            cost_target_scale=self.settings['cost_target_scale'])
        p['population_diagnostics'] = copy.deepcopy(batch['summaries'])
        if any(r['survival'] == 'unknown' for r in batch['summaries']):
            p['phase'] = 'error'; p['error'] = 'engineering unknown population; preserve and stop'
            raise ValueError('engineering unknown population preserved; do not drop or recollect it')
        if any(r['survival'] == 'pending' for r in batch['summaries']):
            raise ValueError('complete known physical population required; pending is not failure')
        if self.settings['objective_weights']['cost'] > 0 and not batch['cost_population_complete']:
            p['phase'] = 'rejected_missing_cost'
            raise ValueError('complete original-cost population required for positive cost weight')
        union = batch['masks']['goal'] | batch['masks']['cost'] | batch['masks']['survival']
        if not bool(union.any()):
            p['phase'] = 'untrainable_no_actions'
            raise ValueError('no resolved actual Actor decisions; do not update')
        data = [batch[k][union] for k in ('observation', 'critic_observation', 'context', 'actions')]
        if not all(bool(torch.isfinite(x).all()) for x in data):
            raise ValueError('finite selected actual decision data required')
        with torch.no_grad():
            new = self.bundle.evaluate_actions(*data)
        differences = {}
        for name, actual in zip(('old_log_prob', 'old_goal_values', 'old_cost_values', 'old_physical_values'), new):
            old = batch['collected'][name][union]
            if not bool(torch.isfinite(old).all() & torch.isfinite(actual).all()):
                raise ValueError('finite actual frozen collection values required')
            differences[name] = float((actual-old).abs().max())
        # Same minor CPU/GPU recording tolerances as the real collector check;
        # no historical state replay or extra physical rollout is performed.
        if differences['old_log_prob'] > 1e-2 or any(
                differences[k] > 1e-3 for k in differences if k != 'old_log_prob'):
            raise ValueError('frozen collection density/value mismatch beyond minor roundoff')
        self._batch = batch  # assemble owns new tensor storage; caller gets no batch reference.
        p['phase'] = 'prepared'
        p['recording_differences'] = differences
        return dict(collection_id=p['collection_id'], fixed_denominator=batch['fixed_denominator'],
            summaries=copy.deepcopy(batch['summaries']), recording_differences=differences)

    def pending_status(self):
        """Inspectable detached status, including rejected/unknown episode summaries."""
        return copy.deepcopy(self._pending)

    def update(self, *, warmup):
        """One consumption only, with fixed old values for every PPO epoch.

        An optimizer failure leaves an error phase and partial state for caller
        preservation. It cannot be silently retried as a fresh collection.
        """
        p, b = self._pending, self._batch
        if not isinstance(warmup, bool) or p is None or p['phase'] != 'prepared' or b is None:
            raise ValueError('one prepared unconsumed population and explicit warmup required')
        if self.fingerprint() != p['fingerprint']:
            raise ValueError('learner changed before update')
        p['phase'] = 'updating'; p['optimizer_steps_before'] = list(self.optimizer_steps); p['optimizer_call_started'] = False; history = []
        try:
            for epoch in range(self.settings['epochs']):
                for opt in self.optimizers: opt.zero_grad(set_to_none=True)
                terms = goal_network_ppo_terms(self.bundle, b['observation'], b['critic_observation'],
                    b['context'], b['actions'], masks=b['masks'], collected=b['collected'],
                    objective_weights=self.settings['objective_weights'], fixed_denominator=b['fixed_denominator'],
                    cost_population_complete=b['cost_population_complete'], clip=self.settings['clip'],
                    actor_aggregation=self.settings['actor_aggregation'])
                # Empty-objective placeholders can be tied to logp. Exclude
                # them so value-only warmup never reaches Actor via zero grads.
                value = sum(loss for name, loss in terms['value_losses'].items()
                            if bool(b['masks'][name].any()))
                union = b['masks']['goal'] | b['masks']['cost'] | b['masks']['survival']
                count = int(union.sum())
                entropy = (self.bundle.log_std + .5 * math.log(2 * math.pi * math.e)).sum() * count / b['fixed_denominator']
                total = value if warmup else value + terms['actor_loss'] - self.settings['entropy_coef'] * entropy
                if not bool(torch.isfinite(total)): raise ValueError('nonfinite goal-PPO loss')
                total.backward()
                for i, (group, opt) in enumerate(zip((self.actor_parameters, self.value_parameters), self.optimizers)):
                    if warmup and i == 0:
                        if any(x.grad is not None for x in group): raise ValueError('critic warmup reached Actor')
                        continue
                    grads = [x for x in group if x.grad is not None]
                    if not grads or not all(bool(torch.isfinite(x.grad).all()) for x in grads):
                        raise ValueError('missing or nonfinite optimizer gradients')
                    torch.nn.utils.clip_grad_norm_(grads, self.settings['grad_clip'], error_if_nonfinite=True)
                    p['optimizer_call_started'] = True
                    opt.step(); self.optimizer_steps[i] += 1
                if not all(bool(torch.isfinite(x).all()) for x in self.bundle.parameters()):
                    raise ValueError('nonfinite updated goal network parameters')
                history.append(dict(epoch=epoch+1, actor=float(terms['actor_loss'].detach()), value=float(value.detach())))
        except BaseException as exc:
            p['phase'] = 'error'; p['error'] = repr(exc)
            p['optimizer_steps_after'] = list(self.optimizer_steps)
            p['step_outcome'] = ('may_be_partial; preserve state, no implicit retry'
                                 if p['optimizer_call_started'] else 'no_optimizer_step_started')
            raise
        receipt = dict(collection_id=p['collection_id'], inventory_sha256=p['inventory_sha256'],
            completed_collection=self.collection_index, warmup=warmup, history=history,
            optimizer_steps=list(self.optimizer_steps), post_fingerprint=self.fingerprint())
        self.collection_index += 1; self._pending = None; self._batch = None
        return receipt

    def checkpoint(self):
        """Owned learner state; pending state is forensic, not resumable here.

        Physical carry/RNG/raw files belong to the coordinator and are not
        implied by a learner checkpoint. No automatic resume or restart occurs.
        """
        return copy.deepcopy(dict(schema='sttw_goal_learner_v2', config=self.bundle.config,
            settings=self.settings, network=self.bundle.state_dict(),
            optimizers=[opt.state_dict() for opt in self.optimizers],
            optimizer_steps=self.optimizer_steps, collection_index=self.collection_index,
            pending=self.pending_status(), fingerprint=self.fingerprint()))

    @classmethod
    def restore(cls, checkpoint):
        """Restore completed-batch learner/Adam state only, not physical state."""
        from .path_command_policy import make_torch_path_network
        from .goal_networks import GoalNetworkBundle
        if checkpoint.get('schema') != 'sttw_goal_learner_v2' or checkpoint.get('pending') is not None:
            raise ValueError('v2 completed-batch checkpoint required; v1 is not silently migrated and pending/error is forensic only')
        if 'cost_target_scale' not in checkpoint['settings']:
            raise ValueError('explicit checkpoint C units required; old raw-C checkpoints are not silently migrated')
        config = copy.deepcopy(checkpoint['config'])
        parent = make_torch_path_network(351, 2, config)
        bundle = GoalNetworkBundle(parent, config, checkpoint['network']['log_std'])
        bundle.load_state_dict(checkpoint['network'], strict=True)
        obj = cls(bundle, **copy.deepcopy(checkpoint['settings']))
        if len(checkpoint['optimizers']) != 2:
            raise ValueError('both actual Adam states required')
        for opt, saved in zip(obj.optimizers, checkpoint['optimizers']):
            opt.load_state_dict(copy.deepcopy(saved))
        counts = checkpoint['optimizer_steps']
        index = checkpoint['collection_index']
        if (not isinstance(counts, list) or len(counts) != 2 or
                any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts) or
                isinstance(index, bool) or not isinstance(index, int) or index < 0):
            raise ValueError('nonnegative actual optimizer/collection counts required')
        obj.optimizer_steps = list(counts); obj.collection_index = index
        if obj.fingerprint() != checkpoint['fingerprint']:
            raise ValueError('restored learner identity mismatch')
        return obj
