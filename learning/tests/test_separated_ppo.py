"""Verify SeparatedPPO decouples actor gradients from large failure-penalty critic losses."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, '/home/qy/projects/My_unitree_go2_gym/rsl_rl')
from rsl_rl.algorithms import PPO

from sttw_control.go2_ppo_adapter import H75ActorCritic, SeparatedPPO, init_near_zero_output
from sttw_control.torch_path_policy import make_torch_path_network


def make_policy(seed=0):
    cfg = {'network': {'architecture': 'alpha_modulated_v1'}}
    torch.manual_seed(seed)
    actor = make_torch_path_network(351, 2, cfg)
    critic = make_torch_path_network(352, 1, cfg)
    init_near_zero_output(actor)
    return H75ActorCritic(actor, critic, torch.tensor([0.25, 0.10]).log())


def fill_storage(alg, num_envs=16, steps=8, failure_reward=-4500.0):
    alg.init_storage(num_envs, steps, [351], [352], [2])
    torch.manual_seed(123)
    for t in range(steps):
        obs = torch.randn(num_envs, 351) * 0.2
        cobs = torch.cat([obs, torch.full((num_envs, 1), 1.0 - t / steps)], dim=-1)
        alg.act(obs, cobs)
        rewards = torch.full((num_envs,), failure_reward if t == steps - 1 else -0.5)
        dones = torch.full((num_envs,), float(t == steps - 1))
        alg.process_env_step(rewards, dones, {})
    last_cobs = torch.zeros(num_envs, 352)
    alg.compute_returns(last_cobs)


class SeparatedPPOTest(unittest.TestCase):
    def test_near_zero_init_keeps_initial_actions_small(self):
        policy = make_policy(7)
        obs = torch.randn(64, 351) * 0.5
        out = policy.act_inference(obs)
        self.assertLess(out.abs().max().item(), 0.05)

    def test_actor_grad_norm_is_invariant_to_large_critic_loss_unlike_joint_ppo(self):
        # Under joint clip_grad_norm_, large critic losses (~1e5) crush actor gradients by >100x.
        def post_clip_actor_grad_norm(use_separated, failure_reward):
            policy = make_policy(11)
            obs = torch.randn(64, 351) * 0.2
            cobs = torch.cat([obs, torch.full((64, 1), 0.5)], dim=-1)
            policy.act(obs)
            actions = policy.action_mean.detach() + 0.1 * torch.randn(64, 2)
            old_lp = policy.get_actions_log_prob(actions).detach()
            adv = torch.randn(64)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            returns = torch.full((64, 1), failure_reward)
            policy.act(obs)
            ratio = torch.exp(policy.get_actions_log_prob(actions) - old_lp)
            surr = torch.max(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)).mean()
            vloss = (returns - policy.evaluate(cobs)).pow(2).mean()
            actor_params = list(policy.actor.parameters()) + [policy.log_std]
            if use_separated:
                surr.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            else:
                (surr + vloss).backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            return torch.sqrt(sum(p.grad.square().sum() for p in actor_params)).item()

        joint_small = post_clip_actor_grad_norm(False, -1.0)
        joint_large = post_clip_actor_grad_norm(False, -4500.0)
        sep_small = post_clip_actor_grad_norm(True, -1.0)
        sep_large = post_clip_actor_grad_norm(True, -4500.0)

        self.assertLess(joint_large, 0.01 * joint_small)
        self.assertAlmostEqual(sep_small, sep_large, places=6)
        self.assertGreater(sep_large, 100.0 * joint_large)

    def test_critic_warmup_freezes_actor_and_clamps_std_afterward(self):
        policy = make_policy(19)
        sep = SeparatedPPO(policy, num_learning_epochs=2, num_mini_batches=2,
                           actor_lr=1e-4, critic_lr=3e-4, critic_warmup_updates=2,
                           std_min=0.05, std_max=0.5, gamma=0.997, lam=0.99,
                           entropy_coef=0.001, device='cpu')
        actor_before = policy.actor[-1].weight.detach().clone()
        critic_before = policy.critic[-1].weight.detach().clone()
        for _ in range(2):
            fill_storage(sep)
            sep.update()
        self.assertTrue(torch.equal(policy.actor[-1].weight, actor_before))
        self.assertFalse(torch.equal(policy.critic[-1].weight, critic_before))
        self.assertEqual(sep.last_actor_steps, 0)

        fill_storage(sep)
        sep.update()
        self.assertGreater(sep.last_actor_steps, 0)
        self.assertFalse(torch.equal(policy.actor[-1].weight, actor_before))
        std = policy.std.detach()
        self.assertTrue((std >= 0.05 - 1e-6).all() and (std <= 0.5 + 1e-6).all())

    def test_hard_kl_halts_actor_without_inflating_lr(self):
        policy = make_policy(23)
        sep = SeparatedPPO(policy, num_learning_epochs=4, num_mini_batches=4,
                           actor_lr=1e-2, critic_lr=3e-4, critic_warmup_updates=0,
                           hard_kl=1e-6, gamma=0.997, lam=0.99, device='cpu')
        fill_storage(sep)
        sep.update()
        self.assertEqual(sep.learning_rate, 1e-2)
        self.assertLess(sep.last_actor_steps, 16)


if __name__ == '__main__':
    unittest.main()
