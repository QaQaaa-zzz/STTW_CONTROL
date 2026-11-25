"""Train or continue the H75 architecture with Go2's actual runner/PPO/storage on the Lab task."""
import argparse,importlib.util,json,os,shutil,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume-checkpoint',type=Path);p.add_argument('--completed-updates',type=int,default=0);p.add_argument('--profile-first-step',action='store_true')
p.add_argument('--actor-architecture',choices=('legacy','alpha_modulated_v1'),help='Actor-only override; legacy concatenates alpha at input 336 and retains the reference critic')
p.add_argument('--reference-action-channel',choices=('speed','steer'),help='Train one stochastic correction, preserving the nominal follower on the other reference')
a=p.parse_args();s=json.loads(a.config.read_text())
if a.reference_action_channel is not None:
    s['reference_action_channel']=a.reference_action_channel
if a.actor_architecture is not None:
    s['actor_architecture'] = a.actor_architecture
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
s['profile_first_step']=a.profile_first_step;s['profile_output']=str(a.output/'sampling_profile.txt')
sys.path.insert(0,str(Path(s['go2_checkout'])/'rsl_rl'))
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device=s['device']).app
import torch
from sttw_control.go2_training_config import spring_jump_recipe
from sttw_control.go2_path_env import Go2PathEnv
from sttw_control.go2_ppo_adapter import H75ActorCritic, SeparatedPPO, init_near_zero_output
runner_path=Path(s['go2_checkout'])/'rsl_rl/rsl_rl/runners/on_policy_runner.py';spec=importlib.util.spec_from_file_location('sttw_go2_path_runner',runner_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.SeparatedPPO = SeparatedPPO
policy_config=json.loads((Path(s['bundle'])/'config.json').read_text())
actor_architecture = s.get('actor_architecture', policy_config['network'].get('architecture', 'legacy'))
actor_config = dict(policy_config, network=dict(policy_config['network'], architecture=actor_architecture))
if s.get('policy_initialization') == 'fresh':
    from sttw_control.torch_path_policy import make_torch_path_network

    def new_policy(*args, **kwargs):
        # The runner constructs its policy after the physical environment.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(s['seed'])
            actor = make_torch_path_network(351, 2, actor_config)
            critic = make_torch_path_network(352, 1, policy_config)
            if s.get('near_zero_output_init', False):
                init_near_zero_output(actor)
        std = policy_config['training_future_phase_C']['std_initial']
        return H75ActorCritic(actor, critic, torch.tensor(std).log(), active_channel=s.get('reference_action_channel')).to(s['device'])

    module.STTWPathActorCritic = new_policy
    source = dict(initialization='fresh Torch linear layers', near_zero_output_init=bool(s.get('near_zero_output_init', False)), seed=s['seed'])
    initial_description = 'Seeded fresh actor and critic with configured initial exploration std'
else:
    if actor_architecture != policy_config['network'].get('architecture', 'legacy'):
        raise ValueError('A different actor architecture requires fresh initialization or its own fresh-run resume config')
    original = torch.load(s['initial_policy'], map_location='cpu', weights_only=False)
    module.STTWPathActorCritic = lambda *args, **kwargs: H75ActorCritic.from_original_policy(policy_config, original['policy'], s['device'], active_channel=s.get('reference_action_channel'))
    source = {k:v for k,v in original.items() if k != 'policy'}
    initial_description = 'original H75 u77 Actor/Critic/log_std weights; fresh optimizer'
recipe=spring_jump_recipe(s['go2_checkout']);train=recipe['train'];train['algorithm'].update(s['ppo_overrides']);train['runner'].update(policy_class_name='STTWPathActorCritic',algorithm_class_name=s.get('algorithm_class_name','PPO'),num_steps_per_env=s['rollout_steps'],save_interval=s['save_interval'],max_iterations=s['updates'],experiment_name='sttw_h75_offroad')
(a.output/'config.json').write_text(json.dumps(dict(task=s,upstream=recipe,initial_policy_source=source,role=s.get('role','original H75 TRAIN prefix continuation plus switch/terrain/Go2 DR; not independent TEST'),resume_checkpoint=str(a.resume_checkpoint.resolve()) if a.resume_checkpoint else None,resume_completed_updates=a.completed_updates,resume_semantics='weights and optimizer restored; task states and declared RNG resampled' if a.resume_checkpoint else initial_description),indent=2))
torch.manual_seed(s['seed']);torch.set_num_threads(2);started=time.time();status=dict(stage='initializing',pid=os.getpid(),completed_updates=a.completed_updates,started=started,updates=s['updates']);(a.output/'status.json').write_text(json.dumps(status,indent=2))
snapshot=a.output/'source';snapshot.mkdir()
for file in ['learning/cli/train_go2_path.py','learning/src/sttw_control/go2_path_env.py','learning/src/sttw_control/go2_ppo_adapter.py','learning/src/sttw_control/reference_action_channel.py','learning/src/sttw_control/isaac_path_rollout.py','learning/src/sttw_control/isaaclab_vector_env.py','learning/src/sttw_control/isaaclab_randomization.py','learning/src/sttw_control/torch_middle_control.py','learning/src/sttw_control/torch_path_reference.py','learning/src/sttw_control/torch_path_policy.py','learning/src/sttw_control/torch_path_reward.py',str(runner_path)]:shutil.copyfile(file,snapshot/Path(file).name)
class RecordedRunner(module.OnPolicyRunner):
    best=-float('inf')
    def save(self, path, infos=None):
        payload = {
            'actor_architecture': actor_architecture,
            'reference_action_channel': s.get('reference_action_channel'),
            'critic_architecture': policy_config['network'].get('architecture', 'legacy'),
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if hasattr(self.alg, 'critic_optimizer'):
            payload['critic_optimizer_state_dict'] = self.alg.critic_optimizer.state_dict()
            payload['completed_updates'] = self.alg.completed_updates
        torch.save(payload, path)
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=False)
        if loaded_dict.get('reference_action_channel')!=s.get('reference_action_channel'):
            raise ValueError('Resume checkpoint uses a different reference action channel')
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if hasattr(self.alg, 'critic_optimizer') and 'critic_optimizer_state_dict' in loaded_dict:
                self.alg.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer_state_dict'])
        if hasattr(self.alg, 'completed_updates') and 'completed_updates' in loaded_dict:
            self.alg.completed_updates = loaded_dict['completed_updates']
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']
    def log(self,locs,width=80,pad=35):
        super().log(locs,width,pad)
        info=dict(stage='training',completed_updates=locs['it']+1,elapsed_s=time.time()-started,sample_seconds=locs['collection_time'],optimize_seconds=locs['learn_time'],push_events=env.rollout.push_events)
        if hasattr(self.alg, 'last_kl'):
            info.update(last_kl=self.alg.last_kl, last_actor_steps=self.alg.last_actor_steps)
        if env.rollout.channel_audit is not None:
            ticks,target,filtered,applied,active=env.rollout.channel_audit.tolist()
            signed,absolute=env.rollout.channel_offset_sums.tolist()
            info['reference_action_audit']=dict(channel=s['reference_action_channel'],ppo_action_dim=env.num_actions,
                lower_ticks=ticks,disabled_target_nonzero=target,disabled_filter_nonzero=filtered,
                disabled_offset_nonzero=applied,enabled_offset_nonzero=active,
                enabled_offset_signed_sum=signed,enabled_offset_abs_sum=absolute,
                offset_unit='m/s' if env.action_index==0 else 'rad')
            with (a.output/'reference_action_audit.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(completed_updates=locs['it']+1,**info['reference_action_audit']))+'\n')
            if target or filtered or applied:
                raise RuntimeError(f'Single-channel correction violated its action permission: {info["reference_action_audit"]}')
            self.writer.add_scalar('Audit/single_channel_lower_ticks',ticks,locs['it'])
            self.writer.add_scalar('Audit/enabled_offset_nonzero_ticks',active,locs['it'])
        if env.fixed_preference_counts is not None:
            actual, wrong = env.fixed_preference_counts.tolist()
            info.update(fixed_preference=s['alphas'][0], valid_transitions=actual, wrong_preference_transitions=wrong)
            if wrong:
                raise RuntimeError(f'{wrong} valid transitions used a different preference')
            self.writer.add_scalar('Audit/valid_fixed_preference_transitions', actual, locs['it'])
        status.update(info);(a.output/'status.json').write_text(json.dumps(status,indent=2))
        if locs['ep_infos']:
            score=float(torch.stack([x['reward'].reshape(()) for x in locs['ep_infos']]).mean());row=dict(completed_updates=locs['it']+1,episode_reward=score,scope='completed-episode reward peak candidate; episodes span policy updates, not fixed-case best')
            with (a.output/'episode_rewards.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if score>self.best:
                self.best=score;self.save(a.output/'training/model_best.pt',infos=row);(a.output/'best_model.json').write_text(json.dumps(dict(row,checkpoint='training/model_best.pt'),indent=2))
env=Go2PathEnv(s,a.output);(a.output/'randomized_properties.json').write_text(json.dumps(env.rollout.randomization_properties));(a.output/'plant.json').write_text(json.dumps(env.plant.metadata));runner=RecordedRunner(env,train,log_dir=str(a.output/'training'),device=s['device']);(a.output/'training').mkdir(exist_ok=True)
if a.resume_checkpoint:
    runner.load(str(a.resume_checkpoint),load_optimizer=not s.get('reset_optimizer_on_resume', False))
    if hasattr(runner.alg, 'critic_optimizer') and 'actor_lr' in s['ppo_overrides']:
        for pg in runner.alg.actor_optimizer.param_groups:pg['lr'] = s['ppo_overrides']['actor_lr']
        for pg in runner.alg.critic_optimizer.param_groups:pg['lr'] = s['ppo_overrides']['critic_lr']
        runner.alg.learning_rate = s['ppo_overrides']['actor_lr']
    else:
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
    runner.current_learning_iteration=a.completed_updates
runner.save(a.output/'training/model_initial.pt',infos={'completed_updates':a.completed_updates,'source':str(a.resume_checkpoint) if a.resume_checkpoint else initial_description})
if s.get('initial_weight_reference'):
    reference = torch.load(s['initial_weight_reference'], map_location='cpu', weights_only=False)['model_state_dict']
    current = runner.alg.actor_critic.state_dict()
    if current.keys() != reference.keys() or not all(torch.equal(v.cpu(), reference[k]) for k,v in current.items()):
        raise ValueError('Fresh actor/critic parameters differ from the paired shared initialization')
    (a.output/'initial_weight_check.json').write_text(json.dumps(dict(reference=s['initial_weight_reference'], arrays=len(reference), elements=sum(v.numel() for v in reference.values()), identical=True),indent=2))
runner.learn(num_learning_iterations=s['updates']-a.completed_updates,init_at_random_ep_len=False)
runner.writer.close()
status.update(stage='training_complete',completed_updates=s['updates'],elapsed_s=time.time()-started,final_checkpoint=f'training/model_{s["updates"]}.pt');(a.output/'status.json').write_text(json.dumps(status,indent=2));app.close(wait_for_replicator=False,skip_cleanup=True)
