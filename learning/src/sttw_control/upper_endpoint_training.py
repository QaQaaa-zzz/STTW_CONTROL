"""Independent alpha endpoints on registered frozen310-dim lower controllers."""
from pathlib import Path
import json,time
from .direct_command_training import Campaign,write,notify
from .direct_command_budget import ComputeBudget,BudgetStop

def load_endpoint_spec(path):
    spec=json.loads(Path(path).read_text())
    if spec.get('upper_alpha') not in (0,1) or not spec.get('lower_reference_centered'):raise ValueError('independent endpoint contract')
    if spec['lower_controller']['alias'] not in ('STTW_R196_ALPHA1','STTW_R244_ALPHA1') or spec['lower_controller']['alpha']!=1.:raise ValueError('registered ALPHA1 required')
    if spec['lower_controller']['path_capacity']!=3201:raise ValueError('16s needs3201 causal points')
    expected=make_spec(spec['lower_controller']['alias'],spec['upper_alpha'])
    if spec!=expected:raise ValueError('changed frozen endpoint configuration')
    return spec

def make_spec(alias,alpha):
    root=Path(__file__).resolve().parents[2]/'configs'
    s=json.loads((root/'direct_command_frozen_lower_500.json').read_text())
    s['upper_alpha']=int(alpha);s['lower_reference_centered']=True
    s['lower_controller']={'alias':alias,'alpha':1.,'path_capacity':3201}
    s['scope']['one_actor']=True;s['scope']['alphas']=[float(alpha)]
    s['ppo'].update(default_updates=250,future_total_updates_only_after_user_approval=250,kl_group_index=None,alpha_slots='all_envs_fixed_upper_alpha')
    s['budget'].update(default_policy_transitions=512*128*250,default_control_transitions_upper=512*128*250*4,additional_compute_wall_seconds=16800,compile_wall_seconds=900,engineering_wall_seconds=600,training_wall_seconds=14400,evaluation_wall_seconds=1200,full_episode_wall_seconds=480)
    return s

class EndpointCampaign(Campaign):
    def __init__(self,config,output):
        self.config=Path(config);self.spec=load_endpoint_spec(config);self.out=Path(output)
        if (self.out/'manifest.json').exists():raise FileExistsError('immutable run identity exists')
        self.budget=ComputeBudget(output,self.spec);self.budget.recover_interrupted()
        self.status=dict(run_id=self.out.name,state='initializing',stage='interface',completed_updates=0,declared_updates=250,policy_transitions=0,control_ticks=0,initialization='fresh_independent_upper',training_seed=self.spec['ppo']['seed'],upper_alpha=self.spec['upper_alpha'],lower_alpha=1.,lower_alias=self.spec['lower_controller']['alias'])
        self.seen_cases=set();self._status()
    def initialize(self):
        super().initialize()
        import jax,jax.numpy as jp,numpy as np
        from .closed_loop_kernel import preview_controls
        e=self.env;s=e.reset(self.sample,jp.int32(0),jp.int32(0),jp.asarray(float(self.spec['upper_alpha'])))
        # Nonzero upper command must enter the complete frozen lower chain unchanged.
        def probe(s):
            end,log=e._tick(s,jp.array([.4,-.3]),False)
            m,_,_=e.physics.observe(s.physical.data);gov=log['governed']
            expected,_,goal=preview_controls(s.physical.controller,m,gov,gov,s.physical.physical_tick*e.cc.dt>3.,e.cc)
            return end,log,expected,goal
        fn=self.compile('nonzero upper / frozen lower control contract',probe,s)
        self.budget.reserve(1,'reference-centered contract tick')
        with self.budget.measure('smoke','nonzero upper contract tick'):
            end,log,expected,goal=fn(s);jax.block_until_ready(end)
            for a,b in zip(jax.tree.leaves(end.physical.controller),jax.tree.leaves(expected)):np.testing.assert_allclose(a,b,atol=1e-6)
            np.testing.assert_allclose(log['u_nom'],jp.array([goal.steer_rate,log['governed'][0]/.1]),atol=1e-6)
            np.testing.assert_allclose(log['requested_residual'],log['lower_action']*jp.array([1.5,10]),atol=1e-6)
            np.testing.assert_allclose(log['limited_command'],s.physical.raw,atol=1e-6)
            assert not np.allclose(log['governed'],log['limited_command'])
            write(self.out/'lower_contract_check.json',dict(passed=True,upper_alpha=self.spec['upper_alpha'],lower_alpha=1.,eso_single_commit=True,base_reference='governed',raw_reference_reward_preserved=True,capacity=3201))

def run_endpoint(config,output):
    c=EndpointCampaign(config,output)
    try:
        c.initialize();c.preflight();c.train('smoke',2)
        _,completed=c.train('pilot',250)
        if completed!=250 or c.status.get('training_stop_reason'):raise RuntimeError('endpoint incomplete or optimizer stopped; retain last checkpoint, stop queue')
        c._status(state='complete',stage='training_complete',completed_updates=completed)
        notify('STTW upper endpoint complete',f"{c.status['lower_alias']} upper_alpha={c.status['upper_alpha']} {completed}/250")
    except BaseException as e:
        c._status(state='budget_stopped' if isinstance(e,BudgetStop) else 'error',reason=str(e));raise
