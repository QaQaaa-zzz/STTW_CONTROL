"""Execute the patched VecEnv.step on scripted solver readings, without Isaac."""
import ast
import json
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
import torch
from sttw_control import phase_u_events
from sttw_control.ditch_jump_adapter import DitchJumpAdapter
from sttw_control.ditch_terrain import build_ditch_terrain

LEARN = Path(__file__).resolve().parents[1]

def native_step():
    # Only the optional Isaac imports/constructor require a GPU. Compile the
    # original method text and supply the same event routines used in training.
    path = LEARN/'src/sttw_control/phase_u_env.py'
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n,ast.FunctionDef) and n.name=='step')
    g = {**vars(phase_u_events), 'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),g)
    return g['step']


class NativeTransitionTests(unittest.TestCase):
    def test_success_crash_timeout_three_native_wirings(self):
        torch.set_num_threads(1)
        for case in ('success','crash','timeout','strict_success','focused_success','focused_crash','focused_timeout'):
            with self.subTest(case=case):
                self.check_episode(case)

    def check_episode(self,case):
        one=torch.arange(1)
        cfg=json.loads((LEARN/'configs/isaac_rb_jump_ditch_goal150.json').read_text())
        cfg['ditch_landing_focus']=case.startswith('focused_')
        case=case.removeprefix('focused_')
        cfg['plant_options']['wheel_patch_capacity']=64
        cfg['verified_ditch_support_mode']='strict' if case=='strict_success' else 'rough'
        case=case.removeprefix('strict_')
        spec=cfg['ditch_terrain']
        task=json.loads((LEARN/'configs/isaac_phase_u_task_ditch_goal.json').read_text())['task']
        max_steps=110 if case=='success' else 10
        task['ppo']['episode_horizon']=max_steps
        e=NS(cfg=cfg,dt=.02,num_envs=1,num_obs=76,device='cpu',task=task,ditch=spec,
             physics_substeps=8,sensor_substeps=2,rand={'push_interval_s':4},max_episode_length=max_steps)
        q=lambda *shape:torch.zeros(shape)
        e.episode_length_buf=one.clone()
        e.episode_return=q(1);e.last_action=q(1,4);e.wheel_ids=[0,1]
        e.landing_reward=True;e.platform_held=torch.tensor([False]);e.obs_buf=q(1,76)
        e.events=phase_u_events.initial_events(torch.tensor([1.5]),task)
        e.total_physics_steps=0;e.targets=lambda v:v
        p=NS(physics_dt=.0025,wheel_radius_m=.07,roots=q(1,13),origin=q(1,3),nonwheel=[2],
             rigid=q(1,3,13),dofs=q(1,2,2),contact=q(1,3,3),pendulum_servos=NS(indices=[0,1]),
             last_pendulum_torque=q(1,2),bodies=['frontwheel','rearwheel','chasis'])
        p.rigid[0,:2,3]=.5**.5
        p.rigid[0,:2,6]=.5**.5
        p.robot=NS(data=NS(root_link_lin_vel_w=q(1,3),body_com_acc_w=q(1,3,3)))
        p.refresh=lambda:None
        full=lambda *v:q(1,106)
        e.physical=lambda:(p.roots,(p.contact[:,2].abs().sum(-1)>0),full())
        e.geometry=NS(frontmost_x=lambda x:p.rigid[:,0,0])
        e.history=NS(privileged=full,make_frame=lambda *v:q(1,25),advance_frame=lambda *v:q(1,76))
        e.randomization=NS(command=lambda x:x,sensor_frame=lambda x:x)
        tick=0
        def sample_solver(*args,refresh_diagnostics,**kw):
            nonlocal tick
            assert refresh_diagnostics
            t=tick*.0025-.12
            f=3.+1.6*t
            flying=0.<=t<.75
            p.rigid[0,:2,0]=torch.tensor([f,f-.408])
            p.rigid[0,:2,2]=.21 if flying else .07
            p.roots[0,:3]=torch.tensor([f-.204,0.,.20])
            p.roots[0,7]=1.6
            p.contact[:]=0
            if not flying:p.contact[0,:2,2]=12.
            # One solved substep only: original 20ms endpoint reads no collision.
            if case=='crash' and tick==27:p.contact[0,2,2]=10.
            tick+=1
        p.step=sample_solver
        def filter_data(i):
            def read(dt):
                assert dt==p.physics_dt
                pt=p.rigid[0,i,:3].clone();pt[2]=0
                return (p.contact[:,i,2:3],pt[None,:],torch.tensor([[0.,0.,1.]]),
                        None,(p.contact[:,i,2:3]>0).long(),torch.tensor([[0]]))
            return NS(contact_physx_view=NS(get_contact_data=read))
        p.wheel_sensors=[filter_data(0),filter_data(1)]
        e.plant=p
        e.verified_jump=DitchJumpAdapter(e,build_ditch_terrain(spec)[0])
        e.reset_idx=lambda ids:e.verified_jump.reset(ids)
        run_step=native_step()
        total=0.
        for _ in range(max_steps):
            _,_,rew,done,info=run_step(e,torch.zeros(1,4))
            total+=float(info['unscaled_reward'][0])
            self.assertAlmostEqual(float(rew[0]),float(info['unscaled_reward'][0])*.1,places=5)
            self.assertTrue(torch.isfinite(rew).all())
            if bool(done):break
        self.assertTrue(bool(done))
        self.assertEqual(float(info['time_outs'][0]),0.)
        self.assertEqual(bool(info['jump_success'][0]),case=='success')
        self.assertEqual(int(info['end_code'][0]),dict(success=13,crash=12,timeout=9)[case])
        self.assertAlmostEqual(total,dict(success=110.,crash=-100.,timeout=-95.)[case],places=4)


if __name__=='__main__':
    unittest.main()
