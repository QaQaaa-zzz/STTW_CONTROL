"""Chunked solved-physics-step contact buffers for frozen STTW diagnostics."""
import json,math,time
from pathlib import Path
import numpy as np
import torch


class PhysicsContactRecorder:
    def __init__(self, output, env, config):
        if not env.plant.track_contact_wrench or env.plant.servo_mode != 'explicit':
            raise ValueError('Contact wrench recording requires tracked wheels and explicit motor torque')
        self.output=Path(output);self.output.mkdir()
        self.copy_seconds=0.;self.compression_seconds=0.
        self.chunk_steps=int(config.get('chunk_steps',250))
        self.buffers=None;self.used=0;self.total=0;self.chunks=[];self.elapsed=0;self.last_recorded=0
        self.window=config.get('record_window_s')
        self.window_steps=None if self.window is None else (math.ceil(self.window[0]/env.physics_dt-1e-9),math.floor(self.window[1]/env.physics_dt+1e-9))
        self.metadata=dict(physics_dt_s=env.physics_dt,body_order=env.plant.bodies,joint_order=env.plant.joints,
            origins=env.plant.origin.cpu().tolist(),sampling='post-solve every physics step, world coordinates, body quaternion xyzw',
            force_units='N (API dt=physics_dt); points m; motor torque Nm in generalized joint coordinates',
            indexing='normal_count/start and friction_count/start independently index each wheel flat patch buffer; only counted entries valid',
            phase='alive_before_this_step retained; same-step motor_torque is applied explicit actuator contribution, not passive torque',
            motor_limits=env.plant.servos.force.cpu().tolist() if hasattr(env.plant.servos,'force') else None,record_window_s=self.window)
    def record(self, env, lower_tick, physics_tick):
        self.elapsed+=1
        if self.window_steps and not self.window_steps[0]<=self.elapsed<=self.window_steps[1]:return
        self.last_recorded=self.elapsed
        values=env.plant.read_wheel_contact_samples()
        values.update(alive_before=env.alive.clone(),alpha=env.alpha.clone())
        if self.buffers is None:
            self.buffers={k:torch.empty((self.chunk_steps,)+v.shape,device=v.device,dtype=v.dtype) for k,v in values.items()}
        for key,value in values.items():self.buffers[key][self.used].copy_(value)
        self.used+=1;self.total+=1
        if self.used==self.chunk_steps:self.flush()
    def flush(self):
        if not self.used:return
        name=f'chunk_{len(self.chunks):05d}.npz'
        started=time.perf_counter()
        arrays={k:v[:self.used].cpu().numpy() for k,v in self.buffers.items()}
        self.copy_seconds+=time.perf_counter()-started
        arrays['physics_step']=np.arange(self.last_recorded-self.used+1,self.last_recorded+1,dtype=np.int64)
        started=time.perf_counter()
        np.savez_compressed(self.output/name,**arrays)
        self.compression_seconds+=time.perf_counter()-started
        self.chunks.append(dict(file=name,first_step=self.last_recorded-self.used+1,last_step=self.last_recorded));self.used=0
    def close(self):
        self.flush();self.metadata.update(total_physics_steps=self.elapsed,recorded_physics_steps=self.total,chunks=self.chunks,copy_and_sync_s=self.copy_seconds,compression_write_s=self.compression_seconds)
        (self.output/'index.json').write_text(json.dumps(self.metadata,indent=2)+'\n')
