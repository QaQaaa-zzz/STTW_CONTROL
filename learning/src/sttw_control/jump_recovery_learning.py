"""Torch port of RecoveryBridge short-sequence teaching and source retention.

Reference (read-only): DVGC/JIT generative_bridge/network.py and student.py.
The student architecture and frozen normalizers stay those of Phase U. This
port uses fixed disturbance proposals initially; it is not the full learned
explorer campaign reported in the RecoveryBridge paper.
"""
import inspect
import textwrap
import types
import numpy as np
import torch
from torch import nn


class ConditionalBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.conv0=nn.Conv1d(cin,cout,3,padding=1);self.norm0=nn.GroupNorm(8,cout,eps=1e-5)
        self.film=nn.Linear(512,2*cout)
        self.conv1=nn.Conv1d(cout,cout,3,padding=1);self.norm1=nn.GroupNorm(8,cout,eps=1e-5)
        self.residual=nn.Identity() if cin==cout else nn.Conv1d(cin,cout,1)
    def forward(self,x,c):
        scale,shift=self.film(nn.functional.silu(c)).chunk(2,-1)
        h=nn.functional.silu(self.norm0(self.conv0(x))*(1+scale[:,:,None])+shift[:,:,None])
        return self.residual(x)+nn.functional.silu(self.norm1(self.conv1(h)))


class JumpSequenceTeacher(nn.Module):
    """H16 conditional diffusion U-Net; only observation width changes to 80."""
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',mean.detach().clone());self.register_buffer('std',std.detach().clone())
        self.register_buffer('frequency',torch.exp(-np.log(10000.)*torch.arange(64)/63.))
        t=torch.arange(101,dtype=torch.float64)/100
        f=torch.cos((t+.008)/1.008*torch.pi/2).square()
        self.register_buffer('alpha_bar',torch.cumprod(1-(1-f[1:]/f[:-1]).clamp(max=.999),0).float())
        self.obs=nn.Sequential(nn.Linear(len(mean),128),nn.SiLU(),nn.Linear(128,256))
        self.time=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,256))
        self.stem=nn.Conv1d(4,64,3,padding=1)
        self.enc0=nn.ModuleList([ConditionalBlock(64,64) for _ in range(2)])
        self.down0=nn.Conv1d(64,128,4,stride=2,padding=1)
        self.enc1=nn.ModuleList([ConditionalBlock(128,128) for _ in range(2)])
        self.down1=nn.Conv1d(128,256,4,stride=2,padding=1)
        self.middle=nn.ModuleList([ConditionalBlock(256,256) for _ in range(2)])
        self.up1=nn.Conv1d(256,128,3,padding=1)
        self.dec1=nn.ModuleList([ConditionalBlock(256,128),ConditionalBlock(128,128)])
        self.up0=nn.Conv1d(128,64,3,padding=1)
        self.dec0=nn.ModuleList([ConditionalBlock(128,64),ConditionalBlock(64,64)])
        self.out_norm=nn.GroupNorm(8,64,eps=1e-5);self.out=nn.Conv1d(64,4,1)
    def forward(self,actions,observations,step):
        angle=step[:,None]*self.frequency[None,:]
        c=torch.cat((self.obs((observations-self.mean)/self.std),self.time(torch.cat((angle.sin(),angle.cos()),-1))),-1)
        x=self.stem(actions.transpose(1,2))
        for block in self.enc0:x=block(x,c)
        skip0=x;x=self.down0(x)
        for block in self.enc1:x=block(x,c)
        skip1=x;x=self.down1(x)
        for block in self.middle:x=block(x,c)
        x=torch.cat((self.up1(x.repeat_interleave(2,-1)),skip1),1)
        for block in self.dec1:x=block(x,c)
        x=torch.cat((self.up0(x.repeat_interleave(2,-1)),skip0),1)
        for block in self.dec0:x=block(x,c)
        return self.out(nn.functional.silu(self.out_norm(x))).transpose(1,2)
    @torch.no_grad()
    def sample(self,observations):
        x=torch.randn(len(observations),16,4,device=observations.device)
        schedule=torch.arange(99,-1,-5,device=x.device)
        for i,t in enumerate(schedule):
            noise=self(x,observations,t.expand(len(x)))
            abar=self.alpha_bar[t]
            clean=((x-(1-abar).sqrt()*noise)/abar.sqrt()).clamp(-1,1)
            previous=self.alpha_bar[schedule[i+1]] if i+1<len(schedule) else torch.ones((),device=x.device)
            x=previous.sqrt()*clean+(1-previous).sqrt()*noise
        return x.clamp(-1,1)


def fit_teacher(model,observations,sequences,updates,batch_size=256):
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    model.train();history=[]
    for iteration in range(updates):
        ids=torch.randint(len(observations),(batch_size,),device=observations.device)
        clean=sequences[ids];step=torch.randint(100,(batch_size,),device=clean.device)
        noise=torch.randn_like(clean);a=model.alpha_bar[step,None,None]
        loss=(model(a.sqrt()*clean+(1-a).sqrt()*noise,observations[ids],step)-noise).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if iteration%100==0 or iteration+1==updates:history.append(dict(update=iteration+1,noise_mse=float(loss.detach())))
    model.eval();return history


def teaching_arrays(trace,successful):
    """Only actual jump-control steps of successful full continuations."""
    obs=[];act=[];weights=[];roots=[];seqobs=[];seqact=[]
    for lane in np.flatnonzero(successful):
        mask=trace['valid'][:,lane]&trace['jump_active'][:,lane]
        ids=np.flatnonzero(mask)
        if not len(ids):continue
        obs.extend(trace['observation'][ids,lane]);act.extend(trace['action'][ids,lane])
        weights.extend([1/len(ids)]*len(ids));roots.extend([int(lane)]*len(ids))
        for t in ids:
            if t+16<=len(mask) and mask[t:t+16].all():
                seqobs.append(trace['observation'][t,lane]);seqact.append(trace['action'][t:t+16,lane])
    return dict(observation=np.asarray(obs,np.float32),action=np.asarray(act,np.float32),weight=np.asarray(weights,np.float32),root=np.asarray(roots),sequence_observation=np.asarray(seqobs,np.float32),sequence_action=np.asarray(seqact,np.float32))


def install_recovery_loss(algorithm,demo,keep,contract):
    """Reuse upstream Go2 PPO minibatches; add original demo/keep MSE jointly.

    No old transitions are inserted into the on-policy storage. Supervised
    arrays remain on GPU; there is no per-minibatch CPU sampling or copying.
    """
    policy=algorithm.actor_critic;device=next(policy.parameters()).device
    def prepare(data):
        o=torch.as_tensor(data['observation'],device=device);a=torch.as_tensor(data['action'],device=device)
        w=torch.as_tensor(data['weight'],device=device);return o,a,w/w.sum()
    demo=prepare(demo);keep=prepare(keep)
    algorithm.recovery_metrics={}
    def auxiliary(self):
        total=torch.zeros((),device=device)
        for name,data in (('demo',demo),('retention',keep)):
            o,a,w=data;ids=torch.multinomial(w,contract[name+'_batch_size'],replacement=True)
            pred=torch.tanh(policy.act_inference(o[ids]));loss=(pred-a[ids]).square().mean()
            self.recovery_metrics[name+'_mse']=loss.detach()
            total=total+contract[name+'_coefficient']*loss
        return total
    algorithm.recovery_auxiliary_loss=types.MethodType(auxiliary,algorithm)
    source=textwrap.dedent(inspect.getsource(type(algorithm).update))
    old='loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.sym_coef * sym_loss'
    new=f"loss = {contract['ppo_coefficient']!r} * (surrogate_loss - self.entropy_coef * entropy_batch.mean()) + self.value_loss_coef * value_loss + self.sym_coef * sym_loss + self.recovery_auxiliary_loss()"
    if source.count(old)!=1:raise ValueError('Go2 PPO loss changed; joint student adapter requires review')
    namespace=dict(type(algorithm).update.__globals__)
    exec(compile(source.replace(old,new),'<Go2 PPO with RecoveryBridge auxiliary losses>','exec'),namespace)
    algorithm.update=types.MethodType(namespace['update'],algorithm)
