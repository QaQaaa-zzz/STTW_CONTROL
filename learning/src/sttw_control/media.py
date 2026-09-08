"""Render recorded states, never integrate physics to manufacture a video."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import mujoco
from .model import load_model
from .path import CircleConfig,circle_reference,tracking_metrics


def frame_indices(times,fps):
    times=np.asarray(times)
    if times.ndim!=1 or len(times)<1 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('video requires finite strictly increasing timestamps')
    if not math.isfinite(fps) or fps<=0: raise ValueError('fps must be positive finite')
    count=int(math.ceil((times[-1]-times[0])*fps-1e-8))+1
    samples=np.minimum(times[0]+np.arange(count)/fps,times[-1])
    samples[-1]=times[-1]
    upper=np.minimum(np.searchsorted(times,samples),len(times)-1)
    lower=np.maximum(upper-1,0)
    ids=np.where(abs(times[lower]-samples)<abs(times[upper]-samples),lower,upper)
    ids[-1]=len(times)-1
    return ids


def validate_trace(trace):
    for key in ('time','qpos','qvel','measurement','command'):
        if key not in trace: raise ValueError('missing trace field '+key)
    frame_indices(trace['time'],30)
    n=len(trace['time'])
    for key in ('qpos','qvel','measurement','command'):
        array=np.asarray(trace[key])
        if array.ndim!=2 or len(array)!=n or not np.isfinite(array).all():
            raise ValueError('invalid trace shape/values: '+key)
    if trace['measurement'].shape[1]!=7 or trace['command'].shape[1]!=2:
        raise ValueError('unexpected measurement or command schema')


def state_series(trace,config):
    """Derive only observable geometry/velocity from recorded states."""
    w,x,y,z=trace['qpos'][:,3:7].T
    forward=np.column_stack([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)])
    speed=np.sum(trace['qvel'][:,:3]*forward,axis=1)
    yaw=np.arctan2(forward[:,1],forward[:,0])
    reference=trace.get('motion_command')
    if reference is None:
        if config.get('circle') is not None:
            raise ValueError('circle media requires captured motion command')
        steer=config['steer_reference']+config['steer_amplitude']*np.sin(2*np.pi*config['steer_frequency']*trace['time'])
        reference=np.column_stack([steer,np.full(len(steer),config['speed_reference'])])
    roll_reference=trace.get('reference_roll')
    if roll_reference is None:
        c=config['controller'];v=np.maximum(c['minimum_speed'],trace['measurement'][:,5]*.1)
        m2=-(c['mass']*v*v*c['cg_height']-c['mass']*c['cg_forward']*c['trail']*c['gravity'])*np.cos(c['caster'])/c['wheelbase']
        m2=np.where(m2<0,-1,1)*np.maximum(abs(m2),.1)
        roll_reference=reference[:,0]*m2/(-c['mass']*c['gravity']*c['cg_height'])
    return {'speed':speed,'yaw':yaw,'reference':reference,'roll_reference':roll_reference}


def plot_states(trace,config,output,controller_label='ECBC + ESO baseline'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
                         'savefig.dpi':180})
    blue,orange='#2265a0','#c07924'
    t=trace['time'];m=trace['measurement'];s=state_series(trace,config)
    event=np.asarray(trace['event'][0]) if 'event' in trace else np.array([config['disturbance_start']/config['controller']['dt'],(config['disturbance_start']+config['disturbance_duration'])/config['controller']['dt'],config.get('disturbance_steer_rate',0.),config['disturbance_force'],float(config.get('disturbance_waveform')=='half_sine')])
    event_start,event_end=event[:2]*config['controller']['dt']
    has_event=bool(event[2] or event[3])
    circle=CircleConfig(**config['circle']) if config.get('circle') else None
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    xy=trace['qpos'][:,:2]
    if circle:
        ref=circle_reference(circle)
        axes[0].plot(ref[:,0],ref[:,1],'--',color=orange,lw=2,label='Reference circle')
        error=np.linalg.norm(xy-np.array([circle.center_x,circle.center_y]),axis=1)-circle.radius
        metrics=tracking_metrics(xy,circle)
    elif config.get('steer_reference',0)==0 and config.get('steer_amplitude',0)==0:
        axes[0].plot([xy[0,0],xy[-1,0]],[xy[0,1],xy[0,1]],'--',color=orange,lw=2,label='Straight reference')
        error=xy[:,1]-xy[0,1]
        metrics={'lateral_rmse_m':float(np.sqrt(np.mean(error**2))),'lateral_max_abs_m':float(abs(error).max())}
    else:
        # A time-varying steering instruction is not an XY reference path.
        error=np.rad2deg(m[:,2]-s['reference'][:,0])
        metrics={'steering_error_rmse_deg':float(np.sqrt(np.mean(error**2)))}
    axes[0].plot(xy[:,0],xy[:,1],color=blue,lw=1.8,label='Recorded chassis path')
    axes[0].scatter(*xy[0],marker='o',facecolors='white',edgecolors=blue,zorder=3,label='Start')
    axes[0].scatter(*xy[-1],marker='x',color=blue,zorder=3,label='End')
    axes[0].set(xlabel='World X [m]',ylabel='World Y [m]',title='Reference and recorded path',aspect='equal')
    axes[0].legend(fontsize=9)
    axes[1].plot(t,error,color=blue,lw=1.4)
    if has_event:
        axes[1].axvspan(event_start,event_end,color='red',alpha=.15,label=f'Disturbance {event_start:.2f}-{event_end:.2f}s')
        axes[1].legend(fontsize=8)
    axes[1].axhline(0,color='#555555',lw=.8,ls='--')
    is_path=circle is not None or (config.get('steer_reference',0)==0 and config.get('steer_amplitude',0)==0)
    axes[1].set(xlabel='Simulation time [s]',ylabel=('Radial error [m], outward +' if circle else 'Lateral error [m]') if is_path else 'Steering error [deg]',title='Path tracking error' if is_path else 'Steering reference error')
    fig.suptitle(f'{controller_label} | {t[-1]-t[0]:.1f} s',fontsize=14)
    fig.savefig(output/'trajectory.png');fig.savefig(output/'trajectory.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
    panels=[(m[:,0]*180/np.pi,s['roll_reference']*180/np.pi,'Roll [deg]','Roll and equilibrium reference'),
            (m[:,1]*180/np.pi,None,'Roll rate [deg/s]','Roll angular velocity'),
            (s['speed'],s['reference'][:,1],'Speed [m/s]','Longitudinal speed'),
            (m[:,2]*180/np.pi,s['reference'][:,0]*180/np.pi,'Steering [deg]','Steering angle'),
            (m[:,3],trace['command'][:,0],'Steering rate [rad/s]','Steering rate and command'),
            (m[:,5],trace['command'][:,1],'Rear wheel rate [rad/s]','Rear drive and command')]
    for ax,(actual,ref,unit,title) in zip(axes.flat,panels):
        ax.plot(t,actual,color=blue,lw=1.35,label='Measured' if 'command' in title else 'Actual')
        if ref is not None: ax.plot(t,ref,color=orange,ls='--',lw=1.3,label='Command' if 'command' in title else 'Reference')
        ax.set(ylabel=unit,title=title)
        if has_event:ax.axvspan(event_start,event_end,color='red',alpha=.15)
        if ref is not None: ax.legend(loc='best',fontsize=9)
    for ax in axes[-1]: ax.set_xlabel('Simulation time [s]')
    fig.suptitle('Recorded vehicle states and control targets | '+controller_label,fontsize=14)
    fig.savefig(output/'states.png');fig.savefig(output/'states.pdf');plt.close(fig)
    if has_event:
        phase=np.clip((t-event_start)/(event_end-event_start),0,1)
        factor=np.where((t>=event_start)&(t<event_end),np.sin(np.pi*phase) if event[4] else 1.,0.)
        steer=trace.get('injected_steer_rate',event[2]*factor)
        force=np.linalg.norm(trace['applied_wrench'][:,:3],axis=1)*np.sign(event[3]) if 'applied_wrench' in trace else event[3]*factor
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
        for ax,y,label in zip(axes,[steer,force],['Injected steer-rate pulse [rad/s]','Applied lateral force [N]']):
            ax.plot(t,y);ax.axvspan(event_start,event_end,color='red',alpha=.15);ax.set(ylabel=label)
        axes[-1].set_xlabel('Simulation time [s]')
        fig.suptitle(f'Disturbance {event_start:.2f}-{event_end:.2f}s | duration {event_end-event_start:.2f}s')
        fig.savefig(output/'disturbance.png');fig.savefig(output/'disturbance.pdf');plt.close(fig)
    return metrics


def _lines(scene,points,color,width):
    for a,b in zip(points[:-1],points[1:]):
        if scene.ngeom>=scene.maxgeom: raise ValueError('render scene geom capacity exceeded')
        if np.linalg.norm(a-b)<1e-8: continue
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.asarray(color,dtype=np.float32))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64))
        scene.ngeom+=1


def render_run(run_path,*,fps=30):
    """Write media into a new directory; all frames map to real trace indices."""
    import mediapy
    from PIL import Image,ImageDraw,ImageFont
    run=Path(run_path)
    declaration=json.loads((run/'declaration.json').read_text())
    trace_path=run/'trace.npz'
    with np.load(trace_path,allow_pickle=False) as archive:
        trace={key:archive[key] for key in archive.files}
    validate_trace(trace)
    bundle=load_model()
    if bundle.identity!=declaration['model']:
        raise ValueError('recorded model differs from available render model')
    if trace['qpos'].shape[1]!=bundle.model.nq or trace['qvel'].shape[1]!=bundle.model.nv:
        raise ValueError('trace model dimensions mismatch')
    output=run/'media'
    output.mkdir(exist_ok=False)
    try:
        controller_label='ECBC + ESO baseline' if declaration['controller']=='baseline' else 'ECBC + ESO + residual policy'
        metrics=plot_states(trace,declaration['config'],output,controller_label)
        config=declaration['config']
        series=state_series(trace,config)
        circle=CircleConfig(**config['circle']) if config.get('circle') else None
        xy=trace['qpos'][:,:2]
        straight=config.get('steer_reference',0)==0 and config.get('steer_amplitude',0)==0
        ref=circle_reference(circle,181) if circle else (np.column_stack([np.linspace(xy[0,0],xy[-1,0],100),np.full(100,xy[0,1])]) if straight else np.empty((0,2)))
        ref3=np.column_stack([ref,np.full(len(ref),.012)])
        model=bundle.model
        model.vis.global_.offwidth=640;model.vis.global_.offheight=640
        data=mujoco.MjData(model)
        options=mujoco.MjvOption();options.geomgroup[0]=0
        follow=mujoco.MjvCamera();mujoco.mjv_defaultCamera(follow)
        top=mujoco.MjvCamera();mujoco.mjv_defaultCamera(top)
        follow.distance=1.65;follow.elevation=-18
        top.elevation=-89.5;top.azimuth=90
        if circle:
            top.lookat[:]=[circle.center_x,circle.center_y,0.]
            top.distance=2.9*circle.radius
        else:
            top.lookat[:]=[(xy[:,0].min()+xy[:,0].max())/2,(xy[:,1].min()+xy[:,1].max())/2,0]
            top.distance=max(3.,1.65*np.ptp(xy,axis=0).max())
        from matplotlib.font_manager import findfont,FontProperties
        font_path=findfont(FontProperties(family='DejaVu Sans'))
        font=ImageFont.truetype(font_path,20)
        small=ImageFont.truetype(font_path,17)
        event_path=run/'event.json'
        event=json.loads(event_path.read_text()) if event_path.exists() else {'start_seconds':config['disturbance_start'],'end_seconds':config['disturbance_start']+config['disturbance_duration'],'steer_rate_peak':config.get('disturbance_steer_rate',0.),'force_peak':config['disturbance_force'],'waveform':config.get('disturbance_waveform','constant')}
        ids=frame_indices(trace['time'],fps)
        with mujoco.Renderer(model,640,640,max_geom=3000) as renderer, mediapy.VideoWriter(output/'replay.mp4',(720,1280),fps=fps,crf=20,ffmpeg_args=['-movflags','+faststart']) as writer:
            for frame_number,index in enumerate(ids):
                data.qpos[:]=trace['qpos'][index];data.qvel[:]=trace['qvel'][index];data.time=float(trace['time'][index])
                mujoco.mj_forward(model,data)
                follow.lookat[:]=data.xpos[bundle.chassis]
                follow.azimuth=float(np.rad2deg(series['yaw'][index]))+135
                panels=[]
                for camera in (follow,top):
                    renderer.update_scene(data,camera=camera,scene_option=options)
                    _lines(renderer.scene,ref3,[.78,.45,.12,1],.012)
                    sampled=np.unique(np.linspace(0,index,min(index+1,260),dtype=int))
                    actual=np.column_stack([xy[sampled],np.full(len(sampled),.027)])
                    _lines(renderer.scene,actual,[.08,.39,.7,1],.013)
                    panels.append(renderer.render().copy())
                canvas=Image.new('RGB',(1280,720),'#f8fafc')
                canvas.paste(Image.fromarray(panels[0]),(0,40));canvas.paste(Image.fromarray(panels[1]),(640,40))
                draw=ImageDraw.Draw(canvas)
                draw.text((16,8),controller_label+' | recorded replay',font=font,fill='#243442')
                draw.text((655,8),'Overhead: orange reference / blue actual' if len(ref) else 'Overhead: blue actual (no XY reference)',font=font,fill='#243442')
                end=' | END' if index==len(xy)-1 else ''
                draw.text((16,688),f't = {data.time:5.2f} s   speed = {series["speed"][index]:.2f} m/s   roll = {np.rad2deg(trace["measurement"][index,0]):+.2f} deg{end}',font=small,fill='#243442')
                if event['force_peak'] or event['steer_rate_peak']:
                    active=event['start_seconds']<=data.time<event['end_seconds']
                    phase=np.clip((data.time-event['start_seconds'])/(event['end_seconds']-event['start_seconds']),0.,1.)
                    factor=(np.sin(np.pi*phase) if event['waveform']=='half_sine' else 1.) if active else 0.
                    text=f"{'DISTURBANCE ON' if active else 'Disturbance OFF'} | {event['start_seconds']:.2f}-{event['end_seconds']:.2f}s | steer {event['steer_rate_peak']*factor:+.2f} rad/s | force {event['force_peak']*factor:+.2f} N"
                    draw.rectangle((12,48,1268,80),fill='#a12d21' if active else '#243442')
                    draw.text((20,53),text,font=small,fill='white')
                frame=np.asarray(canvas)
                writer.add_image(frame)
                if frame_number==len(ids)//2: canvas.save(output/'preview.png')
                if (event['force_peak'] or event['steer_rate_peak']) and abs(data.time-(event['start_seconds']+event['end_seconds'])/2)<=1/fps:
                    canvas.save(output/'disturbance_active.png')
                if frame_number==len(ids)-1: canvas.save(output/'terminal.png')
        manifest={'fps':fps,'frame_count':len(ids),'first_trace_index':int(ids[0]),'last_trace_index':int(ids[-1]),
                  'source_trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  'source_declaration_sha256':hashlib.sha256((run/'declaration.json').read_bytes()).hexdigest(),
                  'simulation_duration_s':float(trace['time'][-1]-trace['time'][0]),
                  'video_duration_s':len(ids)/fps,'method':'qpos_qvel_mj_forward_only_no_physics_step',
                  'frame_indices':ids.tolist(),'tracking_metrics':metrics,'status':'complete'}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        return manifest
    except Exception as exc:
        (output/'manifest.json').write_text(json.dumps({'status':'error','error':str(exc)})+'\n')
        raise


def compare_runs(baseline,candidate,output,*,candidate_label='Learned residual'):
    """Compare immutable same-contract recorded runs, without loading a policy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline,candidate,output=map(Path,(baseline,candidate,output))
    declarations=[json.loads((p/'declaration.json').read_text()) for p in (baseline,candidate)]
    for field in ('config','model','seed','backend','reset_semantics'):
        if declarations[0][field]!=declarations[1][field]:
            raise ValueError('comparison mismatch: '+field)
    traces=[dict(np.load(p/'trace.npz')) for p in (baseline,candidate)]
    for trace in traces: validate_trace(trace)
    for field in ('qpos','qvel'):
        if not np.array_equal(traces[0][field][0],traces[1][field][0]):
            raise ValueError('comparison initial state mismatch')
    c=declarations[0]['config'];circle=CircleConfig(**c['circle'])
    summaries=[json.loads((p/'summary.json').read_text()) for p in (baseline,candidate)]
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    ref=circle_reference(circle)
    axes[0,0].plot(ref[:,0],ref[:,1],'k:',label='Reference')
    for trace,label,color,ls,summary in zip(traces,['ECBC + ESO',candidate_label],['#24567a','#b45f24'],['-','--'],summaries):
        t=trace['time'];s=state_series(trace,c)
        xy=trace['qpos'][:,:2]
        radial=np.linalg.norm(xy-np.array([circle.center_x,circle.center_y]),axis=1)-circle.radius
        rmse=summary['circle_tracking']['radial_rmse_m']
        axes[0,0].plot(*xy.T,color=color,ls=ls,label=f'{"Baseline" if label=="ECBC + ESO" else "Residual"}: RMSE {rmse:.4f} m')
        for ax,y in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2]],
                        [radial,np.rad2deg(trace['measurement'][:,0]),s['speed'],trace['command'][:,0],trace['command'][:,1]]):
            ax.plot(t,y,color=color,ls=ls,label=label)
    axes[0,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Reference and recorded trajectories',aspect='equal')
    for ax,title,ylabel in zip(axes.flat[1:],['Signed radial error','Roll angle (left positive)','True longitudinal speed','Applied steering-rate command','Applied rear-wheel-rate command'],['m','deg','m/s','rad/s','rad/s']):
        ax.set(title=title,xlabel='Time (s)',ylabel=ylabel)
    axes[0,1].axhline(0,color='black',lw=.7)
    axes[1,0].axhline(c['speed_reference'],color='black',ls=':',label='Reference')
    for ax in axes.flat: ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,loc='center')
    fig.suptitle(f"Recorded {declarations[0]['backend'].upper()} comparison | seed {declarations[0]['seed']} | {candidate_label}\nSame configuration and initial state; paired engineering evaluation",fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(output/f'comparison.{suffix}',dpi=170)
    plt.close(fig)
    manifest={'baseline':str(baseline),'candidate':str(candidate),'candidate_label':candidate_label,
              'trace_sha256':[hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest() for p in (baseline,candidate)],'summaries':summaries}
    (output/'comparison.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
