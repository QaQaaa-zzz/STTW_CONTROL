#!/usr/bin/env python3
"""One bounded training run, then independent CPU standard panels and media."""
import argparse,json,os,subprocess,sys
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['task','training','panel','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if json.loads(a.task.read_text()).get('priority') is not None:
        raise ValueError('conditioned policies require train.py then explicit disturbance.py --priority-alpha panels; generic pipeline does not select an alpha')
    a.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        (a.output/'pipeline_status.json').write_text(json.dumps({'phase':phase,**extra},indent=2)+'\n')
    def invoke(args,log,cpu=False):
        env=os.environ.copy()
        if cpu:env['JAX_PLATFORMS']='cpu'
        with log.open('x') as stream:
            subprocess.run([sys.executable,*map(str,args)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    train=a.output/'training'
    try:
        status('training')
        invoke(['learning/cli/train.py','--task',a.task,'--config',a.training,'--output',train],a.output/'training.log')
        result=json.loads((train/'status.json').read_text())
        checkpoint=result['best_checkpoint'] or result['last_checkpoint']
        status('standard_evaluation',checkpoint=checkpoint,development_eligible=result['best_checkpoint'] is not None)
        panel=json.loads(a.panel.read_text())
        for seed in panel['evaluation_seeds']:
            out=a.output/'evaluation'/f'seed_{seed}'
            invoke(['learning/cli/disturbance.py','--task',a.task,'--panel',a.panel,'--training-run',train,'--checkpoint',checkpoint,'--output',out,'--seed',seed],a.output/f'evaluation_{seed}.log',True)
        status('media',checkpoint=checkpoint)
        from sttw_control.disturbance import plot_panel
        first=a.output/'evaluation'/f"seed_{panel['evaluation_seeds'][0]}"
        plot_panel(first)
        invoke(['learning/cli/panel_media.py','--panel-run',first,'--output',a.output/'complete_media'],a.output/'complete_media.log',True)
        rows=[]
        for seed in panel['evaluation_seeds']:
            for line in (a.output/'evaluation'/f'seed_{seed}'/'results.jsonl').read_text().splitlines():
                rows.append({'seed':seed,**json.loads(line)})
        (a.output/'standard_results.json').write_text(json.dumps(rows,indent=2)+'\n')
        status('complete',checkpoint=checkpoint,episodes=len(rows),development_eligible=result['best_checkpoint'] is not None)
    except Exception as exc:
        status('error',error=str(exc));raise
