"""Run a frozen supervised model matrix; caller holds the shared GPU lock and timeout."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();r=a.root
m=json.loads((r/'manifest.json').read_text());start=time.time();done=[]
for e in m['entries']:
    name=f"{e['kind']}_seed{e['seed']}"
    (r/'status.json').write_text(json.dumps(dict(stage='running',model=name,completed=done,total=len(m['entries']),elapsed_s=time.time()-start),indent=2))
    command=[sys.executable,'learning/cli/train_playground_history_model.py','--evaluation',m['evaluation'],'--nominal',m['nominal'],'--output',e['output'],'--phase',str(m.get('phase',2)),'--updates',str(m['updates_each']),'--seed',str(e['seed']),'--balanced-mixture','--terrain-features','--model-kind',e['kind'],'--additional-evaluations',*m.get('additional_evaluations',[])]
    with (r/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    subprocess.Popen([sys.executable,'learning/scripts/notify_chapter3_job.py','--title','第三章预测模型阶段结束','--message',name+'退出码'+str(result.returncode)+'；预测器拟合阶段，非整体任务完成。','--receipt',str(r/(name+'_bark.json'))])
    if result.returncode:
        (r/'status.json').write_text(json.dumps(dict(stage='failed',model=name,exit_code=result.returncode,completed=done,elapsed_s=time.time()-start),indent=2));sys.exit(result.returncode)
    done.append(name)
(r/'status.json').write_text(json.dumps(dict(stage='complete',completed=done,elapsed_s=time.time()-start),indent=2))
