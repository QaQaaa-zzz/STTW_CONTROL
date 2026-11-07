"""Send one explicit STTW job notification through the existing private Bark config."""
import argparse,datetime,json
from pathlib import Path
from urllib.request import Request,urlopen
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser();p.add_argument('--message',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
c=json.loads(Path('/home/qy/.config/jit-email/bark.json').read_text());now=datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
payload=json.dumps(dict(device_key=c['device_key'],title='STTW 连续石子路面',body=now+'\n'+a.message,group='STTW 石子路面',level='active'),ensure_ascii=False).encode()
with urlopen(Request(c['server'].rstrip('/')+'/push',data=payload,headers={'Content-Type':'application/json; charset=utf-8'},method='POST'),timeout=15) as response:result=json.loads(response.read())
if result.get('code')!=200:raise RuntimeError('Bark notification failed')
a.receipt.write_text(json.dumps(dict(status='sent',time=now,message=a.message),ensure_ascii=False,indent=2)+'\n')
