"""Send one explicit STTW job notification through the existing private Bark config."""
import argparse,datetime,json
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener, getproxies, urlopen
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser();p.add_argument('--message',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
c=json.loads(Path('/home/qy/.config/jit-email/bark.json').read_text());now=datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
payload=json.dumps(dict(device_key=c['device_key'],title='STTW 连续石子路面',body=now+'\n'+a.message,group='STTW 石子路面',level='active'),ensure_ascii=False).encode()
request = Request(c['server'].rstrip('/')+'/push', data=payload,
                  headers={'Content-Type':'application/json; charset=utf-8'}, method='POST')
try:
    response = urlopen(request, timeout=15)
except URLError as error:
    # A TLS handshake timeout precedes the POST. Use the direct route once.
    if not (getproxies().get('https') and isinstance(error.reason, TimeoutError)
            and 'handshake' in str(error.reason).lower()):
        raise
    response = build_opener(ProxyHandler({})).open(request, timeout=15)
with response:
    result = json.loads(response.read())
if result.get('code')!=200:raise RuntimeError('Bark notification failed')
a.receipt.write_text(json.dumps(dict(status='sent',time=now,message=a.message),ensure_ascii=False,indent=2)+'\n')
