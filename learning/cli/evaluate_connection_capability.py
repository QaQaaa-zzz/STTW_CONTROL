"""Evaluate executed paired connection outcomes, holding out whole origin runs."""
import argparse,json
from pathlib import Path
from sttw_control.empirical_capability import evaluate_connection_groups
p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());rows=[json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
result=evaluate_connection_groups(rows,c['candidates'],c['feature_scales'],c['bandwidth'],c['minimum_support'])
result['inputs']=dict(records=str(a.records.resolve()),config=str(a.config.resolve()))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
