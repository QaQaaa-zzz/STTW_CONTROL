"""Export the Markdown manuscript, rendering local equations offline as SVG."""
import argparse,base64,io,re
from html import escape
from pathlib import Path
import markdown
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
from weasyprint import HTML
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();source=a.source.read_text()
def equation(m):
 expr=m[1];number=re.search(r'\\tag\{(\d+)\}',expr);expr=re.sub(r'\\tag\{\d+\}','',expr);b=io.BytesIO()
 math_to_image('$'+expr+'$',b,format='svg',dpi=180)
 return '\n<div class="equation"><img alt="equation '+number[1]+'" src="data:image/svg+xml;base64,'+base64.b64encode(b.getvalue()).decode()+'"><span>('+number[1]+')</span></div>\n'
source=re.sub(r'\$\$(.*?)\$\$',equation,source,flags=re.S)
source=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', lambda m: '\n<figure><img src="'+escape(m[2],quote=True)+'" alt="'+escape(m[1],quote=True)+'"><figcaption>'+escape(m[1])+'</figcaption></figure>\n', source)
body=markdown.markdown(source,extensions=['tables','fenced_code'])
css='''@page {size:A4;margin:18mm 18mm;@bottom-center{content:counter(page);font-size:9pt}}body{font-family:"Noto Serif CJK SC",serif;font-size:10pt;line-height:1.65;color:#202b35}h1{font-size:20pt}h2{font-size:14pt;margin-top:1.3em}h3{font-size:11pt}h1,h2,h3{break-after:avoid}p{orphans:3;widows:3}img{max-width:100%}figure{margin:14px 0;break-inside:avoid}figure img{display:block;max-height:190mm;margin:auto}figcaption{font-size:9pt;line-height:1.4;margin-top:7px}.equation{display:flex;align-items:center;justify-content:center;gap:8px;break-inside:avoid;margin:14px 0}.equation img{max-width:92%;max-height:65px}table{border-collapse:collapse;width:100%;font-size:8pt;line-height:1.5}th,td{padding:5px;border-bottom:1px solid #ccd4d8}th{background:#eef2f4}tr{break-inside:avoid}a{color:#28587b;text-decoration:none}code{font-size:8pt;overflow-wrap:anywhere}'''
html='<html lang="zh"><meta charset="utf-8"><title>STTW manuscript</title><style>'+css+'</style><body>'+body+'</body></html>'
a.source.with_suffix('.html').write_text(html);HTML(string=html,base_url=str(a.source.parent.resolve())).write_pdf(a.source.with_suffix('.pdf'))
print(a.source.with_suffix('.pdf'))
