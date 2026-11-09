"""A training exception must persist failure and bypass hanging simulator teardown."""
import ast
import json
import subprocess
import sys
from pathlib import Path


def test_training_exception_writes_status_and_exits_without_teardown(tmp_path):
    source=Path('learning/cli/train_isaac_phase_u.py').read_text()
    tree=ast.parse(source)
    hook=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training_exception')
    code='import os,sys,json,atexit\nfrom pathlib import Path\nfrom types import SimpleNamespace\n'
    code+=f'a=SimpleNamespace(output=Path({str(tmp_path)!r}))\n'
    code+='atexit.register(lambda: (a.output/"teardown").write_text("ran"))\n'
    code+=ast.unparse(hook)+'\nsys.excepthook=training_exception\nraise RuntimeError("simulated CUDA failure")\n'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
    assert result.returncode==1
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['stage']=='failed' and status['error']=='simulated CUDA failure'
    assert not (tmp_path/'teardown').exists()
    assert 'RuntimeError: simulated CUDA failure' in result.stderr
