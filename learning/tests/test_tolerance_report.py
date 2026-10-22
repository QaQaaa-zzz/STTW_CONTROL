"""Synthetic reporting fixtures are not scientific evaluation evidence."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('report_tolerance',Path(__file__).parents[1]/'cli/report_tolerance_comparison.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_report_renders_all_groups_and_retains_paired_losses(tmp_path):
 policies=['parent','baseline_update50','static_update50','dynamic_update50']
 protocol=dict(policies=policies,routes=['synthetic'],speeds=[2.45],endpoints=[0,1],forces=[0,2],onsets=[.1],pulse_duration=.1)
 (tmp_path/'manifest.json').write_text(json.dumps(dict(protocol=protocol,fixture_only=True)))
 (tmp_path/'status.json').write_text(json.dumps(dict(phase='completed')))
 rows=[];time=np.arange(50)*.005
 for policy in policies:
  for alpha in (0,1):
   for force in (0,2):
    name=f'{policy}_{alpha}_{force}'
    success=force==0
    if policy=='dynamic_update50':success=not success
    rows.append(dict(policy=policy,route='synthetic',speed=2.45,alpha=alpha,force_N=force,onset_s=.1,name=name,success=success,physical_failure=False,domain_failure=False,prefix_valid=True,speed_rmse=.1,cross_track_rmse=.02))
    np.savez(tmp_path/(name+'.npz'),time=time,actual_xy=np.column_stack((2.45*time,alpha*.02+force*time*.01)),path_progress=2.45*time)
 np.savez(tmp_path/'synthetic_path.npz',xy=np.column_stack((np.arange(100)*.02,np.zeros(100))),s=np.arange(100)*.02)
 (tmp_path/'results.json').write_text(json.dumps(dict(cases=rows)))
 out=module.report(tmp_path);summary=json.loads((out/'summary.json').read_text())
 assert len(summary['figures'])==2
 assert all((out/name).stat().st_size>1000 for name in summary['figures'])
 pair=summary['paired']['a0_baseline_update50_vs_dynamic_update50']
 assert pair['improved']==1 and pair['regressed']==1 and pair['difference']==0
 text=(out/'REPORT.md').read_text();assert '\n| Comparison' in text and '\\n' not in text
 rows.pop();(tmp_path/'results.json').write_text(json.dumps(dict(cases=rows)))
 with pytest.raises(ValueError,match='missing or duplicated'):module.report(tmp_path)

def test_smoke_is_never_presented_as_performance(tmp_path):
 (tmp_path/'manifest.json').write_text(json.dumps(dict(engineering_only=True,protocol={})))
 with pytest.raises(ValueError,match='smoke'):module.report(tmp_path)
