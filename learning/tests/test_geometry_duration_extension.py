import copy
import importlib.util
import json
from pathlib import Path
import types
import sys
import numpy as np
import pytest
from sttw_control.native_goal_run import tolerance_inventory,apply_route_context_variant,force_pulses,frozen_control_core_identity


def protocol():return json.loads((Path(__file__).parents[1]/'configs/h75_geometry_duration_extension240.json').read_text())


def cli_module(name):
    file=Path(__file__).parents[1]/'cli'/name;spec=importlib.util.spec_from_file_location('_geometry_test_'+file.stem,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_exact240_duration_realforce_and_24groups():
    p=protocol();rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==240
    assert p['maximum_lower_ticks']==240*10/.005==480000
    groups={}
    for r in rows:groups.setdefault((r['bank_index'],r['route'],r['event']),[]).append(r)
    assert len(groups)==24 and all(len(v)==10 for v in groups.values())
    assert all(r['duration_s']==.15 and r['onset_s']==2.6 for r in rows)
    pulses=np.array(force_pulses(rows,3.5));np.testing.assert_allclose(pulses[:,1]-pulses[:,0],.15)
    assert p['coordination']['hypothetical_duration_s']==.1
    assert sum(r['force_N']==0 for r in rows)==80


def test_explicit_contexts_really_replace_plan_geometry_not_only_inventory():
    p=protocol();old_contexts=[dict(route='S_left_90_R1p8',radius_m=1.8,angle_deg=90,speed=2.8,direction=1.)]
    plan=dict(mode=p['mode'],protocol=p,contexts=copy.deepcopy(old_contexts))
    actual=apply_route_context_variant(plan)
    assert actual is plan and plan['contexts']==p['route_contexts']
    assert [x['angle_deg'] for x in plan['contexts']]==[60,60,120,120]
    assert all(x['radius_m']==2.2 and x['speed']==2.6 for x in plan['contexts'])
    # The same runtime geometry constructor consumes these explicit fields.
    from sttw_control.path_command_scenarios import _integrate_route
    import jax.numpy as j
    pathcfg=dict(spacing_m=.02,curvature_transition_length_m=.5,straight_prefix_m=3.,goal_section_offset_after_turn_m=4.)
    context=plan['contexts'][2]
    path=_integrate_route(j.zeros(3),pathcfg,2,context['radius_m'],np.deg2rad(context['angle_deg']),context['speed'],context['direction'],0.,0.).path
    assert path.xy.shape==(5001,2)
    assert float(path.goal)==pytest.approx(20.21533845,abs=2e-6)
    assert float(j.max(abs(path.curvature)))==pytest.approx(1/2.2)
    assert old_contexts[0]['radius_m']==1.8


def test_unknown_variants_and_changed_context_are_rejected_legacy200_unchanged():
    p=protocol();p['variant']='unregistered'
    with pytest.raises(ValueError):tolerance_inventory(p)
    p=protocol();p['route_contexts'][2]['radius_m']=1.9
    with pytest.raises(ValueError):tolerance_inventory(p)
    old=json.loads((Path(__file__).parents[1]/'configs/h75_frozen_confirmation200.json').read_text());rows=tolerance_inventory(old)
    assert len(rows)==200 and all(r['duration_s']==.1 for r in rows)
    plan=dict(mode=old['mode'],protocol=old,contexts=['original']);assert apply_route_context_variant(plan)['contexts']==['original']


def test_initial_layout240_and_unknown_variant_rejects():
    m=cli_module('audit_native_goal_initial.py');p=protocol()
    assert m.confirmation_initial_layout(dict(mode=p['mode'],protocol=p))==(240,10)
    p['variant']='other'
    with pytest.raises(ValueError):m.confirmation_initial_layout(dict(mode=p['mode'],protocol=p))


def test_core_identity_checks_recursive_local_helpers(tmp_path):
    roots=['infer_probabilities','native_offsets','candidate_features','candidate_probabilities','preview_curvature','choose_alpha','coordinate']
    text='def extra(x):\n return x+1\n'+''.join(f'def {name}(x):\n return extra(x)\n' for name in roots)
    a=tmp_path/'a.py';b=tmp_path/'b.py';a.write_text(text+'\ndef exporter():\n return 0\n');b.write_text(text+'\ndef exporter():\n return 1\n')
    result=frozen_control_core_identity(a,b);assert not result['whole_files_equal'] and 'extra' in result['function_ast_sha256']
    b.write_text(b.read_text().replace('return x+1','return x+2'))
    with pytest.raises(ValueError):frozen_control_core_identity(a,b)


def test_notification_imports_are_late_and_failures_do_not_escape(tmp_path,monkeypatch):
    m=cli_module('run_native_goal_confirmation.py');before=list(sys.path);source=str(Path(__file__).parents[1]/'src')
    # Importing the CLI itself has not inserted its source path.
    assert sys.path==before
    package=types.ModuleType('sttw_control');package.__path__=['/tmp/frozen-only']
    helper=types.ModuleType('engineer_reference_bias');calls=[]
    helper.notify=lambda root,title,body:calls.append((root,title,body))
    monkeypatch.setitem(sys.modules,'sttw_control',package);monkeypatch.setitem(sys.modules,'engineer_reference_bias',helper)
    monkeypatch.setattr(sys,'path',list(before))
    assert m.notify_after_execution(tmp_path,'done','message')
    assert source in sys.path and str(Path(source)/'sttw_control') in package.__path__ and len(calls)==1
    def broken(*args):raise RuntimeError('notification unavailable')
    helper.notify=broken
    assert not m.notify_after_execution(tmp_path,'failed','original preserved')
    assert 'notification unavailable' in (tmp_path/'notification_errors.jsonl').read_text()
