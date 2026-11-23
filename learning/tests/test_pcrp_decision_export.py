"""Offline candidate indices retain the exact location of every original trace and score."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cli'))
from export_pcrp_decision_evidence import export, offline_links, read_csv, trace_index, write_csv


def test_each_candidate_is_indexed_but_only_warm_starts_and_winner_are_copied(tmp_path):
    root=tmp_path/'source'
    out=tmp_path/'output'
    folder=Path('e1/E1a/analysis/ridge/rollouts')
    (root/folder).mkdir(parents=True)
    raw_blob=root/'small_fixture'
    raw_blob.write_bytes(b'fixture')
    rows=[]
    for generation in range(11):
        for candidate in range(6 if generation==0 else 32):
            name=f'g{generation:02d}_c{candidate:02d}'
            raw=root/'physics'/name
            raw.mkdir(parents=True)
            (raw/'trajectory.npz').hardlink_to(raw_blob)
            for ext in ['npz','csv']:
                (root/folder/(name+'.'+ext)).hardlink_to(raw_blob)
            rows.append(dict(case='ridge',method=name,generation=generation,candidate=candidate,raw=str(raw),
                             lane=0,npz=name+'.npz',csv=name+'.csv'))
    write_csv(root/folder/'index.csv',rows)
    index=trace_index(root,out,'E1a','ridge',(4,9))
    assert len(index)==326 and sum(i['included_in_package'] for i in index)==7
    assert len(list((out/folder).glob('*.npz')))==7 and not list(out.rglob('trajectory.npz'))
    offline=read_csv(out/folder/'index.csv')
    for r in offline:
        assert int(r['csv_bytes'])==int(r['npz_bytes'])==int(r['physics_npz_bytes'])==7
        assert Path(r['npz_source']).is_file() and bool(r['csv'])==bool(int(r['included_in_package']))
    # Every offline index is checked against the source, even for a non-selected candidate.
    Path(index[-1]['csv_source']).unlink()
    with pytest.raises(FileNotFoundError):
        trace_index(root,out,'E1a','ridge',(4,9))


def test_report_links_resolve_offline_and_incomplete_runs_do_not_publish(tmp_path):
    root=tmp_path/'source'; root.mkdir()
    out=tmp_path/'out'; out.mkdir()
    panel=tmp_path/'old_case'/'xy.png'
    panel.parent.mkdir()
    panel.write_bytes(b'fixture')
    page=out/'README.md'
    page.write_text(f'[old XY]({panel}) · [original code](/original/machine/script.py:12)')
    offline_links(root,out,page)
    assert '](/' not in page.read_text()
    assert (out/'support/old_case/xy.png').read_bytes()==panel.read_bytes()
    page.write_text('[missing](missing.csv)')
    with pytest.raises(ValueError,match='Broken offline'):
        offline_links(root,out,page)
    with pytest.raises(FileNotFoundError):
        export(root,tmp_path/'never_published')
    assert not (tmp_path/'never_published').exists()
