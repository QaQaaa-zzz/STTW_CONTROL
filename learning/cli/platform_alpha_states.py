"""Read verified handbacks of the five approach candidates from the 30-root development panel."""
import json
from pathlib import Path

import pandas as pd


def development_states(panel):
    rows = pd.read_csv(panel/'physical.csv').query('speed==2 and verified and (alpha==0 or approach==1)').copy()
    rows['candidate'] = rows.approach.where(rows.alpha==0, (2+2*rows.alpha).astype(int))
    rows['departure'] = rows.dep
    groups = rows.trajectory.map(lambda path: Path(path).parents[1])
    for folder, group in rows.groupby(groups):
        results = json.loads((folder/'status.json').read_text())['results']
        times = {r['case_dir']: r['handoff_s'] for r in results}
        rows.loc[group.index,'handoff_s'] = group.trajectory.map(lambda p: times[Path(p).parent.name])
    return rows
