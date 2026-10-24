"""Check a possible command/actual-rate gain mismatch without fitting fall labels."""
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def main():
    root=Path('runs/zero_middle_nominal_full_v1');sources={};rows=[]
    for row in json.loads((root/'results.json').read_text())['cases']:
        file=root/'evaluation'/(row['name']+'.npz');sources[str(file.resolve())]=digest(file)
        with np.load(file,allow_pickle=False) as z:
            command=z['final_command'][:,0];actual=z['actual_delta_rate']
            selected=(z['time']<4.)&(np.abs(command)>.15)
            gain=float(np.dot(command[selected],actual[selected])/np.dot(command[selected],command[selected]))
            rows.append(dict(name=row['name'],actual_speed_request_m_s=row['actual_requested_speed_m_s'],
                             samples=int(selected.sum()),gain_through_origin=gain,
                             rate_rmse_rad_s=float(np.sqrt(np.mean((actual[selected]-command[selected])**2))),
                             command_range_rad_s=[float(command.min()),float(command.max())],
                             measured_range_rad_s=[float(actual.min()),float(actual.max())]))
    code=Path('learning/src/sttw_control/teleop_env.py');sources[str(code.resolve())]=digest(code)
    code=Path('learning/src/sttw_control/closed_loop_kernel.py');sources[str(code.resolve())]=digest(code)
    atomic_json(Path('runs/nominal_steer_tracking_audit_v1.json'),dict(cases=rows,source_sha256=sources,
                audit_source_sha256=digest(__file__),new_physics_ticks=0,optimizer_updates=0,
                hypothesis='Large constant steering command attenuation could explain optimistic force units',
                scope='Unforced nominal first4s, abs(command)>.15rad/s; held command versus end-of-tick measured rate, not a dynamic transfer-function identification',
                conclusion='Observed nominal gain is near1; do not multiply force margins by0.8 based on an assumed actuator attenuation. Large-force coupling/saturation remains untested here.'))
    print(json.dumps(rows))


if __name__=='__main__':main()
