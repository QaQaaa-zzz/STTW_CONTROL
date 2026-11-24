"""Execute frozen, same-slot evaluations in one Isaac application and USD scene."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


def jobs(source, common):
    if str(source) == '-':
        records = (json.loads(line) for line in sys.stdin if line.strip())
    else:
        records = json.loads(source.read_text())
        if not isinstance(records, list) or not records:
            raise ValueError('The resident manifest must be a nonempty list of jobs')
    keys = {'config', 'terrain', 'case_manifest', 'output'}
    for record in records:
        if set(record) != keys:
            raise ValueError(f'Each resident job requires exactly {sorted(keys)}')
        yield argparse.Namespace(**{**vars(common), **{k:Path(v) for k,v in record.items()}})


class FixedSlotSession:
    """Keep immutable USD geometry, with new PhysX and controller state for every job."""

    def __init__(self, app_started, app_ready):
        self.app_started = app_started
        self.app_ready = app_ready
        self.plant = None
        self.count = 0

    def prepare(self, settings, cases, output, create):
        # Hold fixed initial state, group draws, asset, terrain inventory and slot
        # order. Only the CEM target and the associated policy selection can vary.
        mutable = {'alphas','lane_alphas','zero_middle','case_labels','evaluation_role',
                   'oracle_knots','oracle_knots_sha256'}
        fixed = {k:v for k,v in settings.items() if k not in mutable}
        case_inputs = [{k:v for k,v in c.items() if k not in mutable and k!='output'} for c in cases]
        signature = (fixed,case_inputs)
        options = settings['plant_options']
        if (options.get('pendulum_rest') is not None or options.get('servo_mode','explicit')!='explicit'
                or not settings.get('fixed_randomization_groups')):
            raise ValueError('The resident PCRP scene requires its frozen physical groups and explicit bike actuators')
        self.setup_started = time.monotonic()
        if self.plant is None:
            self.signature = signature
            self.scene_output = output
            self.plant = create()
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.plant.device)
            self.nominal = self._initial_readback()
        else:
            if signature != self.signature:
                raise ValueError('Resident scene inputs or per-case physical slot order changed')
            self.plant.reset_fixed_scene(self.nominal)
            for key, value in self._initial_readback().items():
                if not torch.equal(self.nominal[key],value):
                    raise ValueError(f'Native {key} after hard reset differs from the first cold start')
            torch.set_rng_state(self.cpu_rng)
            torch.cuda.set_rng_state(self.cuda_rng,self.plant.device)
            for name in ('input','usd'):
                (output/name).symlink_to(os.path.relpath(self.scene_output/name,output),target_is_directory=True)
        self.plant_ready = time.monotonic()
        return self.plant

    def _initial_readback(self):
        view = self.plant.robot.root_physx_view
        clock = torch.tensor([self.plant.sim.current_time, self.plant.sim.current_time_step_index],dtype=torch.float64)
        initial = dict(native_clock=clock, masses=view.get_masses(), inertias=view.get_inertias(),
                       default_mass=self.plant.robot.data.default_mass, default_inertia=self.plant.robot.data.default_inertia,
                       coms=view.get_coms(),
                       materials=view.get_material_properties(), roots=self.plant.roots,
                       joints=self.plant.dofs, contact=self.plant.contact)
        return {key:value.detach().cpu().clone() for key,value in initial.items()}

    def finish(self,output,started,sampling,exporting,finished):
        report = dict(candidate=self.count, process_id=os.getpid(), scene_source=str(self.scene_output),
                      reset='cold construction' if self.count==0 else 'stop and restart native physics on identical USD stage',
                      initial_native_clock_s=float(self.nominal['native_clock'][0]),
                      initial_native_physics_step=int(self.nominal['native_clock'][1]),
                      app_startup_s=self.app_ready-self.app_started if self.count==0 else 0.,
                      plant_setup_or_reset_s=self.plant_ready-self.setup_started,
                      candidate_initialization_s=sampling-started, sampling_s=exporting-sampling,
                      export_s=finished-exporting, total_s=finished-started,
                      queue_scope='one shared GPU lease for this resident job list; exterior waiting is measured by caller')
        (output/'resident_timing.json').write_text(json.dumps(report,indent=2)+'\n')
        self.count += 1
