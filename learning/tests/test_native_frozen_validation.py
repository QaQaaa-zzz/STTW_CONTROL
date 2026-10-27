"""Runtime must bind actual frozen weights and the external frame-scale spec."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from sttw_control.native_frozen_validation import check_model_inputs


class RuntimeIdentityTests(unittest.TestCase):
    def test_weight_and_external_spec_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / 'alpha0/frozen_config_stage52.json'
            spec.parent.mkdir()
            files = {'actor': root / 'actor.pkl', 'checkpoint': root / 'checkpoint.pt', 'spec': spec}
            for name, file in files.items():
                file.write_bytes(name.encode())
            hashes = {name: hashlib.sha256(file.read_bytes()).hexdigest() for name, file in files.items()}
            plan = dict(h75=dict(actor=str(files['actor']), actor_sha256=hashes['actor'],
                                checkpoint=str(files['checkpoint']), checkpoint_sha256=hashes['checkpoint']),
                        initial_source=dict(upper_run=str(root), upper_config_file_sha256={'0': hashes['spec']}))
            check_model_inputs(plan)
            for name, file in files.items():
                with self.subTest(changed_input=name):
                    original = file.read_bytes()
                    file.write_bytes(original + b' changed')
                    with self.assertRaises(ValueError):
                        check_model_inputs(plan)
                    file.write_bytes(original)
                    check_model_inputs(plan)


if __name__ == '__main__':
    unittest.main()
