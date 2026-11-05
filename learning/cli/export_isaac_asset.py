"""Generate an immutable Isaac import bundle with the source physical contract."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.isaac_asset import export_isaac_asset
from sttw_control.model import DEFAULT_XML

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=DEFAULT_XML)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
contract = export_isaac_asset(args.output, args.source)
print(args.output/'contract.json', contract['exported_xml_sha256'])
