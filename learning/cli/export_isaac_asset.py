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
parser.add_argument('--format', choices=('mjcf', 'urdf'), default='mjcf')
parser.add_argument('--wheel-vertices', type=int, default=226, help='Ellipsoid convex mesh vertex budget; historical default 226')
args = parser.parse_args()
if args.format == 'urdf':
    from sttw_control.isaac_urdf import export_isaac_urdf
    contract = export_isaac_urdf(args.output, args.source, wheel_vertices=args.wheel_vertices)
else:
    contract = export_isaac_asset(args.output, args.source)
print(args.output/'contract.json', contract['exported_xml_sha256'])
