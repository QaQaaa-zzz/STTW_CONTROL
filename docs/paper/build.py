"""Build and inventory the single integrated STTW manuscript."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--engine', default='tectonic')
    args = parser.parse_args()
    source = (HERE / 'MANUSCRIPT.tex').read_text()
    figures = re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', source)
    inputs = re.findall(r'\\input\{([^}]+)\}', source)
    files = [HERE / 'MANUSCRIPT.tex', HERE / 'local_font_maps.tex']
    files += [HERE / path for path in figures + inputs]
    files += sorted((ROOT / 'learning/configs').glob('task_matrix_v2_*.json'))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError('missing article input: ' + ', '.join(missing))
    if args.pdf:
        subprocess.run([args.engine, 'MANUSCRIPT.tex', '--outdir', '.', '--keep-logs'],
                       check=True, cwd=HERE)
    manifest = dict(scope='Single integrated article; no separate supplement',
                    manuscript_sha256=digest(HERE / 'MANUSCRIPT.tex'),
                    inputs={str(path.relative_to(ROOT)): digest(path) for path in files},
                    figure_count=len(figures), pdf_built=args.pdf,
                    physics_steps=0, optimizer_updates=0,
                    data_role='Exploratory matched comparisons; source roles stated in article')
    if args.pdf:
        manifest['pdf_sha256'] = digest(HERE / 'MANUSCRIPT.pdf')
    (HERE / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
