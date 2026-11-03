"""Build an allowlisted local source candidate; never uploads anything."""
import argparse
import csv
import hashlib
import io
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {'pyproject.toml', 'MANIFEST.in', 'README.md', 'LICENSE', 'CITATION.cff', '.gitignore', 'THIRD_PARTY_NOTICES.md', 'CONTRIBUTING.md'}
DIRECTORIES = {'src', 'tests', 'docs', 'examples', 'scripts', '.github'}
PRIVATE_DOCS = {'paper_alignment.md', 'method_and_provenance.md', 'validation.md', 'release.md'}


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Release output already exists')
    paths = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] == 'docs' and (path.name in PRIVATE_DOCS or path.name.startswith('review_')):
            continue
        if '__pycache__' in relative.parts or any(p.endswith('.egg-info') for p in relative.parts) or path.suffix in {'.pyc', '.pyo'}:
            continue
        if relative.as_posix() in FILES or relative.parts[0] in DIRECTORIES:
            if path.suffix not in {'.py', '.md', '.toml', '.in', '.cff', '.json', '.ipynb', '.yml'} and path.name not in {'.gitignore', 'LICENSE'}:
                raise ValueError('Unexpected file in release scope: ' + relative.as_posix())
            text = path.read_text(encoding='utf-8')
            if re.search(r'[A-Za-z]:[\\/](?:Users|Projects|backup)[\\/]|(?:hf_|ghp_)[A-Za-z0-9]{20,}', text):
                raise ValueError('Local path or credential candidate: ' + relative.as_posix())
            if path.suffix == '.ipynb':
                import json
                nb = json.loads(text)
                if any(c.get('outputs') or c.get('execution_count') is not None for c in nb['cells'] if c['cell_type'] == 'code'):
                    raise ValueError('Notebook contains executed output')
            paths.append((relative.as_posix(), path))
    rows = [{'path': n, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for n, p in paths]
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['path', 'bytes', 'sha256'])
    writer.writeheader()
    writer.writerows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in paths:
            entry = zipfile.ZipInfo('cogtask-ci/' + name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
        entry = zipfile.ZipInfo('cogtask-ci/SHA256SUMS.csv', (1980, 1, 1, 0, 0, 0))
        archive.writestr(entry, buffer.getvalue().encode('utf-8'))
    print(f'Local candidate: {output}; {len(paths)} source files; no results/local_data included')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.output)
