"""Offline source/docs/citation checks. Run from a source checkout."""
import ast
import json
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    for base in ('src', 'tests', 'scripts'):
        for path in (root/base).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    citation = yaml.safe_load((root/'CITATION.cff').read_text(encoding='utf-8'))
    for field in ('cff-version', 'message', 'title', 'authors'):
        if not citation.get(field):
            raise ValueError('Missing citation field: ' + field)
    if citation['license'] != 'MIT' or not all(a.get('name') or a.get('family-names') for a in citation['authors']):
        raise ValueError('Invalid licence or author metadata')
    if not (root/'LICENSE').read_text(encoding='utf-8').startswith('MIT License'):
        raise ValueError('Missing MIT licence')
    for path in list(root.glob('*.md')) + list((root/'docs').glob('*.md')):
        text = path.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#', 1)[0]
            if not (path.parent/target).exists():
                raise ValueError(f'Broken link in {path.name}: {target}')
    for path in (root/'examples').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    raise ValueError('Executed notebook output: ' + path.name)
                ast.parse(''.join(cell['source']))
    print('Python/Notebook syntax, clean examples, relative documentation links and required citation/licence fields: PASS')


if __name__ == '__main__':
    check()
