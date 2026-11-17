#!/usr/bin/env python3
"""Dependency/asset checks without importing the GPU runtime."""
import ast
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(part in {'.git', '__pycache__', 'outputs', 'output', '.venv'} for part in p.relative_to(ROOT).parts)]
    pyfiles = [p for p in files if p.suffix == '.py']
    for path in pyfiles:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as error:
            errors.append(str(error))
    index = {}
    for directory in (ROOT / 'dywa/src', ROOT / 'dywa/exp/train'):
        for path in directory.rglob('*.py'):
            parts = list(path.relative_to(directory).with_suffix('').parts)
            if parts[-1] == '__init__': parts.pop()
            index['.'.join(parts)] = path
    local_roots = {key.split('.')[0] for key in index}
    built = {'cxx.franka_kin_cuda', 'cxx.ur5_kin_cuda'}
    for key, path in index.items():
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import): names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = key.split('.') if path.name == '__init__.py' else key.split('.')[:-1]
                names = ['.'.join(prefix[:len(prefix)-node.level+1] + ([node.module] if node.module else []))] if node.level else [node.module or '']
            for name in names:
                if name.split('.')[0] in local_roots and name not in index and name not in built and not any(k.startswith(name + '.') for k in index):
                    errors.append(f'{path.relative_to(ROOT)}: unresolved local import {name}')
    assets = ROOT / 'dywa/src/data/assets'
    urdfs = list(assets.rglob('*.urdf'))
    for path in urdfs:
        for mesh in ET.parse(path).iter('mesh'):
            ref = mesh.attrib['filename']
            if ref.startswith('package://'): ref = ref[len('package://'):]
            candidates = [path.parent / ref, assets / ref]
            candidates += [parent / ref for parent in path.parents if parent == assets or assets in parent.parents]
            if not any(p.is_file() for p in candidates):
                errors.append(f'{path.relative_to(ROOT)}: missing mesh {ref}')
    seer = json.loads((ROOT / 'configs/seer.json').read_text())
    if any(seer[k] for k in ('student_FT_head.feed_to_policy', 'student_FT_head.future_film_enabled', 'student_FT_head.structured_phase_film_enabled')):
        errors.append('Default SEER preset enables a future-based action branch')
    for path in files:
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f'Unexpected file >10 MB: {path.relative_to(ROOT)}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'OK: {len(pyfiles)} Python files, local imports, {len(urdfs)} URDF assets, baseline preset, file sizes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
