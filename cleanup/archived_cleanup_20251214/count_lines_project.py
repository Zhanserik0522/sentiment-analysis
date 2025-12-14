"""Count lines of code in the project, excluding common dependency folders.

Excludes: venv, models/archived, .git, node_modules
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_PARTS = ('venv', str(ROOT / 'models' / 'archived'), '.git', 'node_modules')

def is_excluded(p: Path):
    s = str(p)
    return any(x in s for x in EXCLUDE_PARTS)

def count_py(p: Path):
    n = 0
    try:
        for ln in p.read_text(encoding='utf8', errors='ignore').splitlines():
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            n += 1
    except Exception:
        pass
    return n

def count_ipynb(p: Path):
    n = 0
    try:
        data = json.loads(p.read_text(encoding='utf8', errors='ignore'))
        for cell in data.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            for ln in cell.get('source', []):
                if isinstance(ln, str) and ln.strip():
                    n += 1
    except Exception:
        pass
    return n

def main():
    py = 0
    ip = 0
    files = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():
            continue
        if is_excluded(p):
            continue
        if p.suffix == '.py':
            n = count_py(p)
            py += n
            files.append((p.relative_to(ROOT), 'py', n))
        elif p.suffix == '.ipynb':
            n = count_ipynb(p)
            ip += n
            files.append((p.relative_to(ROOT), 'ipynb', n))

    total = py + ip
    print('Project lines (excluding venv/.git/models/archived):')
    print('  Python (.py):', py)
    print('  Notebooks (.ipynb code lines):', ip)
    print('  Total:', total)
    print('\nTop project files by lines:')
    for p, t, n in sorted(files, key=lambda x: x[2], reverse=True)[:20]:
        print(f'  {t:6} {n:6}  {p}')

if __name__ == '__main__':
    main()
