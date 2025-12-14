"""Count lines of code in the repository.

Counts:
- Python (.py): non-blank, non-comment lines
- Jupyter notebooks (.ipynb): lines in code cells (non-blank)
- Other text files (README.md, .txt): optional count as 'docs'

Prints per-extension totals and overall total.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def count_py_lines(p: Path):
    cnt = 0
    try:
        for ln in p.read_text(encoding='utf8', errors='ignore').splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith('#'):
                continue
            cnt += 1
    except Exception:
        pass
    return cnt

def count_ipynb_lines(p: Path):
    cnt = 0
    try:
        data = json.loads(p.read_text(encoding='utf8', errors='ignore'))
        for cell in data.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            src = cell.get('source', [])
            for ln in src:
                if isinstance(ln, str) and ln.strip():
                    cnt += 1
    except Exception:
        pass
    return cnt

def count_text_lines(p: Path):
    try:
        return sum(1 for _ in p.read_text(encoding='utf8', errors='ignore').splitlines() if _.strip())
    except Exception:
        return 0

def main():
    totals = {'py': 0, 'ipynb': 0, 'docs': 0}
    per_file = []
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():
            if p.suffix == '.py':
                n = count_py_lines(p)
                totals['py'] += n
                per_file.append((p.relative_to(ROOT), 'py', n))
            elif p.suffix == '.ipynb':
                n = count_ipynb_lines(p)
                totals['ipynb'] += n
                per_file.append((p.relative_to(ROOT), 'ipynb', n))
            elif p.suffix in {'.md', '.txt', '.rst'}:
                n = count_text_lines(p)
                totals['docs'] += n
                per_file.append((p.relative_to(ROOT), 'docs', n))

    total_lines = sum(totals.values())
    print('Lines of code summary (non-blank / non-comment where applicable):')
    print(f"  Python (.py): {totals['py']}")
    print(f"  Notebooks (.ipynb code cells): {totals['ipynb']}")
    print(f"  Docs (.md/.txt/.rst): {totals['docs']}")
    print('  -----')
    print(f'  Total lines counted: {total_lines}')
    print('\nTop files by counted lines:')
    for p, t, n in sorted(per_file, key=lambda x: x[2], reverse=True)[:20]:
        print(f'  {t:6} {n:6}  {p}')

if __name__ == "__main__":
    main()
