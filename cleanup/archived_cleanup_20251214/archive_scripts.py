"""Move non-essential scripts to scripts/archived/ to declutter the workspace."""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
ARCH = SCRIPTS / 'archived'
ARCH.mkdir(parents=True, exist_ok=True)

keep = {
    'smoke_test_models.py',
    'quick_retrain_sample.py',
    'quick_retrain_negation.py',
    'test_predict.py',
    'cleanup_archive_models.py',
    'archive_scripts.py'
}

moved = []
for p in SCRIPTS.glob('*.py'):
    if p.name not in keep:
        dest = ARCH / p.name
        shutil.move(str(p), str(dest))
        moved.append(dest)

print('Moved scripts to', ARCH)
for m in moved:
    print(' -', m)
