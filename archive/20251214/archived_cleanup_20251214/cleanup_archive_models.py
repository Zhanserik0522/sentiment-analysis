"""Archive non-essential model artifacts into models/archived/<timestamp>/.
Keeps a small whitelist of model files and moves others to archive.
"""
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
ARCH = MODELS / 'archived' / f'cleanup_{int(time.time())}'
ARCH.mkdir(parents=True, exist_ok=True)

# whitelist - keep these in models/
keep = {
    'model_binary_best.joblib',
    'model_multi_quick.joblib',
    'model_quick_retrained.joblib',
    'model_quick_retrained_negation.joblib',
    'model_negation_retrain.joblib'
}

moved = []
for p in MODELS.glob('*.joblib'):
    if p.name not in keep:
        dest = ARCH / p.name
        shutil.move(str(p), str(dest))
        moved.append(dest)

# also move full_search_quick and full_search subfolders (if present)
for sub in ['full_search', 'full_search_quick']:
    sp = MODELS / sub
    if sp.exists():
        dst = ARCH / sub
        shutil.move(str(sp), str(dst))

print('Archived to:', ARCH)
print('Moved files:')
for m in moved:
    print(' -', m)
