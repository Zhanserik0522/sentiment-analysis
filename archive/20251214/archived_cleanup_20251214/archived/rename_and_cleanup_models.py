"""Rename three model artifacts to clear names and archive 'junk' files safely.

Behavior:
- Rename (if present):
    models/best_model_final_trained.joblib -> models/model_binary_best.joblib
    models/best_model_quick_retrain.joblib -> models/model_multi_quick.joblib
    models/best_model_negation_ngram12.joblib -> models/model_negation_retrain.joblib

- Move obvious artifact files into cleanup/<timestamp>/ (don't delete):
    models/confusion_matrix.png
    models/top_tfidf_features_lr.png
    models/random_search_summary.csv

- Archive any __pycache__ directories into cleanup/<timestamp>/pycache_archives/ (move files)

This script is conservative and keeps all moved files in `cleanup/<ts>/` so you can restore them.
"""
from pathlib import Path
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
CLEANUP = ROOT / 'cleanup' / str(int(time.time()))
CLEANUP.mkdir(parents=True, exist_ok=True)

# rename map
rename_map = {
    'best_model_final_trained.joblib': 'model_binary_best.joblib',
    'best_model_quick_retrain.joblib': 'model_multi_quick.joblib',
    'best_model_negation_ngram12.joblib': 'model_negation_retrain.joblib',
}

renamed = []
for old_name, new_name in rename_map.items():
    src = MODELS / old_name
    dst = MODELS / new_name
    if src.exists():
        if dst.exists():
            print(f'Target {dst} already exists, skipping rename of {src}')
        else:
            shutil.move(str(src), str(dst))
            renamed.append((str(src), str(dst)))

# files to archive
to_archive = [
    MODELS / 'confusion_matrix.png',
    MODELS / 'top_tfidf_features_lr.png',
    MODELS / 'random_search_summary.csv',
]
archived = []
for f in to_archive:
    if f.exists():
        target = CLEANUP / f.name
        shutil.move(str(f), str(target))
        archived.append((str(f), str(target)))

# move __pycache__ contents
pycache_targets = []
for p in ROOT.rglob('__pycache__'):
    dest = CLEANUP / 'pycache_archives' / p.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        for child in p.iterdir():
            if child.is_file():
                target = CLEANUP / 'pycache_archives' / child.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(child), str(target))
                pycache_targets.append((str(child), str(target)))
        # try to remove dir if empty
        try:
            p.rmdir()
        except Exception:
            pass
    except Exception as e:
        print(f'Failed to archive pycache {p}: {e}')

print('\nRenamed:')
for a, b in renamed:
    print(f' - {a} -> {b}')
print('\nArchived artifacts:')
for a, b in archived:
    print(f' - {a} -> {b}')
print('\nArchived pycache files:')
for a, b in pycache_targets:
    print(f' - {a} -> {b}')
print('\nCleanup folder:', CLEANUP)
print('Done.')
