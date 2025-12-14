from pathlib import Path
import sys
import joblib

"""Single-model smoke test.

Usage:
  python scripts/smoke_test_models.py [path/to/model.joblib]

If no argument is given the script defaults to `models/model_final_imdb_augmented.joblib`.
"""

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'models' / 'model_final_imdb_augmented.joblib'

examples = [
    "I love this product! It's fantastic and works perfectly.",
    "This is the worst experience I've had. Totally disappointed.",
    "It's okay, not great but not terrible.",
    "Absolutely amazing — exceeded expectations!",
    "Terrible, will never buy again.",
    "I hate it"
]


def clean_text(s: str) -> str:
    try:
        from src.preprocessing import clean_text as _clean
        return _clean(s)
    except Exception:
        return ' '.join([w.lower() for w in s.split() if w.isalnum() or w.isalpha()])


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print('Testing single model:', model_path)
    if not model_path.exists():
        print('Model not found:', model_path)
        return 2
    try:
        m = joblib.load(model_path)
    except Exception as e:
        print('Failed to load model:', e)
        return 3
    classes = getattr(m, 'classes_', None)
    print('classes =', classes)
    for t in examples:
        ct = clean_text(t)
        try:
            pred = m.predict([ct])[0]
        except Exception as e:
            pred = f'ERROR: {e}'
        probs_display = None
        try:
            if hasattr(m, 'predict_proba'):
                probs = m.predict_proba([ct])[0]
                cls = getattr(m, 'classes_', [])
                probs_display = {str(c): float(p) for c, p in zip(cls, probs)}
        except Exception as e:
            probs_display = f'ERROR: {e}'
        print('\nTEXT:', t)
        print('cleaned:', ct)
        print('PRED:', pred)
        print('PROBS:', probs_display)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
models = [
    ROOT/'models'/'model_binary_best.joblib',
    ROOT/'models'/'model_quick_retrained.joblib',
    ROOT/'models'/'model_quick_retrained_negation.joblib',
    ROOT/'models'/'model_grid_best.joblib',
    ROOT/'models'/'model_multi_quick.joblib',
    ROOT/'models'/'model_grid_best_calibrated.joblib'
]
examples = [
    "I love this product! It's fantastic and works perfectly.",
    "This is the worst experience I've had. Totally disappointed.",
    "It's okay, not great but not terrible.",
    "Absolutely amazing — exceeded expectations!",
    "Terrible, will never buy again.",
    "I hate it"
]
for m in models:
    print('\n--- Testing model:', m)
    if not m.exists():
        print('Model not found:', m)
        continue
    try:
        p = joblib.load(m)
    except Exception as e:
        print('Failed to load', m, e)
        continue
    classes = getattr(p, 'classes_', None)
    print('classes =', classes)
    for t in examples:
        try:
            pred = p.predict([t])[0]
        except Exception as e:
            pred = f'ERROR: {e}'
        probs = None
        try:
            if hasattr(p, 'predict_proba'):
                probs = p.predict_proba([t])[0].tolist()
        except Exception as e:
            probs = f'ERROR: {e}'
        print('\nTEXT:', t)
        print('PRED:', pred)
        print('PROBS:', probs)
