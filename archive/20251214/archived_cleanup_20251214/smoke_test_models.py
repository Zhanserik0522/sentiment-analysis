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
