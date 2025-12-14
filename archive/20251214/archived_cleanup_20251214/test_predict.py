import sys
import pathlib
from pathlib import Path

# Ensure project root is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import joblib
    from src.preprocessing import clean_text
except Exception as e:
    print('Import error:', e)
    raise

MODEL_PATH = Path(ROOT) / 'models' / 'best_model.joblib'
if not MODEL_PATH.exists():
    print('Model file not found at', MODEL_PATH)
    raise SystemExit(1)

pipeline = joblib.load(MODEL_PATH)

examples = [
    "I love this product! It's fantastic and works perfectly.",
    "This is the worst experience I've had. Totally disappointed.",
    "It's okay, not great but not terrible.",
    "Absolutely amazing — exceeded expectations!",
    "Terrible, will never buy again."
]

print('Loaded pipeline from', MODEL_PATH)
for t in examples:
    cleaned = clean_text(t)
    try:
        pred = pipeline.predict([cleaned])[0]
    except Exception as e:
        print('Prediction error for text:', t)
        print(e)
        pred = None
    print('\nTEXT:')
    print(t)
    print('CLEANED:')
    print(cleaned)
    print('PREDICTION:')
    print(pred)
