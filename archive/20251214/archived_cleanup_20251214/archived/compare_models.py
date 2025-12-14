from pathlib import Path
import sys
import joblib
# ensure project root is on sys.path so `src` package imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_text

root = Path(__file__).resolve().parents[1]
model_dir = root / 'models'
files = []
if model_dir.exists():
    files += list(model_dir.glob('*.joblib'))
full = model_dir / 'full_search'
if full.exists():
    files += list(full.glob('*.joblib'))
files = sorted(files, key=lambda p: p.name)
text = 'i hate it'
cleaned = clean_text(text)
print('input:', text, '-> cleaned:', cleaned, '\n')
for f in files:
    try:
        m = joblib.load(f)
        classes = getattr(m, 'classes_', None)
        if hasattr(m, 'predict_proba'):
            probs = m.predict_proba([cleaned])[0]
        elif hasattr(m, 'named_steps') and hasattr(m.named_steps.get('clf'), 'predict_proba'):
            probs = m.named_steps['clf'].predict_proba(m.named_steps['vectorizer'].transform([cleaned]))[0]
        else:
            probs = None
        pred = m.predict([cleaned])[0]
        print(f.name, '-> pred:', pred, 'classes:', classes, 'probs:', probs)
    except Exception as e:
        print(f.name, 'LOAD/INFER ERROR:', e)
