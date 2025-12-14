from pathlib import Path
import sys
import joblib
import csv
import json

# ensure project root is on sys.path so `src` package imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_text

root = ROOT
model_dir = root / 'models'
files = []
if model_dir.exists():
    files += list(model_dir.glob('*.joblib'))
full = model_dir / 'full_search'
if full.exists():
    files += list(full.glob('*.joblib'))
files = sorted(files, key=lambda p: p.name)

# Examples to test across models
examples = [
    'I hate it',
    'I love this product!',
    "It's okay, not great but not terrible.",
    'Absolutely amazing — exceeded expectations!',
    'Terrible, will never buy again.',
    'good',
    'bad'
]

rows = []
for f in files:
    for text in examples:
        cleaned = clean_text(text)
        pred = None
        probs = None
        classes = None
        error = None
        try:
            m = joblib.load(f)
            classes = getattr(m, 'classes_', None)
            # try standard API first
            if hasattr(m, 'predict_proba'):
                probs = m.predict_proba([cleaned])[0].tolist()
            # then try pipeline.named_steps access
            elif hasattr(m, 'named_steps') and m.named_steps.get('clf') is not None and hasattr(m.named_steps['clf'], 'predict_proba'):
                probs = m.named_steps['clf'].predict_proba(m.named_steps['vectorizer'].transform([cleaned]))[0].tolist()
            pred = m.predict([cleaned])[0]
        except Exception as e:
            error = str(e)

        rows.append({
            'model': f.name,
            'example': text,
            'cleaned': cleaned,
            'pred': str(pred) if pred is not None else '',
            'probs': json.dumps(probs) if probs is not None else '',
            'classes': json.dumps(list(classes)) if classes is not None else '',
            'error': error or ''
        })

# Write summary CSV to models/
out = model_dir / 'compare_restored_results.csv'
with open(out, 'w', encoding='utf-8', newline='') as cf:
    fieldnames = ['model', 'example', 'cleaned', 'pred', 'probs', 'classes', 'error']
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Wrote compare report to', out)
