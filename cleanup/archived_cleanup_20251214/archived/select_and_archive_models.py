"""Select best binary and best multiclass model among saved .joblib artifacts
and archive the rest into models/archived/<timestamp>/.

Selection metric: accuracy on data/sample_sentiment.csv. If that file is
missing, the script will not run and will show instructions.

Usage:
    python scripts/select_and_archive_models.py
"""
from pathlib import Path
import shutil
import time
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
ARCHIVE_DIR = MODELS_DIR / 'archived'
SAMPLE_CSV = ROOT / 'data' / 'sample_sentiment.csv'

if not SAMPLE_CSV.exists():
    print('data/sample_sentiment.csv not found. Please provide a small labeled CSV for evaluation.')
    sys.exit(1)

# Try to import pandas and joblib, fallbacks if missing
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import joblib
except Exception:
    joblib = None


def load_sample():
    if pd is None:
        # very small fallback parser
        texts = []
        labels = []
        with open(SAMPLE_CSV, 'r', encoding='utf-8') as fh:
            header = fh.readline()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                label = parts[-1].strip()
                text = ','.join(parts[:-1]).strip().strip('"')
                texts.append(text)
                labels.append(label)
        return texts, labels
    else:
        df = pd.read_csv(SAMPLE_CSV)
        # try to find text/label columns
        text_col = None
        label_col = None
        for c in df.columns:
            if c.lower() in ('text', 'review', 'sentence', 'content'):
                text_col = c
            if c.lower() in ('label', 'sentiment', 'target'):
                label_col = c
        if text_col is None or label_col is None:
            # fallback to first two
            text_col = df.columns[0]
            label_col = df.columns[1]
        return df[text_col].astype(str).tolist(), df[label_col].astype(str).tolist()


texts, labels = load_sample()

# gather joblib files
candidates = list(MODELS_DIR.glob('**/*.joblib'))
if not candidates:
    print('No .joblib files found under models/. Nothing to do.')
    sys.exit(0)

print(f'Found {len(candidates)} model files. Evaluating...')

results = []
for p in candidates:
    try:
        if joblib is not None:
            m = joblib.load(p)
        else:
            import pickle
            with open(p, 'rb') as fh:
                m = pickle.load(fh)
    except Exception as e:
        print(f'Failed to load {p}: {e}')
        continue
    # determine classes
    classes = getattr(m, 'classes_', None)
    if classes is None:
        # try to unwrap pipeline's final step
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(m, Pipeline):
                model = m.steps[-1][1]
                classes = getattr(model, 'classes_', None)
        except Exception:
            classes = None
    # make predictions on sample
    y_pred = []
    try:
        for t in texts:
            # assume model accepts raw strings or cleaned strings already
            y_pred.append(str(m.predict([t])[0]))
    except Exception as e:
        print(f'Prediction failed for {p}: {e}')
        continue
    # compute accuracy
    correct = sum(1 for a, b in zip(y_pred, labels) if str(a) == str(b))
    acc = correct / len(labels) if labels else 0.0
    results.append({'path': str(p), 'classes': None if classes is None else list(classes), 'accuracy': acc})

# choose best binary (len==2) and best multiclass (len==3)
best_binary = None
best_multi = None
for r in results:
    cls = r['classes']
    if cls is None:
        continue
    if len(cls) == 2:
        if best_binary is None or r['accuracy'] > best_binary['accuracy']:
            best_binary = r
    if len(cls) == 3:
        if best_multi is None or r['accuracy'] > best_multi['accuracy']:
            best_multi = r

print('\nEvaluation complete. Summary:')
print(json.dumps({'best_binary': best_binary, 'best_multi': best_multi}, indent=2))

# archive others
ts = int(time.time())
archive_target = ARCHIVE_DIR / str(ts)
archive_target.mkdir(parents=True, exist_ok=True)
kept = []
archived = []
for r in results:
    path = Path(r['path'])
    keep = False
    if best_binary and r['path'] == best_binary['path']:
        keep = True
    if best_multi and r['path'] == best_multi['path']:
        keep = True
    if keep:
        kept.append(r['path'])
    else:
        # move to archive
        try:
            shutil.move(str(path), str(archive_target / path.name))
            archived.append(str(archive_target / path.name))
        except Exception as e:
            print(f'Failed to move {path}: {e}')

print('\nKept:')
for k in kept:
    print(' -', k)
print('\nArchived:')
for a in archived:
    print(' -', a)

print('\nDone.')
