"""Quick retrain on data/sample_sentiment.csv — TFIDF(1,2) + LogisticRegression.
Saves model to models/model_quick_retrained.joblib
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA = ROOT / 'data' / 'sample_sentiment.csv'
OUT = ROOT / 'models' / 'model_quick_retrained.joblib'

if not DATA.exists():
    print('Sample data not found at', DATA)
    raise SystemExit(1)

df = pd.read_csv(DATA)
# normalize column names
cols = [c.lower() for c in df.columns]
if 'text' in cols:
    text_col = df.columns[cols.index('text')]
elif 'review' in cols:
    text_col = df.columns[cols.index('review')]
else:
    text_col = df.columns[0]

if 'label' in cols:
    label_col = df.columns[cols.index('label')]
elif 'sentiment' in cols:
    label_col = df.columns[cols.index('sentiment')]
else:
    label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

X = df[text_col].astype(str).tolist()
y = df[label_col].astype(str).tolist()

# Robust train/test splitting for small samples.
from collections import Counter
n = len(X)
label_counts = Counter(y)
n_classes = len(label_counts)

if n < 2:
    # Not enough data to split — train on all data and skip evaluation
    print('Warning: very small dataset (n=', n, '). Training on all samples and skipping test split.')
    X_train, y_train = X, y
    X_test, y_test = [], []
else:
    # prefer stratified split but only when safe: every class must have >=2 samples
    # and the test set must be large enough to contain at least one sample per class
    test_frac = 0.2
    est_test_count = max(1, int(test_frac * n))
    can_stratify = (n_classes > 1 and min(label_counts.values()) >= 2 and est_test_count >= n_classes)
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_frac, random_state=42, stratify=y
        )
    else:
        # fallback to non-stratified split using an integer test size when small
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=est_test_count, random_state=42
        )

pipe = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    # use lbfgs which supports multiclass; keep max_iter generous for convergence
    ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
])

print(f'Training on {len(X_train)} samples...')
pipe.fit(X_train, y_train)
print('Training done.')
if len(X_test) > 0:
    print('Evaluating...')
    y_pred = pipe.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
else:
    print('No test split available; skipped evaluation.')

joblib.dump(pipe, OUT)
print('Saved quick retrain model to', OUT)
