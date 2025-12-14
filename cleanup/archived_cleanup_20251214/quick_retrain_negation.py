"""Quick retrain using negation-aware preprocessing.
Saves model to models/model_quick_retrained_negation.joblib
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.preprocessing import clean_text_negation

DATA = ROOT / 'data' / 'sample_sentiment.csv'
OUT = ROOT / 'models' / 'model_quick_retrained_negation.joblib'

if not DATA.exists():
    print('Sample data not found at', DATA)
    raise SystemExit(1)

df = pd.read_csv(DATA)
# infer columns
cols = [c.lower() for c in df.columns]
text_col = df.columns[cols.index('text')] if 'text' in cols else (df.columns[cols.index('review')] if 'review' in cols else df.columns[0])
label_col = df.columns[cols.index('label')] if 'label' in cols else (df.columns[cols.index('sentiment')] if 'sentiment' in cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0]))

X = df[text_col].fillna('').astype(str).tolist()
y = df[label_col].astype(str).tolist()

# apply negation-aware cleaning
X_clean = [clean_text_negation(t) for t in X]

# simple robust split logic similar to quick_retrain_sample
from collections import Counter
n = len(X_clean)
label_counts = Counter(y)
n_classes = len(label_counts)

if n < 2:
    X_train, y_train = X_clean, y
    X_test, y_test = [], []
else:
    test_frac = 0.2
    est_test_count = max(1, int(test_frac * n))
    can_stratify = (n_classes > 1 and min(label_counts.values()) >= 2 and est_test_count >= n_classes)
    from sklearn.model_selection import train_test_split
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=test_frac, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=est_test_count, random_state=42)

pipe = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
])

print(f'Training (negation) on {len(X_train)} samples...')
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
print('Saved negation-aware model to', OUT)
