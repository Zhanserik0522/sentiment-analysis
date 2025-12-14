"""Compute learning curve on data/IMDB_Dataset.csv using TF-IDF + LogisticRegression.
Saves plot to models/learning_curve_imdb.png and prints summary.
Do NOT use sample_sentiment.csv anywhere.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'IMDB_Dataset.csv'
OUT = ROOT / 'models' / 'learning_curve_imdb.png'

if not DATA.exists():
    print('IMDB dataset not found at', DATA)
    raise SystemExit(1)

# load dataset: try common column names
df = pd.read_csv(DATA)
cols = [c.lower() for c in df.columns]
if 'review' in cols:
    text_col = df.columns[cols.index('review')]
elif 'text' in cols:
    text_col = df.columns[cols.index('text')]
else:
    text_col = df.columns[0]

if 'sentiment' in cols:
    label_col = df.columns[cols.index('sentiment')]
elif 'label' in cols:
    label_col = df.columns[cols.index('label')]
else:
    label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

X = df[text_col].astype(str).values
y = df[label_col].astype(str).values

# define pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes = np.linspace(0.1, 1.0, 5)
print('Computing learning curve on IMDB (~50k samples). This may take a minute...')
train_sizes_abs, train_scores, test_scores = learning_curve(pipe, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve (TF-IDF + LogisticRegression) on IMDB')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT)
print('Saved learning curve to', OUT)
for n, tr_mean, tr_std, te_mean, te_std in zip(
    train_sizes_abs, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
):
    print(f'n={int(n)}: train={tr_mean:.4f} (+/- {tr_std:.4f}), val={te_mean:.4f} (+/- {te_std:.4f})')
