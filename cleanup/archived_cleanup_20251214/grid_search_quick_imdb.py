"""Quick grid search on IMDB: vary tfidf__max_features and LogisticRegression C.
Saves best pipeline to models/model_grid_best.joblib
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'IMDB_Dataset.csv'
OUT = ROOT / 'models' / 'model_grid_best.joblib'

if not DATA.exists():
    print('IMDB dataset not found at', DATA)
    raise SystemExit(1)

print('Loading IMDB dataset...')
df = pd.read_csv(DATA)
cols = [c.lower() for c in df.columns]
text_col = df.columns[cols.index('review')] if 'review' in cols else (df.columns[0])
label_col = df.columns[cols.index('sentiment')] if 'sentiment' in cols else df.columns[1]
X = df[text_col].astype(str).values
y = df[label_col].astype(str).values

# to keep this quick, sample up to 40000 rows preserving class balance
from sklearn.model_selection import train_test_split
if len(X) > 40000:
    X, _, y, _ = train_test_split(X, y, train_size=40000, stratify=y, random_state=42)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
])

param_grid = {
    'tfidf__max_features': [20000, 10000, 5000],
    'clf__C': [1.0, 0.1, 0.01]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=1)
print('Starting GridSearch (quick)...')
search.fit(X, y)
print('Best params:', search.best_params_)
print('Best CV score:', search.best_score_)

best = search.best_estimator_
joblib.dump(best, OUT)
print('Saved best model to', OUT)
