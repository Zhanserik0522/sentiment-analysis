"""Calibrate the grid-searched best model using IMDB data and CalibratedClassifierCV.
Saves calibrated model to models/model_grid_best_calibrated.joblib
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'IMDB_Dataset.csv'
IN_MODEL = ROOT / 'models' / 'model_grid_best.joblib'
OUT = ROOT / 'models' / 'model_grid_best_calibrated.joblib'

if not DATA.exists():
    print('IMDB dataset not found at', DATA)
    raise SystemExit(1)
if not IN_MODEL.exists():
    print('Input model not found at', IN_MODEL)
    raise SystemExit(1)

print('Loading IMDB dataset...')
df = pd.read_csv(DATA)
cols = [c.lower() for c in df.columns]
text_col = df.columns[cols.index('review')] if 'review' in cols else df.columns[0]
label_col = df.columns[cols.index('sentiment')] if 'sentiment' in cols else df.columns[1]
X = df[text_col].astype(str).values
y = df[label_col].astype(str).values

# limit to 40000 samples to speed up calibration, stratified
if len(X) > 40000:
    X, _, y, _ = train_test_split(X, y, train_size=40000, stratify=y, random_state=42)

print('Loading base estimator...')
base = joblib.load(IN_MODEL)

print('Fitting CalibratedClassifierCV (sigmoid, cv=3) -- this will refit the model inside the calibrator')
cal = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3)
cal.fit(X, y)

joblib.dump(cal, OUT)
print('Saved calibrated model to', OUT)
