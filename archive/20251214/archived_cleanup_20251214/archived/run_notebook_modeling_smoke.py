# Smoke test for notebook modeling cell: trains LR and RF on small sample and saves plots
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root on sys.path so `src` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_dataset, preprocess_df, split_data
from src.models import build_tfidf_vectorizer, train_logistic_regression, train_random_forest, evaluate_model, save_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'sample_sentiment.csv'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

print('Loading dataset:', DATA)
df = load_dataset(str(DATA))
df = df.rename(columns={c: c for c in df.columns})
print('Raw shape:', df.shape)

pdf = preprocess_df(df)
train, val, test = split_data(pdf)
print('Split sizes:', len(train), len(val), len(test))

train_sample = train.sample(n=min(20000, len(train)), random_state=42)
val_sample = val.sample(n=min(5000, len(val)), random_state=42)
test_sample = test.sample(n=min(5000, len(test)), random_state=42)

vectorizer = build_tfidf_vectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_sample['text'])
X_val = vectorizer.transform(val_sample['text'])
X_test = vectorizer.transform(test_sample['text'])

lr = train_logistic_regression(X_train, train_sample['label'])
rf = train_random_forest(X_train, train_sample['label'])

best = lr if evaluate_model(lr, X_val, val_sample['label'])['f1'] >= evaluate_model(rf, X_val, val_sample['label'])['f1'] else rf
print('Chosen best (smoke):', type(best))

# Confusion matrix (protected)
cm = confusion_matrix(test_sample['label'], best.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_sample['label']))
try:
    # Manual safe plotting of confusion matrix to avoid locator/label mismatches
    try:
        labels = list(np.unique(test_sample['label']))
        n_labels = len(labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(n_labels))
        ax.set_yticks(np.arange(n_labels))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        # annotate
        for i in range(n_labels):
            for j in range(n_labels):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix (test sample)')
        fig.tight_layout()
        fig.colorbar(im, ax=ax)
        plt.savefig(MODELS_DIR / 'confusion_matrix.png')
        plt.close(fig)
    except Exception as e:
        print('Confusion matrix plotting failed:', e)
    print('Saved confusion matrix')
except Exception as e:
    print('Confusion matrix plotting failed:', e)

# LR feature importance
try:
    if hasattr(best, 'coef_'):
        coefs = best.coef_
        feature_names = vectorizer.get_feature_names_out()
        avg_coefs = np.mean(coefs, axis=0) if coefs.ndim > 1 else coefs
        top_idx = np.argsort(np.abs(avg_coefs))[-40:]
        top_features = feature_names[top_idx]
        top_values = avg_coefs[top_idx]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(range(len(top_features)), top_values, color=['tab:blue' if v>=0 else 'tab:orange' for v in top_values])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_title('Top TF-IDF features (by abs coef)')
        plt.tight_layout()
        plt.savefig(MODELS_DIR / 'top_tfidf_features_lr.png')
        plt.close(fig)
        print('Saved LR top features')
except Exception as e:
    print('LR feature importance failed:', e)

# RF feature importance
try:
    if hasattr(best, 'feature_importances_'):
        fi = best.feature_importances_
        try:
            feature_names = vectorizer.get_feature_names_out()
        except Exception:
            feature_names = np.array([f'f{i}' for i in range(len(fi))])
        top_idx = np.argsort(fi)[-40:]
        top_features = feature_names[top_idx]
        top_values = fi[top_idx]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(range(len(top_features)), top_values, color='tab:green')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_title('Top features by RandomForest importance')
        plt.tight_layout()
        plt.savefig(MODELS_DIR / 'top_features_rf.png')
        plt.close(fig)
        print('Saved RF top features')
except Exception as e:
    print('RF feature importance failed:', e)

# Save pipeline
pipeline = None
try:
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('vectorizer', vectorizer), ('clf', best)])
    save_model(pipeline, str(MODELS_DIR / 'best_model.joblib'))
    print('Saved pipeline')
except Exception as e:
    print('Saving pipeline failed:', e)

print('Smoke run complete')
