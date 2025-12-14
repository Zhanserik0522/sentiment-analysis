import joblib
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
# ensure project root is on sys.path so `src` is importable when running scripts
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.preprocessing import load_dataset, preprocess_df, split_data
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

PROJECT = Path('.').resolve()
MODELS_DIR = PROJECT / 'models' / 'full_search'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset (prefer full IMDB if available)
imdb_path = PROJECT / 'data' / 'IMDB_Dataset.csv'
sample_path = PROJECT / 'data' / 'sample_sentiment.csv'
if imdb_path.exists():
    df = load_dataset(str(imdb_path))
    if 'review' in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'review':'text', 'sentiment':'label'})
else:
    df = load_dataset(str(sample_path))

pdf = preprocess_df(df)
train, val, test = split_data(pdf)

# Load best LR pipeline
lr_path = MODELS_DIR / 'best_model_lr_randomsearch.joblib'
if not lr_path.exists():
    print('LR artifact not found at', lr_path)
    raise SystemExit(1)

print('Loading', lr_path)
lr_pipe = joblib.load(lr_path)
# Combine train+val
combined = pd.concat([train, val]).reset_index(drop=True)
X_combined = combined['text']
y_combined = combined['label']
X_test = test['text']
y_test = test['label']
# Refit pipeline on combined
print('Re-fitting pipeline on train+val...')
lr_pipe.fit(X_combined, y_combined)
# Evaluate on test
y_pred = lr_pipe.predict(X_test)
print('Test accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = list(np.unique(y_test))
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.title('Confusion matrix — LR final')
plt.tight_layout()
cm_path = MODELS_DIR / 'best_lr_confusion_matrix.png'
fig.colorbar(im, ax=ax)
fig.savefig(cm_path)
plt.close(fig)
print('Saved', cm_path)
# Feature importance (coefficients)
try:
    vectorizer = lr_pipe.named_steps.get('vectorizer')
    clf = lr_pipe.named_steps.get('clf')
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_
    avg_coefs = np.mean(coefs, axis=0) if coefs.ndim>1 else coefs
    top_idx = np.argsort(avg_coefs)[-40:]
    top_features = feature_names[top_idx]
    top_values = avg_coefs[top_idx]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(range(len(top_features)), top_values, color=['tab:blue' if v>=0 else 'tab:orange' for v in top_values])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_title('Top TF-IDF features (LR)')
    plt.tight_layout()
    feat_path = MODELS_DIR / 'best_lr_top_features.png'
    fig.savefig(feat_path)
    plt.close(fig)
    print('Saved', feat_path)
except Exception as e:
    print('Failed to save top features:', e)
# Save final pipeline
final_path = MODELS_DIR / 'best_model_lr_final.joblib'
joblib.dump(lr_pipe, final_path)
print('Saved final pipeline to', final_path)
